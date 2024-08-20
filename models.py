import torch, os
import vision_models, language_models, rotary, constants
from torch import nn
from galois_runtime import grutils

class Model(nn.Module):
    def __init__(self, model_args, dtype, bnb_args):
        super().__init__()
        self.vision = vision_models.VisionModel(model_args.vision_model, dtype)
        self.language = language_models.LanguageModel(model_args.language_model, dtype, bnb_args)
        self.pe = rotary.RotaryPositionEncoder(self.language.hidden_size, self.language.num_attention_heads)
        self.mm_projector = self._build_projector()
        self.dtype = dtype
        self._keys_to_ignore_on_save = None

    @property
    def config(self): return self.language.module.config

    def _build_projector(self):
        return nn.Sequential(
            nn.Linear(self.vision.hidden_size, self.language.hidden_size),
            nn.GELU(),
            nn.Linear(self.language.hidden_size, self.language.hidden_size))

    def prepare_for_training(self, train_args):
        self.language.prepare_for_training(train_args)

    def merge(self): self.language.merge()

    def load_model(self, ckpt_path):
        ckpt = torch.load(ckpt_path)
        self.mm_projector.load_state_dict(ckpt['projector'])

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        self.language.module.gradient_checkpointing_enable(gradient_checkpointing_kwargs)

    def forward(self, images, input_ids, labels, attention_mask):
        args = self.prepare_inputs_labels_for_multimodal(images, input_ids, attention_mask, labels)
        return self.language(**args)

    def encode_vision(self, images):
        vision_outputs = self.vision(images)
        vision_states = self.mm_projector(vision_outputs['hidden_states'].to(self.dtype))
        vision_states = self.pe.apply_rotary_v(vision_outputs['position'], vision_states)
        vision_states = vision_states.split(vision_outputs['lengths'])
        return vision_states

    def prepare_inputs_labels_for_multimodal(self, images, input_ids, attention_mask, labels=None):
        vision_states = self.encode_vision(images)
        input_ids = [x[y] for x, y in zip(input_ids, attention_mask)]
        labels = [x[y] for x, y in zip(labels, attention_mask)]
        new_input_embs = []
        new_labels = []
        for k in range(len(images)):
            _input_ids = input_ids[k]
            image_token_pos, = torch.where(_input_ids.eq(self.language.tokenizer.image_token_id))
            _input_ids[image_token_pos] = self.language.tokenizer.pad_token_id
            embeds = self.language.embed_tokens(_input_ids)
            offset = 0
            _vision_states = [vision_states[k]]
            _labels = labels[k]
            combined_embs, combined_labels = [], []
            for image_pos, image_embs in zip(image_token_pos, _vision_states):
                combined_embs.append(embeds[offset:image_pos])
                combined_embs.append(image_embs)
                combined_labels.append(_labels[offset:image_pos])
                combined_labels.append(torch.full((image_embs.shape[0],), constants.IGNORE_INDEX, device=image_embs.device))
                offset = image_pos + 1
            if offset < _input_ids.shape[0]:
                combined_embs.append(embeds[offset:])
                combined_labels.append(_labels[offset:])
            combined_embs = torch.cat(combined_embs, 0)
            combined_labels = torch.cat(combined_labels, 0)
            new_input_embs.append(combined_embs[:self.language.tokenizer.model_max_length])
            new_labels.append(combined_labels[:self.language.tokenizer.model_max_length])
        max_len = max(x.shape[0] for x in new_input_embs)
        new_labels = torch.nn.utils.rnn.pad_sequence(new_labels, True, constants.IGNORE_INDEX)
        lengths = torch.tensor([x.shape[0] for x in new_input_embs], device=attention_mask.device)
        attention_mask = grutils.sequence_mask(lengths)
        #position_ids = torch.arange(max_len, device=attention_mask.device)[None, :].repeat(len(images), 1)
        new_input_embs = torch.nn.utils.rnn.pad_sequence(new_input_embs, True, 0)
        assert max_len <= self.language.tokenizer.model_max_length
        return {
            'input_embeds': new_input_embs,
            #'position_ids': position_ids,
            'attention_mask': attention_mask,
            'labels': new_labels
        }

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        #image_sizes = kwargs.pop("image_sizes", None)
        inputs = self.language.module.prepare_inputs_for_generation(input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs)
        if images is not None: inputs['images'] = images
        #if image_sizes is not None: inputs['image_sizes'] = image_sizes
        return inputs

    @torch.inference_mode()
    def generate(self, inputs, images, **kwargs):
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs: raise NotImplementedError("`inputs_embeds` is not supported")
        prepared = self.prepare_inputs_labels_for_multimodal(images, inputs, attention_mask)
        return self.language.module.generate(
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=prepared['input_embeds'],
            **kwargs)

    def save(self, folder):
        self.vision.save(os.path.join(folder, 'vision'))
        self.language.save(os.path.join(folder, 'language'))
        ckpt_path = os.path.join(folder, 'state.ckpt')
        state_dict = {
            'mm_projector': self.mm_projector.state_dict()
        }
        torch.save(state_dict, ckpt_path)

    def load(self, folder):
        ckpt_path = os.path.join(folder, 'state.ckpt')
        ckpt = torch.load(ckpt_path)
        self.mm_projector.load_state_dict(ckpt['mm_projector'])

def save(training_folder, output_folder, model_args, dtype):
    import os, transformers, safetensors
    train_args = torch.load(os.path.join(training_folder, transformers.trainer.TRAINING_ARGS_NAME))
    model = Model(model_args, dtype, {})
    model.prepare_for_training(train_args)
    delta_path = os.path.join(training_folder, transformers.trainer.SAFE_WEIGHTS_NAME)
    state_dict = safetensors.torch.load_file(delta_path, device='cpu')
    model.load_state_dict(state_dict, strict=False)
    model.merge()
    model.save(output_folder)

def load(folder, dtype):
    import types
    folder = os.path.abspath(folder)
    vision_folder = os.path.join(folder, 'vision')
    language_folder = os.path.join(folder, 'language')
    model = Model(types.SimpleNamespace(vision_model=vision_folder, language_model=language_folder), dtype, {})
    model.load(folder)
    return model