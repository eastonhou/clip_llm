import torch
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

    def _build_projector(self):
        return nn.Sequential(
            nn.Linear(self.vision.hidden_size, self.language.hidden_size),
            nn.GELU(),
            nn.Linear(self.language.hidden_size, self.language.hidden_size))

    def prepare_for_training(self, train_args, bnb_args):
        self.language.prepare_for_training(train_args, bnb_args)

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
        vision_states = self.mm_projector(vision_outputs['hidden_states'])
        vision_states = self.pe.apply_rotary_v(vision_outputs['position'], vision_states)
        vision_states = [x[y] for x, y in zip(vision_states, vision_outputs['mask'])]
        return vision_states

    def prepare_inputs_labels_for_multimodal(self, images, input_ids, attention_mask, labels):
        vision_states = self.encode_vision(images)
        #position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
        input_ids = [x[y] for x, y in zip(input_ids, attention_mask)]
        labels = [x[y] for x, y in zip(labels, attention_mask)]
        new_input_embs = []
        new_labels = []
        for k in range(len(images)):
            _input_ids = input_ids[k]
            image_token_pos, = torch.where(_input_ids.eq(self.language.tokenizer.image_token_id))
            _input_ids[image_token_pos] = self.language.tokenizer.pad_token_id
            embeds = self.language.embed_tokens(_input_ids)
            #split_input_ids = input_ids[k].split(self.language.tokenizer.image_token_id)
            #split_sizes = [len(x) for x in split_input_ids]
            #split_labels = grutils.group(labels[k], split_sizes)
            #embeds = torch.split(embeds, split_sizes, dim=0)
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
            new_input_embs.append(combined_embs)
            new_labels.append(combined_labels)
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
