import torch, transformers
import torch.nn as nn

class LanguageModel(nn.Module):
    def __init__(self, model_name, dtype, bnb_args):
        super().__init__()
        self.module = transformers.LlamaForCausalLM.from_pretrained(
            model_name,
            attn_implementation='flash_attention_2',
            torch_dtype=dtype,
            **bnb_args)
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_name,
            model_max_length=self.module.config.max_position_embeddings,
            padding_side='right',
            use_fast=False)
        self.tokenizer.add_tokens(['<image>'], special_tokens=True)
        self.tokenizer.image_token_id = self.tokenizer.convert_tokens_to_ids('<image>')

    @property
    def hidden_size(self): return self.module.config.hidden_size
    @property
    def num_attention_heads(self): return self.module.config.num_attention_heads

    def prepare_for_training(self, train_args):
        from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
        self.tokenizer.model_max_length = train_args.model_max_length
        if train_args.bits in [4, 8]:
            self.module.config.torch_dtype = torch.float16 if train_args.fp16 else (torch.bfloat16 if train_args.bf16 else torch.float32)
            self.module = prepare_model_for_kbit_training(self.module, use_gradient_checkpointing=train_args.gradient_checkpointing)
        if train_args.gradient_checkpointing: self.module.enable_input_require_grads()
        if train_args.lora_enable:
            lora_config = LoraConfig(
                r=train_args.lora_r,
                lora_alpha=train_args.lora_alpha,
                target_modules=self.find_all_linear_names(),
                lora_dropout=train_args.lora_dropout,
                bias=train_args.lora_bias,
                task_type='CAUSAL_LM')
            self.module = get_peft_model(self.module, lora_config)
        else:
            self.requires_grad_(False)

    def merge(self):
        from peft import peft_model
        if isinstance(self.module, peft_model.PeftModel):
            self.module = self.module.merge_and_unload()

    def find_all_linear_names(self):
        lora_module_names = set()
        for name, module in self.named_modules():
            if isinstance(module, torch.nn.Linear):
                names = name.split('.')
                lora_module_names.add(names[0] if len(names) == 1 else names[-1])
        if 'lm_head' in lora_module_names: # needed for 16-bit
            lora_module_names.remove('lm_head')
        return list(lora_module_names)

    def embed_tokens(self, tokens):
        if hasattr(self.module.base_model, 'embed_tokens'): return self.module.base_model.embed_tokens(tokens)
        else: return self.module.base_model.model.model.embed_tokens(tokens)

    def forward(self, input_embeds, attention_mask, labels):
        outputs = self.module(inputs_embeds=input_embeds,
                              position_ids=None,
                              attention_mask=attention_mask,
                              labels=labels)
        return outputs

    def save(self, folder):
        self.module.generation_config.do_sample = True
        self.tokenizer.save_pretrained(folder)
        self.module.save_pretrained(folder)