import torch, transformers, pathlib, os
import models, loader, trutils
from typing import Optional
from dataclasses import dataclass, field

local_rank = None

@dataclass
class ModelArguments:
    vision_model: Optional[str] = field(default="openai/clip-vit-large-patch14-336")
    language_model: Optional[str] = field(default="lmsys/vicuna-7b-v1.5")
    #tune_mm_mlp_adapter: bool = field(default=False)
    #mm_vision_select_layer: Optional[int] = field(default=-2)   # default to the last layer
    #pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    #checkpoint_path: Optional[str] = field(default='data/checkpoints/default.ckpt')
    #mm_projector_type: Optional[str] = field(default='mlp2x_gelu')

@dataclass
class DataArguments:
    data_path: str = field(default='None', metadata={"help": "Path to the training data."})
    lazy_preprocess: bool = True
    #is_multimodal: bool = False
    image_folder: Optional[str] = field(default=None)
    #image_aspect_ratio: str = 'square'

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    #cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    #remove_unused_columns: bool = field(default=False)
    #freeze_mm_mlp_adapter: bool = field(default=False)
    #mpt_attn_impl: Optional[str] = field(default="triton")
    model_max_length: int = field(
        default=4096,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    double_quant: bool = field(
        default=True,
        metadata={"help": "Compress the quantization statistics through double quantization."}
    )
    quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
    )
    bits: int = field(
        default=16,
        metadata={"help": "How many bits to use."}
    )
    remove_unused_columns: bool = False
    lora_enable: bool = False
    lora_r: int = 128
    lora_alpha: int = 256
    lora_dropout: float = 0.05
    lora_weight_path: str = ''
    lora_bias: str = 'none'
    mm_projector_lr: Optional[float] = field(default=None)
    bf16: Optional[bool] = field(default=True)
    group_by_modality_length: bool = field(default=False)
    gradient_checkpointing: bool = field(default=True)
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 8
    save_steps: int = 100
    save_total_limit: int = 1
    num_train_epochs: int = 1
    lr_scheduler_type: str = 'cosine'
    warmup_ratio: float = 0.03
    pretrain_dir: Optional[str] = field(default=None)
    group_by_length: Optional[bool] = True
    report_to: str = 'wandb'
    logging_steps: int = 1
    eval_steps: int = 10

def _make_bnb_args(train_args):
    from transformers import BitsAndBytesConfig
    compute_dtype = torch.float16 if train_args.fp16 else (torch.bfloat16 if train_args.bf16 else torch.float32)
    if train_args.bits in [4, 8]:
        return dict(device_map={'': train_args.device},
                    quantization_config=BitsAndBytesConfig(
                        load_in_4bit=train_args.bits == 4,
                        load_in_8bit=train_args.bits == 8,
                        llm_int8_skip_modules=["mm_projector"],
                        llm_int8_threshold=6.0,
                        llm_int8_has_fp16_weight=False,
                        bnb_4bit_compute_dtype=compute_dtype,
                        bnb_4bit_use_double_quant=train_args.double_quant,
                        bnb_4bit_quant_type=train_args.quant_type # {'fp4', 'nf4'}
                    ))
    else:
        return {}

class Trainer(transformers.Trainer):
    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        #lengths = self.train_dataset.modality_lengths
        if self.args.group_by_length:
            return loader.LengthGroupedSampler(
                batch_size=self.args.train_batch_size,
                world_size=self.args.world_size * self.args.gradient_accumulation_steps,
                datasource=self.train_dataset)
        else:
            return super()._get_train_sampler()

    def create_optimizer(self):
        if self.optimizer is not None: return self.optimizer
        decay_parameters = trutils.get_parameter_names(self.model, transformers.trainer.ALL_LAYERNORM_LAYERS)
        decay_parameters = [name for name in decay_parameters if 'bias' not in name]
        if self.args.mm_projector_lr is not None:
            projector_parameters = [name for name, _ in self.model.named_parameters() if 'mm_projector' in name]
        else:
            projector_parameters = []
        groups = {
            (False, False): {'params': [], 'weight_decay': 0.0},
            (False, True): {'params': [], 'weight_decay': 0.0, 'lr': self.args.mm_projector_lr},
            (True, False): {'params': [], 'weight_decay': self.args.weight_decay},
            (True, True): {'params': [], 'weight_decay': self.args.weight_decay, 'lr': self.args.mm_projector_lr},
        }
        for n, p in self.model.named_parameters():
            if not p.requires_grad: continue
            cond_a = n in decay_parameters
            cond_b = n in projector_parameters
            groups[(cond_a, cond_b)]['params'].append(p)
        parameter_groups = [x for x in groups.values() if x['params']]
        optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)
        self.optimizer = optimizer_cls(parameter_groups, **optimizer_kwargs)
        assert optimizer_cls.__name__ != "Adam8bit"
        return self.optimizer

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        state_dict = {n: p for n, p in self.model.named_parameters() if p.requires_grad}
        super()._save(output_dir, state_dict)

    def _save_checkpoint(self, model, trial, metrics=None):
        super()._save_checkpoint(model, trial, metrics)

def train():
    global local_rank
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, train_args = parser.parse_args_into_dataclasses()
    bnb_args = _make_bnb_args(train_args)
    model_dtype = (torch.bfloat16 if train_args.bf16 else None)
    model = models.Model(model_args, dtype=model_dtype, bnb_args=bnb_args)
    model.prepare_for_training(train_args)
    dataset = loader.SupervisedDataset(model.vision.processor, model.language.tokenizer, data_args.image_folder, data_args.data_path)
    collator = loader.SupervisedDataCollator(model.language.tokenizer)
    trainer = Trainer(model=model, args=train_args, train_dataset=dataset, data_collator=collator)
    if list(pathlib.Path(train_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        if train_args.pretrain_dir:
            trainer._load_from_checkpoint(transformers.trainer_utils.get_last_checkpoint(train_args.pretrain_dir))
        trainer.train()
    trainer.save_state()
    models.save(
        transformers.trainer_utils.get_last_checkpoint(train_args.output_dir),
        os.path.join(train_args.output_dir, 'complete'),
        model_args, model_dtype)

if __name__ == '__main__':
    train()