export HF_ENDPOINT=https://hf-mirror.com
deepspeed --include localhost:0,1,2,3,4,6,7 trainer.py \
    --deepspeed ./scripts/zero2.json \
    --data_path data/llm/LLaVA-Finetune/llava_v1_5_mix665k.json \
    --image_folder data/llm/LLaVA-Finetune \
    --output_dir data/checkpoints/finetune \
    --pretrain_dir data/checkpoints/pretrain \
    --mm_projector_lr 2e-5 \
    --learning_rate 2e-4 \
    --bits 4
