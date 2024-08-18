export HF_ENDPOINT=https://hf-mirror.com
deepspeed --include localhost:0,1,2,3,4 trainer.py --mm_projector_lr 2e-5 \
    --deepspeed ./scripts/zero3.json \
    --data_path data/llm/LLaVA-Finetune/llava_v1_5_mix665k.json \
    --image_folder data/llm/LLaVA-Finetune \
    --output_dir data/checkpoints/finetune \
    --pretrain_dir data/checkpoints/pretrain \
    --learning_rate 2e-4
