export HF_ENDPOINT=https://hf-mirror.com
deepspeed --include localhost:0,1,2,3,4 trainer.py \
    --deepspeed ./scripts/zero2.json \
    --data_path data/llm/LLaVA-Pretrain/blip_laion_cc_sbu_558k.json \
    --image_folder data/llm/LLaVA-Pretrain/images \
    --output_dir data/checkpoints/pretrain \
    --mm_projector_lr 1e-3
