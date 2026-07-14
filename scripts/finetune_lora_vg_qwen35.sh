#!/usr/bin/env bash
#SBATCH --job-name=qwen35-vg-lora
#SBATCH --partition=kolyoz-cuda          # H200/CUDA13; recot-train (cu128) runs here AND on palamut-cuda (idle A100s) — swap if you want faster scheduling
#SBATCH --exclude=kolyoz10,kolyoz11,kolyoz19   # corrupt/broken GPUs: 11 (bad CUDA init), 19 (corrupt), 10 (no device handle)
#SBATCH --account=ogam6
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16               # kolyoz rule: 16 CPUs per GPU
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --output=/arf/scratch/aalatan/Re-CoT/Qwen-VL-Series-Finetune/output/qwen35-vg-lora-%j.out
#SBATCH --error=/arf/scratch/aalatan/Re-CoT/Qwen-VL-Series-Finetune/output/qwen35-vg-lora-%j.err

set -euo pipefail

# ---- paths ----
REPO=/arf/scratch/aalatan/Re-CoT/Qwen-VL-Series-Finetune
ENV=/arf/home/aalatan/mert/envs/recot-train
MODEL=/arf/scratch/aalatan/Re-CoT/models/Qwen3.5-0.8B
DATA=/arf/scratch/aalatan/Re-CoT/datasets/VG_sft/vg_sft.json
IMAGE_FOLDER=/arf/scratch/aalatan/Re-CoT/datasets/VG_sft     # image field is "images/<name>", resolved under here
OUTPUT_DIR=$REPO/output/qwen35-vg-lora

cd "$REPO"
export PYTHONPATH=src:${PYTHONPATH:-}
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

# ---- batch math (single GPU; a 0.8B LoRA fits easily on one H200/A100) ----
GLOBAL_BATCH_SIZE=128
BATCH_PER_DEVICE=4
NUM_DEVICES=1
GRAD_ACCUM_STEPS=$((GLOBAL_BATCH_SIZE / (BATCH_PER_DEVICE * NUM_DEVICES)))

echo "### node=$(hostname) date=$(date)"
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader

# No deepspeed (dropped from env). torchrun handles single-/multi-GPU; Qwen3.5 -> SDPA (--disable_flash_attn2 True).
"$ENV/bin/torchrun" --standalone --nproc_per_node=$NUM_DEVICES src/train/train_sft.py \
    --use_liger_kernel True \
    --lora_enable True \
    --vision_lora True \
    --use_dora False \
    --lora_namespan_exclude "['lm_head', 'embed_tokens']" \
    --lora_rank 32 \
    --lora_alpha 64 \
    --lora_dropout 0.05 \
    --num_lora_modules -1 \
    --model_id "$MODEL" \
    --data_path "$DATA" \
    --image_folder "$IMAGE_FOLDER" \
    --remove_unused_columns False \
    --freeze_vision_tower True \
    --freeze_llm True \
    --freeze_merger True \
    --bf16 True \
    --fp16 False \
    --disable_flash_attn2 True \
    --output_dir "$OUTPUT_DIR" \
    --num_train_epochs 1 \
    --per_device_train_batch_size $BATCH_PER_DEVICE \
    --gradient_accumulation_steps $GRAD_ACCUM_STEPS \
    --image_min_pixels $((256 * 28 * 28)) \
    --image_max_pixels $((1280 * 28 * 28)) \
    --learning_rate 2e-4 \
    --weight_decay 0.1 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --gradient_checkpointing False \
    --report_to tensorboard \
    --lazy_preprocess True \
    --save_strategy "steps" \
    --save_steps 200 \
    --save_total_limit 10 \
    --dataloader_num_workers 4

echo "### DONE date=$(date)"
