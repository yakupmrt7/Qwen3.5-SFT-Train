#!/usr/bin/env bash
#SBATCH --job-name=qwen35-cot-vqa
#SBATCH --partition=kolyoz-cuda
#SBATCH --exclude=kolyoz10,kolyoz11,kolyoz13,kolyoz14,kolyoz19,kolyoz24   # corrupt GPUs: CUDA init fails ("CUDA unknown error") or no device handle
#SBATCH --requeue                                                         # allow self-requeue when we land on a corrupt GPU
#SBATCH --account=ogam6
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16               # kolyoz rule: 16 CPUs per GPU
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --mem=128G
#SBATCH --time=24:00:00                  # CoT seqs are ~2x SFT
#SBATCH --output=/arf/scratch/aalatan/Re-CoT/Qwen-VL-Series-Finetune/output/qwen35-cot-vqa-%j.out
#SBATCH --error=/arf/scratch/aalatan/Re-CoT/Qwen-VL-Series-Finetune/output/qwen35-cot-vqa-%j.err

# Stage 2: chain-of-thought training on top of the merged VQA SFT model.
# A fresh LoRA adapter is trained over qwen35-vqa-merged (base weights frozen).

set -euo pipefail

# ---- paths ----
REPO=/arf/scratch/aalatan/Re-CoT/Qwen-VL-Series-Finetune
ENV=/arf/home/aalatan/mert/envs/recot-train
MODEL=$REPO/output/qwen35-vqa-merged                                        # stage-1 SFT model, LoRA already merged in
DATA=/arf/scratch/aalatan/Re-CoT/datasets/reasoning/VQA/vqa_sft_reasoning.json
IMAGE_FOLDER=/arf/scratch/aalatan/Re-CoT/datasets/reasoning/VQA/images      # image field is "<DATASET>/<name>", resolved under here
OUTPUT_DIR=$REPO/output/qwen35-cot-vqa-lora

cd "$REPO"
export PYTHONPATH=src:${PYTHONPATH:-}
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

# ---- batch math (single GPU) ----
GLOBAL_BATCH_SIZE=128
BATCH_PER_DEVICE=4
NUM_DEVICES=1
GRAD_ACCUM_STEPS=$((GLOBAL_BATCH_SIZE / (BATCH_PER_DEVICE * NUM_DEVICES)))

echo "### node=$(hostname) date=$(date)"
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader

# Several kolyoz nodes have GPUs that pass nvidia-smi but fail torch's CUDA init.
if ! "$ENV/bin/python" -c "import torch; torch.zeros(1).cuda(); print('### CUDA OK:', torch.cuda.get_device_name(0))"; then
    echo "### CUDA BROKEN on $(hostname) — requeueing onto another node"
    scontrol requeue "$SLURM_JOB_ID"
    sleep 60
    exit 1
fi

# CoT vs stage-1 SFT, and why:
#   lora_rank 64 / alpha 128 - generating long reasoning is a much larger behavioral
#       change than emitting a one-word label; rank 32 was sized for short answers.
#   learning_rate 1e-4       - second-stage refinement on a model that already solves
#       the task; 2e-4 risks trampling the SFT skill while learning the CoT format.
#   vision_lr 2e-5           - keep the already-good visual features nearly fixed while
#       the LLM learns to reason (vision LoRA stays on for spatial grounding).
#   num_train_epochs 1       - single pass over the reasoning data.
# Targets are ~400 tokens (vs ~5 in SFT); truncation is disabled in the dataset, so
# nothing is silently cut. If this OOMs, set --gradient_checkpointing True first.
"$ENV/bin/torchrun" --standalone --nproc_per_node=$NUM_DEVICES src/train/train_sft.py \
    --use_liger_kernel True \
    --lora_enable True \
    --vision_lora True \
    --use_dora False \
    --lora_namespan_exclude "['lm_head', 'embed_tokens']" \
    --lora_rank 64 \
    --lora_alpha 128 \
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
    --learning_rate 1e-4 \
    --vision_lr 2e-5 \
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
