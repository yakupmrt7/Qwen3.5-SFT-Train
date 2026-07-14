#!/usr/bin/env bash
#SBATCH --job-name=qwen35-merge-lora
#SBATCH --partition=kolyoz-cuda
#SBATCH --exclude=kolyoz10,kolyoz11,kolyoz19   # corrupt/broken GPUs: 11 (bad CUDA init), 19 (corrupt), 10 (no device handle)
#SBATCH --account=ogam6
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16               # kolyoz rule: 16 CPUs per GPU
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --mem=128G
#SBATCH --time=01:00:00
#SBATCH --output=/arf/scratch/aalatan/Re-CoT/Qwen-VL-Series-Finetune/output/qwen35-merge-%j.out
#SBATCH --error=/arf/scratch/aalatan/Re-CoT/Qwen-VL-Series-Finetune/output/qwen35-merge-%j.err

# Merges the trained LoRA adapters into the base model for evaluation.
# Merge itself runs on CPU (see merge_lora_weights.py device_map='cpu'); the job is
# submitted to SLURM because the login node's 300s CPU limit kills it partway.

set -euo pipefail

REPO=/arf/scratch/aalatan/Re-CoT/Qwen-VL-Series-Finetune
ENV=/arf/home/aalatan/mert/envs/recot-train
MODEL_BASE=/arf/scratch/aalatan/Re-CoT/models/Qwen3.5-0.8B

cd "$REPO"
export PYTHONPATH=src:${PYTHONPATH:-}
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

echo "### node=$(hostname) date=$(date)"

for TASK in cls vqa vg; do
    ADAPTER=$REPO/output/qwen35-${TASK}-lora
    MERGED=$REPO/output/qwen35-${TASK}-merged

    echo
    echo "=============================================="
    echo "### merging ${TASK}: ${ADAPTER} -> ${MERGED}"
    echo "=============================================="

    "$ENV/bin/python" src/merge_lora_weights.py \
        --model-path "$ADAPTER" \
        --model-base "$MODEL_BASE" \
        --save-model-path "$MERGED" \
        --safe-serialization

    echo "### ${TASK} merged -> ${MERGED}"
done

echo
echo "### ALL MERGES DONE date=$(date)"
