#!/usr/bin/env bash
#SBATCH --job-name=qwen35-cot-merge
#SBATCH --partition=kolyoz-cuda
#SBATCH --exclude=kolyoz10,kolyoz11,kolyoz13,kolyoz14,kolyoz19,kolyoz24   # corrupt GPUs: CUDA init fails ("CUDA unknown error") or no device handle
#SBATCH --requeue
#SBATCH --account=ogam6
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --mem=128G
#SBATCH --time=01:00:00
#SBATCH --output=/arf/scratch/aalatan/Re-CoT/Qwen-VL-Series-Finetune/output/qwen35-cot-merge-%j.out
#SBATCH --error=/arf/scratch/aalatan/Re-CoT/Qwen-VL-Series-Finetune/output/qwen35-cot-merge-%j.err

# Merges the stage-2 CoT LoRA adapters for evaluation.
#
# IMPORTANT: the CoT adapter was trained on top of the *stage-1 merged* model, so the
# merge base is qwen35-<task>-merged, NOT the original Qwen3.5-0.8B base. Merging onto
# the raw base would silently throw away all the stage-1 SFT learning.
#
# Runs on CPU (merge_lora_weights.py uses device_map='cpu'); submitted via SLURM because
# the login node's 300s CPU limit kills it partway.
#
# Usage: sbatch merge_lora_cot.sh [task ...]     (default: cls vqa vg; skips unfinished)

set -euo pipefail

REPO=/arf/scratch/aalatan/Re-CoT/Qwen-VL-Series-Finetune
ENV=/arf/home/aalatan/mert/envs/recot-train
BASE_MODEL=/arf/scratch/aalatan/Re-CoT/models/Qwen3.5-0.8B   # only for the processor/tokenizer files

cd "$REPO"
export PYTHONPATH=src:${PYTHONPATH:-}
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

TASKS=("${@:-}")
[ -z "${TASKS[*]}" ] && TASKS=(cls vqa vg)

echo "### node=$(hostname) date=$(date)"

for TASK in "${TASKS[@]}"; do
    ADAPTER=$REPO/output/qwen35-cot-${TASK}-lora
    STAGE1=$REPO/output/qwen35-${TASK}-merged      # merge base: stage-1 SFT model
    MERGED=$REPO/output/qwen35-cot-${TASK}-merged

    if [ ! -f "$ADAPTER/adapter_model.safetensors" ]; then
        echo "### SKIP ${TASK}: no final adapter in ${ADAPTER} (still training?)"
        continue
    fi

    echo
    echo "=============================================="
    echo "### merging cot-${TASK}"
    echo "###   adapter: ${ADAPTER}"
    echo "###   base   : ${STAGE1}   (stage-1 merged, NOT the raw base)"
    echo "###   out    : ${MERGED}"
    echo "=============================================="

    "$ENV/bin/python" src/merge_lora_weights.py \
        --model-path "$ADAPTER" \
        --model-base "$STAGE1" \
        --save-model-path "$MERGED" \
        --safe-serialization

    # processor.save_pretrained() omits the image-processor / BPE files that vLLM needs
    # to build the processor. They are unchanged by a LoRA merge, so copy them across.
    for f in preprocessor_config.json video_preprocessor_config.json merges.txt vocab.json; do
        cp -n "$BASE_MODEL/$f" "$MERGED/$f"
    done

    echo "### ${TASK} merged -> ${MERGED}"
done

echo
echo "### ALL MERGES DONE date=$(date)"
