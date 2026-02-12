#!/bin/bash
#SBATCH --job-name=nli-probe
#SBATCH --partition=gpu_a100_il
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=konrad-rudolf.brueggemann@student.uni-tuebingen.de
#SBATCH --array=0-11

cd "$SLURM_SUBMIT_DIR"
module load devel/python/3.13.3-llvm-19.1

MODELS=(
    "google-bert/bert-base-chinese"
    "google-bert/bert-base-uncased"
    "google-bert/bert-large-uncased"
    "google-bert/bert-base-multilingual-cased"
    "hfl/chinese-roberta-wwm-ext"
    "voidful/albert_chinese_base"
    "shibing624/text2vec-base-chinese"
    "jinaai/jina-embeddings-v2-base-zh"
    "jinaai/jina-embeddings-v3"
    "Qwen/Qwen3-Embedding-0.6B"
    "Qwen/Qwen3-Embedding-4B"
    "DMetaSoul/Dmeta-embedding-zh-small"
)

# Larger models need smaller batch sizes
BATCH_SIZES=(32 32 8 32 32 32 32 32 16 16 4 32)

MODEL="${MODELS[$SLURM_ARRAY_TASK_ID]}"
BS="${BATCH_SIZES[$SLURM_ARRAY_TASK_ID]}"
echo "=== Probe: $MODEL (batch_size=$BS, array index $SLURM_ARRAY_TASK_ID) ==="

uv run python train_probe.py --model "$MODEL" --batch-size "$BS"
