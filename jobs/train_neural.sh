#!/bin/bash
#SBATCH --job-name=nli-neural
#SBATCH --partition=gpu_a100_il
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=konrad-rudolf.brueggemann@student.uni-tuebingen.de

cd "$SLURM_SUBMIT_DIR"
module load devel/cuda/12.8
module load devel/python/3.13.3-llvm-19.1

uv run python train_neural.py
