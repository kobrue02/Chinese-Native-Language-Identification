#!/bin/bash
#SBATCH --job-name=nli-neural
#SBATCH --output=results/slurm_neural_%j.out
#SBATCH --error=results/slurm_neural_%j.err
#SBATCH --partition=gpu
#SBATCH --partition=gpu_a100_il
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00

cd "$SLURM_SUBMIT_DIR"
module load devel/cuda/12.8
module load devel/python/3.13.3-llvm-19.1

uv run python train_neural.py
