#!/bin/bash
#SBATCH --job-name=nli-svm
#SBATCH --output=results/slurm_svm_%j.out
#SBATCH --error=results/slurm_svm_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00

cd "$SLURM_SUBMIT_DIR"

uv run python train_svm.py
