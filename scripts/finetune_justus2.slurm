#!/bin/bash
#SBATCH --job-name=mace_finetune
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32gb
#SBATCH --time=02:00:00
#SBATCH --output=logs/mace_finetune_%j.out
#SBATCH --error=logs/mace_finetune_%j.err

module purge
unset PYTHONPATH

conda activate pentacene-al

cd $HOME/pentacene-gb-active-learning
mkdir -p logs

python scripts/train_finetune_model.py

echo "Job finished on $(hostname) at $(date)"