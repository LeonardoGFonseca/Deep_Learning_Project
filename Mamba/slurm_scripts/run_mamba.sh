#!/bin/bash
#SBATCH --job-name=mamba_30
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=/mnt/storage/admindi/home/dl011/slurm_outputs/mamba-%j.out
#SBATCH --error=/mnt/storage/admindi/home/dl011/slurm_outputs/mamba-%j.err

source ~/miniconda3/etc/profile.d/conda.sh
conda activate torch-env

cd /mnt/storage/admindi/home/dl011/project/Deep_Learning_Mamba
export PYTHONPATH=$(pwd)

python -u src/training/sweep.py \
    --model_type mamba \
    --h5_path data/processed/hd5f/amr_features.h5 \
    --epochs 50 \
    --patience 10 \
    --strain_start 0 \
    --strain_end 30 \
    --output_dir pipeline_logs/mamba_sweep
