#!/bin/bash
#SBATCH --job-name=mamba_resume
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/mnt/storage/admindi/home/dl011/slurm_outputs/mamba_resume-%j.out
#SBATCH --error=/mnt/storage/admindi/home/dl011/slurm_outputs/mamba_resume-%j.err

# Activate Conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate torch-env

# Ensure correct working directory
cd /mnt/storage/admindi/home/dl011/project/Mamba_Multiclasse
export PYTHONPATH=$(pwd)

H5_PATH="data/processed/hd5f/amr_features.h5"

echo "========================================================"
# Clean up the interrupted Fold 9 directory to avoid loading half-trained weights
rm -rf pipeline_logs/mamba_multiclass/results/mamba_KP1226

# Run Mamba Classifier Multiclass (Folds 9-10)
python -u src/training/sweep.py \
    --model_type mamba \
    --task multiclass \
    --h5_path "$H5_PATH" \
    --epochs 50 \
    --patience 10 \
    --strain_start 8 \
    --strain_end 10 \
    --output_dir pipeline_logs/mamba_multiclass

echo "Mamba resume complete!"
echo "========================================================"
