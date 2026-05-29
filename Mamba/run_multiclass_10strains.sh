#!/bin/bash
#SBATCH --job-name=mamba_multi_10
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/mnt/storage/admindi/home/dl011/slurm_outputs/mamba_multi-%j.out
#SBATCH --error=/mnt/storage/admindi/home/dl011/slurm_outputs/mamba_multi-%j.err

# Activate Conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate torch-env

# Ensure correct working directory
cd /mnt/storage/admindi/home/dl011/project/Mamba_Multiclasse
export PYTHONPATH=$(pwd)

H5_PATH="data/processed/hd5f/amr_features.h5"

echo "========================================================"
echo "Starting Multiclass Leave-One-Strain-Out (10 Folds)"
echo "Job ID: $SLURM_JOB_ID"
echo "Start Time: $(date)"
echo "Using HDF5 dataset: $H5_PATH"
echo "========================================================"

# Clean up any existing logs from these directories to avoid confusion
rm -rf pipeline_logs/gap_multiclass
rm -rf pipeline_logs/mamba_multiclass

# Run 1: GAP Baseline Multiclass (Folds 1-10)
echo -e "\n🚀 Run 1: GAP Baseline Multiclass (10 Folds)"
python -u src/training/sweep.py \
    --model_type gap \
    --task multiclass \
    --h5_path "$H5_PATH" \
    --epochs 50 \
    --patience 10 \
    --strain_start 0 \
    --strain_end 10 \
    --output_dir pipeline_logs/gap_multiclass

# Run 2: Mamba Classifier Multiclass (Folds 1-10)
echo -e "\n🚀 Run 2: Mamba Classifier Multiclass (10 Folds)"
python -u src/training/sweep.py \
    --model_type mamba \
    --task multiclass \
    --h5_path "$H5_PATH" \
    --epochs 50 \
    --patience 10 \
    --strain_start 0 \
    --strain_end 10 \
    --output_dir pipeline_logs/mamba_multiclass

echo "Multiclass sweep experiments complete!"
echo "End Time: $(date)"
echo "========================================================"
