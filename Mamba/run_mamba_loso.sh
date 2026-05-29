#!/bin/bash
#SBATCH --job-name=mamba_loso
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=/mnt/storage/admindi/home/dl011/slurm_outputs/mamba-%j.out
#SBATCH --error=/mnt/storage/admindi/home/dl011/slurm_outputs/mamba-%j.err

# Activate Conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate torch-env

# Ensure correct working directory
cd /mnt/storage/admindi/home/dl011/project/Deep_Learning_Mamba

# Set python path
export PYTHONPATH=$(pwd)

# HDF5 file path
H5_PATH="data/processed/hd5f/amr_features.h5"

echo "========================================================"
echo "Starting Leave-One-Strain-Out (LOSO) Cross-Validation Sweep"
echo "Job ID: $SLURM_JOB_ID"
echo "Start Time: $(date)"
echo "Using HDF5 dataset: $H5_PATH"
echo "========================================================"

# Run 1: GAP Baseline
echo -e "\n--------------------------------------------------------"
echo "🚀 Run 1: GAP Baseline (Balanced Undersampling, Threshold=0.5)"
echo "--------------------------------------------------------"
python -u src/training/sweep.py \
    --model_type gap \
    --h5_path "$H5_PATH" \
    --epochs 50 \
    --patience 10 \
    --output_dir pipeline_logs/gap_sweep

# Run 2: Mamba Classifier
echo -e "\n--------------------------------------------------------"
echo "🚀 Run 2: Mamba Classifier (Balanced Undersampling, Threshold=0.5)"
echo "--------------------------------------------------------"
python -u src/training/sweep.py \
    --model_type mamba \
    --h5_path "$H5_PATH" \
    --epochs 50 \
    --patience 10 \
    --output_dir pipeline_logs/mamba_sweep

# Run 3: Mamba Classifier with Threshold Tuning (Threshold=0.65 to reduce False Positives)
echo -e "\n--------------------------------------------------------"
echo "🚀 Run 3: Mamba Classifier with Decision Threshold Tuning (Threshold=0.65)"
echo "--------------------------------------------------------"
python -u src/training/sweep.py \
    --model_type mamba \
    --h5_path "$H5_PATH" \
    --epochs 50 \
    --patience 10 \
    --threshold 0.65 \
    --output_dir pipeline_logs/mamba_threshold_sweep

# Run 4: Mamba Classifier on Full Dataset with Custom Class Weight Penalty (No Undersampling, pos_weight=1.5)
echo -e "\n--------------------------------------------------------"
echo "🚀 Run 4: Mamba Classifier (Full Dataset, No Undersampling, pos_weight=1.5)"
echo "--------------------------------------------------------"
python -u src/training/sweep.py \
    --model_type mamba \
    --h5_path "$H5_PATH" \
    --epochs 50 \
    --patience 10 \
    --no_balanced \
    --pos_weight 1.5 \
    --output_dir pipeline_logs/mamba_weighted_sweep

echo -e "\n========================================================"
echo "Sweep experiments complete!"
echo "End Time: $(date)"
echo "========================================================"
