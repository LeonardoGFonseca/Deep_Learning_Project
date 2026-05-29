#!/bin/bash

# Clean up any existing logs from these directories to avoid confusion
rm -rf pipeline_logs/gap_sweep
rm -rf pipeline_logs/mamba_sweep
rm -rf pipeline_logs/mamba_threshold_sweep
rm -rf pipeline_logs/mamba_weighted_sweep_part1
rm -rf pipeline_logs/mamba_weighted_sweep_part2

echo "Submitting 5 parallel jobs to cover 30 strains across 5 GPUs..."

sbatch slurm_scripts/run_gap.sh
sbatch slurm_scripts/run_mamba.sh
sbatch slurm_scripts/run_threshold.sh
sbatch slurm_scripts/run_weighted_p1.sh
sbatch slurm_scripts/run_weighted_p2.sh

echo "All jobs submitted! Use 'squeue -u dl011@lasige.di.fc.ul.pt' to monitor."
