#!/bin/bash

# SLURM directives
#SBATCH --partition=short-serial
#SBATCH --job-name=process_patches
#SBATCH --output=logs/process_patches_%A_%a.out
#SBATCH --error=logs/process_patches_%A_%a.err
#SBATCH --time=3:00:00
#SBATCH --mem=16000
#SBATCH --array=0-99

mkdir -p logs-step1

# Your script execution
python process_chunk.py ${SLURM_ARRAY_TASK_ID} 100