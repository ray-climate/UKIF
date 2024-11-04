#!/bin/bash

# SLURM directives
#SBATCH --partition=short-serial
#SBATCH --job-name=prepare-training
#SBATCH --output=prepare_data_%a.out
#SBATCH --error=prepare_data_%a.err
#SBATCH --time=24:00:00
#SBATCH --mem=16000

$date_i=$1
echo "Processing date: $date_i"
python prepare_training_patch_i.py $date_i