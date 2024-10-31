#!/bin/bash

INDEX=$1

# SLURM directives
#SBATCH --partition=short-serial
#SBATCH --job-name=auto-annotation
#SBATCH -output=auto-annotation_index${INDEX}.out \
#SBATCH --error=auto-annotation_index${INDEX}.err \
#SBATCH --time=24:00:00
#SBATCH --mem=16000

# Your script execution
python prepare_training_patch.py