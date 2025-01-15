#!/bin/bash

# SLURM directives
#SBATCH --partition=short-serial
#SBATCH --job-name=auto-annotation
#SBATCH -output=prepare_patch.out \
#SBATCH --error=prepare_patch.err \
#SBATCH --time=24:00:00
#SBATCH --mem=16000

# Your script execution
python step-1-data-preparation.py