#!/bin/bash
#SBATCH --account=def-jlalonde
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8 
#SBATCH --mem=32000M
#SBATCH --time=3-00:00 #time (DD-HH:MM)
#SBATCH --job-name=baseline

source slurm_baseline.sh --gpus 4