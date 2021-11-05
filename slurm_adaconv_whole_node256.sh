#!/bin/bash
#SBATCH --account=def-jlalonde
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8 
#SBATCH --mem=32000M
#SBATCH --time=5-00:00 #time (DD-HH:MM)
#SBATCH --job-name=adaconv_final_slowdown
#SBATCH --output=slurm-%x-%j.out

source .activate

date
nvidia-smi

bash train.sh adaconv_final_slowdown --use_adaconv True --gpus 4 --gamma 30 --inject_in_torgb True --normalize_latent True --affine_slowdown 100
