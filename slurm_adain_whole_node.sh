#!/bin/bash
#SBATCH --account=def-jlalonde
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8 
#SBATCH --mem=32000M
#SBATCH --time=1-00:00 #time (DD-HH:MM)
#SBATCH --job-name=adain256_final
#SBATCH --output=slurm-%x-%j.out

source .activate

date
nvidia-smi

bash train.sh adain256_final --gpus 4 --batch 16 --inject_in_torgb True --normalize_latent True --data=./datasets/afhq2_cat256.zip