#!/bin/bash
#SBATCH --account=def-jlalonde
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8 
#SBATCH --mem=32000M
#SBATCH --time=1-00:00 #time (DD-HH:MM)
#SBATCH --job-name=adaconv_with_torgb_fixed
#SBATCH --output=slurm-%x-%j.out

source .activate

date
nvidia-smi

bash train.sh adaconv_with_torgb_fixed --use_adaconv True --gpus 4 --gamma 20 --data=./datasets/afhq2_cat128.zip --inject_in_torgb True

# bash train.sh adaconv_256_gamma_search --use_adaconv True --gpus 4 --gamma 60 --data=./datasets/afhq2_cat256.zip 
# --affine_slowdown 226 --mapper_slowdown 22.6 


# --inject_in_torgb True