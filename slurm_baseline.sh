#!/bin/bash
#SBATCH --account=def-jlalonde
#SBATCH --gres=gpu:1         # Number of GPU(s) per node
#SBATCH --cpus-per-task=8    # CPU cores/threads
#SBATCH --mem=32000M         # memory per node
#SBATCH --time=1-00:00       # time (DD-HH:MM)
#SBATCH --job-name=w_plus_no_norm

source .activate

date
nvidia-smi

python train.py \
    --kimg 25000 \
    --data=./datasets/afhq32cat.zip \
    --metrics=fid50k,pr50k3,ppl2_wend \
    --gamma 100 \
    --sample_w_plus True \
    --outdir="./training-runs/w_plus_no_norm" \
    "${@}"

# !!! wplus