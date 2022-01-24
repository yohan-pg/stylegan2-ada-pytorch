#!/bin/bash
#SBATCH --account=def-jlalonde
#SBATCH --gres=gpu:2         # Number of GPU(s) per node
#SBATCH --cpus-per-task=8    # CPU cores/threads
#SBATCH --mem=32000M         # memory per node
#SBATCH --time=3-00:00       # time (DD-HH:MM)
#SBATCH --job-name=adain-our-params

source .activate

date
nvidia-smi

bash train.sh adain-our-params \
    "${@}"
