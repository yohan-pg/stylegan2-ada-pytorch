

python train.py --outdir="./training-runs/$1" --kimg 5000 --aug noaug --data=./datasets/afhq64cat.zip --batch 8 --metrics=fid50k,pr50k3 --gpus=2 --snap 50 --gamma 10 --use_adaconv True "${@:2}"
# --metrics none
# 
