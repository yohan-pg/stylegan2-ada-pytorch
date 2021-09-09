

python train.py --outdir="./training-runs/$1" --kimg 5000 --data=./datasets/afhq32cat.zip --batch 8 --metrics=fid50k,pr50k3  --gpus=2 --snap 25 --use_adaconv True "${@:2}"
# --metrics none
# 
