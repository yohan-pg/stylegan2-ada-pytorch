python train.py --outdir="./training-runs/$1" --kimg 5000 --data=./datasets/afhq32cat.zip --snap 50 --aug noaug --batch 8 --gpus=2 --metrics=fid50k,pr50k3 "${@:2}" 


