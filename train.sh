python train.py --outdir="./training-runs/$1" --kimg 5000 --data=./datasets/afhq32cat.zip --batch 8 --snap 25 --gpus=2 --metrics=fid50k,pr50k3 "${@:2}" 


