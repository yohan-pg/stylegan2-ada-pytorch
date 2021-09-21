
python train.py \
    --kimg 25000 \
    --data=./datasets/afhq128cat.zip \
    --cfg stylegan2 \
    --metrics=fid50k,pr50k3 \
    --snap 50 \
    --gamma 10 \
    --gpus=2 \
    --outdir="./training-runs/$1" \
    "${@:2}"
