
python train.py \
    --kimg 5000 \
    --data=./datasets/afhq32cat.zip \
    --batch 8 --metrics=fid50k,pr50k3,ppl2_wend \
    --snap 50 \
    --gamma 10 \
    --gpus=1 \
    --outdir="./training-runs/$1" \
    "${@:2}"
