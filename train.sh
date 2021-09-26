
python train.py \
    --kimg 5000 \
    --data=./datasets/afhq64cat.zip \
    --batch 8 \
    --metrics=fid50k,pr50k3,ppl2_wend \
    --snap 50 \
    --gamma 10 \
    --gpus=2 \
    --fp32 True \
    --outdir="./training-runs/$1" \
    "${@:2}"
