
# !!! batch size is reduced to 8

python train.py \
    --kimg 5000 \
    --data=./datasets/afhq2_cat256.zip \
    --batch 8 \
    --snap 50 \
    --gamma 10 \
    --batch 8 \
    --gpus 2 \
    --fp32 True \
    --outdir="./training-runs/$1" \
    "${@:2}"

# --metrics=fid50k,pr50k3,ppl2_wend \
