
python train.py \
    --kimg 25000 \
    --data=./datasets/afhq2_cat256.zip \
    --metrics=fid50k,ppl2_wend \
    --cfg stylegan2map2 \
    --batch 16 \
    --gpus 2 \
    --fp32 True \
    --outdir="./training-runs/$1" \
    "${@:2}"

