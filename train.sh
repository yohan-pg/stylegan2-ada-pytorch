
python train.py \
    --kimg 5000 \
    --data=./datasets/afhq256cat.zip \
    --batch 8 \
    --gamma 10 \
    --gpus=1 \
    --outdir="./training-runs/$1" \
    "${@:2}"

    # --metrics=fid50k,pr50k3,ppl2_wend \
    # --cfg stylegan2map2 \
    # --batch 8 \
