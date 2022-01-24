
export CUDA_VISIBLE_DEVICES=0

bash train.sh debug --gpus=1 --metrics none "${@}"

