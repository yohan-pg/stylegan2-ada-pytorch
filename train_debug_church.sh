
# * Note that on tch, the device # appears reversed (titan v is 0, titant rtx is 1)

export CUDA_VISIBLE_DEVICES=1

bash train_church.sh debug --gpus=1 --metrics none "${@}"

