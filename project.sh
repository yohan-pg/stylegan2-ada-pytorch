set -e 

# ## churches
# python projector.py --outdir=out/church1 --target=datasets/chruch_samples/church1.png \
#     --network=./pretrained/stylegan2-church-config-f.pkl

# python projector.py --outdir=out/church2 --target=datasets/chruch_samples/church2.png \
#     --network=./pretrained/stylegan2-church-config-f.pkl

# faces with regular stylegan
python projector.py --outdir=out/real1 --target=./datasets/ffhq_samples/00015.png \
    --network=./pretrained/ffhq.pkl

python projector.py --outdir=out/real2 --target=./datasets/ffhq_samples/00018.png \
    --network=./pretrained/ffhq.pkl

python projector.py --outdir=out/fake1 --target=out/seed0000.png \
    --network=./pretrained/ffhq.pkl

python projector.py --outdir=out/fake2 --target=out/seed0032.png \
    --network=./pretrained/ffhq.pkl


## faces 
# python projector.py --outdir=out/real1 --target=./datasets/ffhq_samples/00015.png \
#     --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/ffhq.pkl

# python projector.py --outdir=out/real2 --target=./datasets/ffhq_samples/00018.png \
#     --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/ffhq.pkl

# python projector.py --outdir=out/fake1 --target=out/seed0000.png \
#     --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/ffhq.pkl

# python projector.py --outdir=out/fake2 --target=out/seed0032.png \
#     --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/ffhq.pkl