set -e 

case $1 in 
    church*) 
        python projector.py --outdir=out/church1 --target=datasets/chruch_samples/church1.png \
        --network=./pretrained/stylegan2-church-config-f.pkl

        python projector.py --outdir=out/church2 --target=datasets/chruch_samples/church2.png \
            --network=./pretrained/stylegan2-church-config-f.pkl
    ;;
    cars*) 
        python projector.py --outdir=out/car6-f --target=datasets/samples/cars/car6.png \
            --network=./pretrained/stylegan2-car-config-f.pkl # ! should be c
            

        python projector.py --outdir=out/car6-d --target=datasets/samples/cars/car6.png \
            --network=./pretrained/stylegan2-car-config-d.pkl

        python projector.py --outdir=out/car7-c --target=datasets/samples/cars/car7.png \
            --network=./pretrained/stylegan2-car-config-c.pkl

        python projector.py --outdir=out/car7-d --target=datasets/samples/cars/car7.png \
            --network=./pretrained/stylegan2-car-config-d.pkl
    ;;
    ffhq*) 
        python projector.py --outdir=out/real1 --target=./datasets/ffhq_samples/00015.png \
            --network=./pretrained/ffhq.pkl

        python projector.py --outdir=out/real2 --target=./datasets/ffhq_samples/00018.png \
            --network=./pretrained/ffhq.pkl
    ;;
    car-c*) 
        python projector.py --outdir=out/real1 --target=./datasets/ffhq_samples/00015.png \
            --network=pretrained/stylegan2-car-config-c.pkl

        python projector.py --outdir=out/real2 --target=./datasets/ffhq_samples/00018.png \
            --network=pretrained/stylegan2-car-config-c.pkl
    ;;
    car-d*) 
        python projector.py --outdir=out/real1 --target=./datasets/ffhq_samples/00015.png \
            --network=pretrained/stylegan2-car-config-d.pkl

        python projector.py --outdir=out/real2 --target=./datasets/ffhq_samples/00018.png \
            --network=pretrained/stylegan2-car-config-d.pkl
    ;;
    car-f*) 
        python projector.py --outdir=out/real1 --target=./datasets/ffhq_samples/00015.png \
            --network=pretrained/stylegan2-car-config-f.pkl

        python projector.py --outdir=out/real2 --target=./datasets/ffhq_samples/00018.png \
            --network=pretrained/stylegan2-car-config-f.pkl
    ;;
    *) echo "No such dataset."
esac