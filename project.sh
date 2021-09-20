set -e 

export CUDA_VISIBLE_DEVICES=1

case $1 in 
    church*) 
        python projector.py --outdir=out/church1 --target=datasets/chruch_samples/church1.png \
        --network=./pretrained/stylegan2-church-config-f.pkl

        python projector.py --outdir=out/church2 --target=datasets/chruch_samples/church2.png \
            --network=./pretrained/stylegan2-church-config-f.pkl
    ;;
    cars*) 
        python projector.py --outdir=out/car6-f --target=datasets/samples/cars/car6.png \
            --network=./pretrained/stylegan2-car-config-f.pkl
            
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
    cats*)
        python projector.py --outdir=out/cat-1-adaconv --target=./datasets/samples/cats/00000/img00000003.png \
            --network=pretrained/alpha-adaconv-002600.pkl "${@:2}"

        python projector.py --outdir=out/cat-2-adaconv --target=./datasets/samples/cats/00000/img00000013.png \
            --network=pretrained/alpha-adaconv-002600.pkl "${@:2}"

        python3 interpolator.py  pretrained/alpha-adaconv-002600.pkl 'out/cat-1-adaconv' 'out/cat-2-adaconv'
        
        python projector.py --outdir=out/cat-1-adain --target=./datasets/samples/cats/00000/img00000003.png \
            --network=pretrained/alpha-adain-002600.pkl "${@:2}"

        python projector.py --outdir=out/cat-2-adain --target=./datasets/samples/cats/00000/img00000013.png \
            --network=pretrained/alpha-adain-002600.pkl "${@:2}"

        python3 interpolator.py  pretrained/alpha-adain-002600.pkl 'out/cat-1-adain' 'out/cat-2-adain'
    ;;
    *)
    
    
     echo "No such dataset."
esac