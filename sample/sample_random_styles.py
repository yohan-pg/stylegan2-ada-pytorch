from inversion import *


from torchvision.utils import save_image

# G_PATH = "pretrained/adaconv-slowdown-all.pkl"
G_PATH = "pretrained/ffhq.pkl"
# G_PATH = "training-runs/cfg_auto_large_res_adain/00004-afhq256cat-auto2-gamma10-kimg5000-batch8/network-snapshot-001200.pkl"
OUT_DIR = f"out"
VARIABLE_TYPE = make_ZVariableWithDropoutOnZ(0.99)
BATCH_SIZE = 32

if __name__ == "__main__":
    G = open_generator(G_PATH)
    
    with torch.no_grad():
        images = VARIABLE_TYPE.sample_random_from(G, BATCH_SIZE).to_image()
    save_image(images, f"out/random_styles_{VARIABLE_TYPE.__name__}.png", nrow=16)
