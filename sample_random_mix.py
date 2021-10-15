from inversion import *
from training.ops import *

from torchvision.utils import save_image

METHOD = "adaconv"
G_PATH=f"training-runs/afhq_adaconv_no_torgb/00005-afhq64cat-auto1-gamma10-kimg5000-batch8/network-snapshot-000800.pkl"

OUT_DIR = f"out"
BATCH_SIZE = 12

if __name__ == "__main__":
    with torch.no_grad():
        G = open_generator(G_PATH)

        save_image(
            sample_mix(G, BATCH_SIZE), 
            f"out/random_mix_{METHOD}.png", 
            nrow=BATCH_SIZE
        )
        

