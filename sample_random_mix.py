from inversion import *
from training.ops import *

from torchvision.utils import save_image

METHOD = "adaconv"
G_PATH="training-runs/church_adaconv/00000-church64-auto2-gamma10-kimg5000-batch8/network-snapshot-000000.pkl"

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
