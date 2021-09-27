from inversion import *
from training.ops import *

from torchvision.utils import save_image

METHOD = "adaconv"
# G_PATH="training-runs/church_adaconv/00000-church64-auto2-gamma10-kimg5000-batch8/network-snapshot-000000.pkl"
# G_PATH="training-runs/church_adaconv_without_style_mixing_reg/00000-church64-auto1-gamma100-kimg5000-batch8/network-snapshot-000200.pkl"
G_PATH=f"training-runs/afhq_adaconv_k1_sanity_check/00000-afhq64cat-auto1-gamma10-kimg5000-batch8/network-snapshot-000600.pkl"

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
