from inversion import *
from training.ops import *

from torchvision.utils import save_image

METHOD = "adaconv"
G_PATH = f"training-runs/cfg_auto_large_res_adaconv/00000-afhq256cat-auto2-gamma10-kimg5000-batch8/network-snapshot-001600.pkl"

OUT_DIR = f"out"
BATCH_SIZE = 12
NUM_EDITS = 4
VARIABLE_TYPE = WVariable
INTENSITY = 10.0

if __name__ == "__main__":
    with torch.no_grad():
        G = open_generator(G_PATH)

        var = RandomWVariable.sample_from(G, BATCH_SIZE)

        images = []
        for edit in range(NUM_EDITS):
            edit = RandomWVariable.sample_from(G, 1).direction_to(WVariable.sample_from(G, 1))
            images.append((var + edit * INTENSITY).to_image())
        
        save_image(
            torch.cat(images), 
            f"out/random_edits_{METHOD}.png", 
            nrow=BATCH_SIZE
        )
        

