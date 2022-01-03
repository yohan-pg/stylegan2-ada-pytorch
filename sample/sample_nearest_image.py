from inversion import *
from training.ops import *

from torchvision.utils import save_image

G_PATH = f"pretrained/adaconv-slowdown-all.pkl"
TARGET_PATH = f"./datasets/afhq2/train/cat/flickr_cat_000006.png"
BATCH_SIZE = 16
NUM_BATCHES = 1000

torch.manual_seed(0)

if __name__ == "__main__":
    with torch.no_grad():
        G = open_generator(G_PATH)
        target = open_target(G, TARGET_PATH)

        criterion = VGGCriterion()

        nearest_image = None
        best_distance = float("Inf")

        for i in tqdm.tqdm(range(NUM_BATCHES)):
            samples = ZVariable.sample_random_from(G, BATCH_SIZE).to_image()
            
            for sample in samples:
                sample = sample.unsqueeze(0)
                if (distance := criterion(sample, target)).item() < best_distance:
                    best_distance = distance.item()
                    nearest_image = sample

                    print(best_distance)
                    save_image(
                        torch.cat((
                            target, 
                            nearest_image
                        )),    
                        "out/nearest_image.png"
                    )
