import torch
from inversion import * 

G_PATH = "pretrained/adaconv-slowdown-all.pkl"

NUM_BATCHES = 8
BATCH_SIZE = 16
TARGET_PATHS = [
    "./datasets/afhq2/train/cat/flickr_cat_000006.png",

    "datasets/afhq2/train/cat/pixabay_cat_000077.png",
    "datasets/afhq2/train/cat/pixabay_cat_004220.png",

    "./datasets/afhq2/test/cat/flickr_cat_000233.png",
    "./datasets/afhq2/train/cat/pixabay_cat_004436.png",
]

# ----------------------
torch.manual_seed(0)

G = open_generator(G_PATH).cuda()
criterion = VGGCriterion()

all_zs = []
all_xs = []

with torch.no_grad():
    for i in tqdm.tqdm(range(NUM_BATCHES)):
        torch.manual_seed(i)
        zs = torch.randn(BATCH_SIZE, 512, 512).cuda()
        xs = (G(zs, None) + 1) / 2.0
        for z, x in zip(zs, xs):
            all_zs.append(z.unsqueeze(0))
            all_xs.append(x.unsqueeze(0))

average_distances = []
for a in tqdm.tqdm(all_xs):
    dist = torch.tensor([0.0]).cuda()
    for b in all_xs:
        dist += criterion(a, b) / len(all_xs)
    average_distances.append(dist)

average_distances = torch.stack(average_distances)
all_zs[average_distances.argmin().item()]
        
save_image(
    (G(all_zs[average_distances.argmin().item()], None) + 1) / 2.0,
    f"out/mean_init_{NUM_BATCHES * BATCH_SIZE}.png"
)