import torch

vgg = torch.hub.load("pytorch/vision:v0.10.0", "vgg11", pretrained=True).eval().cuda()
for i in range(4, 6):
    del vgg.classifier[i]
from sklearn.neighbors import KDTree
from inversion import *

G_PATH = "pretrained/adaconv-slowdown-all.pkl"

NUM_BATCHES = 5000
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

all_zs = []
all_xs = []

with torch.no_grad():
    for i in tqdm.tqdm(range(NUM_BATCHES)):
        torch.manual_seed(i)
        zs = torch.randn(BATCH_SIZE, 512, 512).cuda()
        xs = vgg((G(zs, None) + 1) / 2.0).cpu()
        for z, x in zip(zs.cpu(), xs.cpu()):
            all_zs.append(z.unsqueeze(0))
            all_xs.append(x.unsqueeze(0))

tree = KDTree(torch.cat(all_xs))

torch.save(tree, f"out/kd_{NUM_BATCHES*BATCH_SIZE}_samples.pkl")

with torch.no_grad():
    results = []
    for path in TARGET_PATHS:
        target = open_target(G, path)
        distance, idx = tree.query(vgg(target).detach().cpu())
        torch.manual_seed(idx // BATCH_SIZE)
        z = torch.randn(BATCH_SIZE, 512, 512).cuda()[idx.item() % BATCH_SIZE].unsqueeze(0)
        x = (G(z, None) + 1) / 2.0
        results.append(
            torch.cat((
                target,
                x,
            ))
        )
    save_image(
        torch.cat(
            results
        ),
        f"out/kd_nearest_{NUM_BATCHES*BATCH_SIZE}_samples.png",
        nrow=2
    )
