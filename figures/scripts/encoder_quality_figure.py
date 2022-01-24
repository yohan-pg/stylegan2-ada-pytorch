from prelude import *

ENCODERS = [
    "encoder-training-runs/encoder_0.3_slower_lr/2022-01-12_22:36:56/encoder-snapshot-000300.pkl",
    "encoder-training-runs/encoder_0.3_slower_lr/2022-01-12_22:36:56/encoder-snapshot-000300.pkl",
    "encoder-training-runs/encoder_0.3_slower_lr/2022-01-12_22:36:56/encoder-snapshot-000300.pkl"
]

IMAGE_PATHS = [
    "datasets/afhq2/test/cat/pixabay_cat_000117.png",
    "datasets/afhq2/train/cat/flickr_cat_000539.png",
    "datasets/afhq2/train/cat/flickr_cat_000018.png"
]

# -----------------------------

print("🛣 Encoder Quality")

OUT_PATH = "figures/out/encoder_quality_figure/"
fresh_dir(OUT_PATH)

errors = {}

preds = []
for i, encoder_path in enumerate(ENCODERS):
    E = open_encoder(encoder_path)

    encoder_errors = {}
    
    for j, path in enumerate(IMAGE_PATHS):
        target = open_target(E, path)
        pred = E(target).to_image()

        encoder_errors[path] = round(CRITERION(pred, target).item(), 3)

        save_image(target, f"{OUT_PATH}/target_{j}.png")
        save_image(E(target).to_image(), f"{OUT_PATH}/pred_{j}_{i}.png")

    errors[encoder_path] = encoder_errors

json.dump(errors, open(f"{OUT_PATH}/errors.json", "w"))