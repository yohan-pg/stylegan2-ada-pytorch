from prelude import *


ENCODERS = [
    ("encoder-training-runs/encoder_0.3_slower_lr/2022-01-12_22:36:56/encoder-snapshot-000300.pkl", WVariable),
    ("encoder-training-runs/encoder_0.3_slower_lr/2022-01-12_22:36:56/encoder-snapshot-000300.pkl", WPlusVariable),
    ("encoder-training-runs/encoder_0.3_slower_lr/2022-01-12_22:36:56/encoder-snapshot-000300.pkl", WVariable)
]

IMAGE_PATHS = [
    "datasets/afhq2/test/cat/pixabay_cat_000117.png",
    "datasets/afhq2/train/cat/flickr_cat_000539.png",
    "datasets/afhq2/train/cat/flickr_cat_000018.png",
]

# -----------------------------

print("🛣 Full Inversion")

OUT_PATH = f"figures/out/full_inversion_quality_figure/"
fresh_dir(OUT_PATH)

errors = {}

preds = []
for i, (encoder_path, variable_type) in enumerate(ENCODERS):
    E = open_encoder(encoder_path)

    encoder_errors = {}

    for j, path in enumerate(IMAGE_PATHS):
        target = open_target(E, path)

        pred = Inverter(
            G_or_E=E,
            num_steps=NUM_OPTIM_STEPS, 
            variable=add_hard_encoder_constraint(variable_type),
            create_optimizer=lambda params: torch.optim.Adam(params, lr=LR)
        )(target).final_pred
        encoder_errors[path] = round(CRITERION(pred, target).item(), 3)

        save_image(target, f"{OUT_PATH}/target_{j}.png")
        save_image(pred, f"{OUT_PATH}/pred_{j}_{i}.png")

    errors[encoder_path] = encoder_errors

json.dump(errors, open(f"{OUT_PATH}/errors.json", "w"))