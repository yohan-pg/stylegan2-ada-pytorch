from prelude import *

ENCODERS = [
    (
        "encoder-training-runs/encoder_0.3_slower_lr/2022-01-12_22:36:56/encoder-snapshot-000300.pkl",
        WVariable,
    ),
    (
        "encoder-training-runs/encoder_0.3_slower_lr/2022-01-12_22:36:56/encoder-snapshot-000300.pkl",
        WPlusVariable,
    ),
    (
        "encoder-training-runs/encoder_0.3_slower_lr/2022-01-12_22:36:56/encoder-snapshot-000300.pkl",
        WVariable,
    ),
]

TARGET_A_PATH = "datasets/afhq2/test/cat/pixabay_cat_000117.png"
TARGET_B_PATH = "datasets/afhq2/train/cat/flickr_cat_000539.png"

# -----------------------------

print("🛣 Interpolation Quality")

OUT_PATH = f"figures/out/interpolation_quality_figure/"
fresh_dir(OUT_PATH)

preds = []
for i, (encoder_path, variable_type) in enumerate(ENCODERS):
    E = open_encoder(encoder_path)

    inverter = Inverter(
        G_or_E=E,
        num_steps=NUM_OPTIM_STEPS,
        variable=add_hard_encoder_constraint(variable_type),
        create_optimizer=lambda params: torch.optim.Adam(params, lr=LR),
    )

    target_A = open_target(E, TARGET_A_PATH)
    target_B = open_target(E, TARGET_B_PATH)

    inversion_A = inverter(target_A)
    inversion_B = inverter(target_B)

    save_image(inversion_A.final_pred, f"{OUT_PATH}/inversion_A_{i}.png")
    save_image(inversion_B.final_pred, f"{OUT_PATH}/inversion_B_{i}.png")

    save_image(
        inversion_A.final_variable.interpolate(
            inversion_B.final_variable, 0.25
        ).to_image(),
        f"{OUT_PATH}/inversion_quarter_{i}.png",
    )
    save_image(
        inversion_A.final_variable.interpolate(
            inversion_B.final_variable, 0.50
        ).to_image(),
        f"{OUT_PATH}/inversion_half_{i}.png",
    )
    save_image(
        inversion_A.final_variable.interpolate(
            inversion_B.final_variable, 0.75
        ).to_image(),
        f"{OUT_PATH}/inversion_three_quarters_{i}.png",
    )

    save_image(target_A, f"{OUT_PATH}/target_A.png")
    save_image(target_B, f"{OUT_PATH}/target_B.png")
