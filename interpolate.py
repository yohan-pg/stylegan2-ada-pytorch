from inversion import *
from training.networks import *

if False:
    METHOD = "adain"
    # PATH = "pretrained/no_torgb_adain_tmp.pkl"
    PATH = "encoder-training-runs/encoder_0.1_baseline/encoder-snapshot-000100.pkl"
else:
    METHOD = "adaconv"
    # PATH = "pretrained/ffhq.pkl"
    PATH = "encoder-training-runs/encoder_0.0/2022-01-04_23:24:05/encoder-snapshot-000050.pkl"


VARIABLE_TYPE = add_soft_encoder_constraint(WPlusVariable, 0.0, 10.0, encoder_init=True)
NUM_STEPS = 50
CRITERION_TYPE = VGGCriterion
SNAPSHOT_FREQ = 50

AIM_FOR_FAKE_A = False
AIM_FOR_FAKE_B = False

# TARGET_A_PATH = "./datasets/afhq2/train/cat/flickr_cat_000006.png"
# TARGET_B_PATH = "./datasets/afhq2/train/cat/flickr_cat_000018.png"

TARGET_A_PATH = "datasets/afhq2/test/cat/pixabay_cat_000117.png"
TARGET_B_PATH = "datasets/afhq2/train/cat/flickr_cat_000539.png"

SEED = 2
SAME_SEED = True
SEQUENTIAL = False

print(VARIABLE_TYPE.__name__)
OUT_DIR = f"out/interpolate/{METHOD}/{VARIABLE_TYPE.__name__}"

if __name__ == "__main__":
    fresh_dir(OUT_DIR)

    G, G_or_E = open_model(PATH)

    def invert_target(target: torch.Tensor, name: str, variable_type: Variable):
        inverter = Inverter(
            G_or_E,
            variable=variable_type,
            num_steps=NUM_STEPS,
            criterion=CRITERION_TYPE(),
            create_optimizer=lambda params: torch.optim.Adam(params, lr=0.02),
            snapshot_frequency=SNAPSHOT_FREQ,
            seed=SEED if SAME_SEED else None,
        )
        try:
            for inversion, snapshot_iter in tqdm.tqdm(
                inverter.all_inversion_steps(target)
            ):
                if snapshot_iter:
                    inversion.snapshot(f"{OUT_DIR}/optim_progress_{name}.png")
        except KeyboardInterrupt:
            pass

        return inversion

    torch.manual_seed(SEED)

    target_A = sample_image(G) if AIM_FOR_FAKE_A else (open_target(G, TARGET_A_PATH))
    target_B = sample_image(G) if AIM_FOR_FAKE_B else (open_target(G, TARGET_B_PATH))

    try:
        z_mean = ZVariableInitAtMean.sample_from(G)
        save_image(z_mean.to_image(), f"{OUT_DIR}/mean_z.png")
        w_mean = WVariable.sample_from(G)
        save_image(w_mean.to_image(), f"{OUT_DIR}/mean_w.png")
    except:
        print("Failed to save mean images")
        pass

    A = invert_target(target_A, "A", VARIABLE_TYPE)
    B = invert_target(
        target_B,
        "B",
        A.final_variable.copy() if SEQUENTIAL else VARIABLE_TYPE,
    )

    Inversion.save_losses_plot(dict(A=A, B=B), f"{OUT_DIR}/losses.png")
    Inversion.save_regularization_plot(dict(A=A, B=B), f"{OUT_DIR}/regularization.png")

    interpolation = Interpolation.from_variables(
        A.final_variable, B.final_variable, gain=1.0
    )
    interpolation.save(f"{OUT_DIR}/interpolation_A_to_B.png", target_A, target_B)

    if hasattr(A.final_variable, "to_W"):
        interpolation = Interpolation.from_variables(
            A.final_variable.to_W(), B.final_variable.to_W(), gain=1.0
        )

    A.save_optim_trace(f"{OUT_DIR}/trace_A.png")
    B.save_optim_trace(f"{OUT_DIR}/trace_B.png")


os.system(
    rf"""
ssh -p 12345 Yohan@localhost -x 'osascript -e "
if frontmost of application \"Visual Studio Code\" then
else
	display notification \"Script Finished\" sound name \"Sound Name\" with title \"Tchai\"
end if
"'
"""
)
