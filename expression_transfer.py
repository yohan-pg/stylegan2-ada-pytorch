"""Project given image to the latent space of pretrained network pickle."""
from edits.prelude import *
from inversion import *
from training.networks import *

if False:
    METHOD = "adain"
    G_PATH = "pretrained/no_torgb_adain_tmp.pkl"
else:
    METHOD = "adaconv"
    E_PATH = "encoder-training-runs/encoder_ffhq_0.1/encoder-snapshot-000050.pkl"

#!!! sequential fails with encoder init

NUM_STEPS = 500
CRITERION_TYPE = VGGCriterion
SNAPSHOT_FREQ = 10
SEQUENTIAL = True 

AIM_FOR_FAKE_A = False
AIM_FOR_FAKE_B = False

TARGET_A_PATH = "datasets/samples/my_face_a.png"
TARGET_B_PATH = "datasets/samples/my_face_b.png"
TARGET_C_PATH = "datasets/samples/hen_face_a.png"

SEED = 2

OUT_DIR = f"out"

if __name__ == "__main__":
    fresh_dir(OUT_DIR)

    E = open_encoder(E_PATH)

    VARIABLE_TYPE = add_hard_encoder_constraint(E.variable_type, 0.1, encoder_init=False)

    criterion = CRITERION_TYPE()

    def invert_target(target: torch.Tensor, name: str, variable_type: Variable):
        inverter = Inverter(
            E,
            variable_type=variable_type,
            num_steps=NUM_STEPS,
            criterion=1000 * criterion,
            create_optimizer=lambda params: torch.optim.Adam(params, lr=1.0),
            create_schedule=lambda optimizer: lr_scheduler.LambdaLR(
                optimizer, lambda epoch: min(1.0, epoch / 100.0)
            ),
            snapshot_frequency=SNAPSHOT_FREQ,
            seed=SEED,
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

    target_A = open_target(E, TARGET_A_PATH)
    target_B = open_target(E, TARGET_B_PATH)
    target_C = open_target(E, TARGET_C_PATH)

    A = invert_target(target_A, "A", VARIABLE_TYPE)
    B = invert_target(
        target_B,
        "B",
        A.final_variable.copy() if SEQUENTIAL else VARIABLE_TYPE,
    )
    C = invert_target(
        target_C,
        "C",
        VARIABLE_TYPE,
    )

    Interpolation.from_variables(C.final_variable, A.final_variable).save(
        f"{OUT_DIR}/face_mix.png"
    )

    Interpolation.from_variables(
        C.final_variable,
        C.final_variable + (B.final_variable - A.final_variable),
        num_steps=12,
        gain=3.0,
    ).save(f"{OUT_DIR}/expression_transfer.png")

    Y = A.final_variable
    Y2 = B.final_variable
    H = C.final_variable

    O = Y.from_data(Y.data * 0 + E.G[0].mapping.w_avg)

    Interpolation.from_variables(
        H,
        O,
        num_steps=9,
        gain=1.0,
    ).save(f"{OUT_DIR}/to_origin.png")
    
    delta_Y = Y2 - Y

    H_to_Y = Y - H
    H_to_Y.data /= H_to_Y.data.norm(dim=(1,2))

    O_to_H = H - O
    O_to_H.data /= O_to_H.data.norm(dim=(1,2))
    
    corrected_delta_Y = delta_Y - H_to_Y * (delta_Y.data * H_to_Y.data).sum()
    rectified_delta_Y = corrected_delta_Y - O_to_H * (corrected_delta_Y.data * O_to_H.data).sum()
    
    Interpolation.from_variables(
        H,
        H + corrected_delta_Y,
        num_steps=9,
        gain=2.0,
    ).save(f"{OUT_DIR}/expression_transfer_corrected.png")

    Interpolation.from_variables(
        H + corrected_delta_Y * 2.0,
        O,
        num_steps=9,
        gain=1.0,
    ).save(f"{OUT_DIR}/transfer_to_origin.png")