# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Project given image to the latent space of pretrained network pickle."""

from inversion import *
from training.networks import *

if False:
    METHOD = "adain"
    G_PATH = "pretrained/no_torgb_adain_tmp.pkl"
else:
    METHOD = "adaconv"
    G_PATH = "pretrained/ffhq.pkl"

VARIABLE_TYPE = WPlusVariableRandomInit
NUM_STEPS = 300
CRITERION_TYPE = VGGCriterion
SNAPSHOT_FREQ = 10
SEQUENTIAL = True

AIM_FOR_FAKE_A = False
AIM_FOR_FAKE_B = False

TARGET_A_PATH = "datasets/samples/my_face_a.png"
TARGET_B_PATH = "datasets/samples/my_face_b.png"
TARGET_C_PATH = "datasets/samples/hen_face_a.png"

SEED = 2

print(VARIABLE_TYPE.__name__)
OUT_DIR = f"out"

if __name__ == "__main__":
    fresh_dir(OUT_DIR)

    G = open_generator(G_PATH)
    G.requires_grad_(True)
    G.train()

    criterion = CRITERION_TYPE()

    def invert_target(target: torch.Tensor, name: str, variable_type: Variable):
        inverter = Inverter(
            G,
            variable_type=variable_type,
            num_steps=NUM_STEPS,
            criterion=100 * criterion,
            create_optimizer=lambda params: torch.optim.Adam(params, lr=0.1),
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

    target_A = open_target(G, TARGET_A_PATH)
    target_B = open_target(G, TARGET_B_PATH)
    target_C = open_target(G, TARGET_C_PATH)

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

    save_image(
        (
            (A.final_variable - B.final_variable) * 0.8
            + WPlusVariable.sample_from(G, 1)
        ).to_image(),
        f"{OUT_DIR}/face_diff.png",
    )

    Interpolation.from_variables(
        C.final_variable - (B.final_variable - A.final_variable),
        C.final_variable + (B.final_variable - A.final_variable),
        num_steps=12,
        gain=3.0,
    ).save(f"{OUT_DIR}/expression_transfer.png")
