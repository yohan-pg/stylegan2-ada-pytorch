# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Project given image to the latent space of pretrained network pickle."""

from inversion import *

# todo why is L1 so bad??
# todo try random noise
# todo measure PPL
# todo try joint optimization
# todo try style mixing
# todo bring back jittering

# ? try training with ppl and without style mixing?
# ? try adaconv 3x3
# ? try training without a mapper
# ? try training without style mixing
# ? try training without noise at all
# ? try training on 256 so the VGG loss looks right
# ? add skip_w_avg_update to mapper

#!!!using SGD

METHOD = "adain"
G_PATH = f"pretrained/alpha-{METHOD}-002600.pkl"
OUT_DIR = f"out/{METHOD}-cat"
NUM_STEPS = 1000
SEQUENTIAL = False
VARIABLE_TYPE = WPlusVariableInitAtMean
CRITERION_TYPE = VGGCriterion
SNAPSHOT_FREQ = 10
# OPTIMIZER_CTOR = lambda params: torch.optim.Adam(
#     params, lr=0.01, betas=(0.0, 0.0)
# )
OPTIMIZER_CTOR = lambda params: torch.optim.Adam(
    params, lr=0.01, betas=(0.9, 0.999)
)
# OPTIMIZER_CTOR = lambda params: torch.optim.Adam(
#     params, lr=1.0, momentum=0.0
# )

AIM_FOR_FAKES = True
TARGET_A_NAME = "img00000013"
TARGET_B_NAME = "img00000003"
SEED = 41

if __name__ == "__main__":
    G = open_pkl(G_PATH)


    def invert_target(target_name: str, variable):
        # target = open_target(G, f"./datasets/samples/cats/00000/{target_name}.png")
        target = sample_image(G) #!!!
        return invert(
            G,
            target=target,
            variable=variable,
            out_path=f"{OUT_DIR}/optim_progress_{target_name}.png",
            num_steps=NUM_STEPS,
            criterion=CRITERION_TYPE(target),
            snapshot_frequency=SNAPSHOT_FREQ,
            optimizer_constructor=OPTIMIZER_CTOR,
        )
    torch.manual_seed(SEED)

    A = invert_target(TARGET_A_NAME, VARIABLE_TYPE.sample_from(G))
    B = invert_target(
        TARGET_B_NAME,
        A.final_variable.copy() if SEQUENTIAL else VARIABLE_TYPE.sample_from(G),
    )

    A.save_losses_plot(f"{OUT_DIR}/losses_{TARGET_A_NAME}.png")
    B.save_losses_plot(f"{OUT_DIR}/losses_{TARGET_B_NAME}.png")

    A.save_optim_trace(f"{OUT_DIR}/trace_{TARGET_A_NAME}.png")
    B.save_optim_trace(f"{OUT_DIR}/trace_{TARGET_B_NAME}.png")

    Interpolator(G).interpolate(
        A.final_variable,
        B.final_variable,
    ).save(OUT_DIR + f"/interpolation_{TARGET_A_NAME}_to_{TARGET_B_NAME}.png")
