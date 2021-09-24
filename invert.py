# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Project given image to the latent space of pretrained network pickle."""

from inversion import *
import shutil

# todo why is L1 so bad??
# todo init at mean for the WConvexCombinationVariable
# todo try random noise mode! seems to make a difference for encoding
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

METHOD = "adain"
# G_PATH = f"pretrained/alpha-{METHOD}-002600.pkl"
G_PATH = "pretrained/stylegan2-church-config-f.pkl"
OUT_DIR = f"out/{METHOD}-cat"
NUM_STEPS = 10_000
SEQUENTIAL = False
VARIABLE_TYPE = WPlusVariable
CRITERION_TYPE = VGGCriterion
SNAPSHOT_FREQ = 10
OPTIMIZER_CTOR = lambda params: torch.optim.Adam(
    params, lr=0.01
)

AIM_FOR_FAKES = True
TARGET_A_PATH = f"./datasets/samples/cats/00000/img00000013.png"
TARGET_B_PATH = f"./datasets/samples/cats/00000/img00000003.png"
SEED = 11

if __name__ == "__main__":
    shutil.rmtree(OUT_DIR, ignore_errors=True)

    G = open_generator(G_PATH)

    def invert_target(target: torch.Tensor, name: str, variable):
        return invert(
            G,
            target=target,
            variable=variable,
            out_path=f"{OUT_DIR}/optim_progress_{name}.png",
            num_steps=NUM_STEPS,
            criterion=CRITERION_TYPE(),
            snapshot_frequency=SNAPSHOT_FREQ,
            optimizer_constructor=OPTIMIZER_CTOR,
        )
    torch.manual_seed(SEED)

    target_A = sample_image(G) if AIM_FOR_FAKES else open_target(G, TARGET_A_PATH)
    target_B = sample_image(G) if AIM_FOR_FAKES else open_target(G, TARGET_B_PATH)

    A = invert_target(target_A, "A", VARIABLE_TYPE.sample_from(G))
    B = invert_target(
        target_B,
        "B",
        A.final_variable.copy() if SEQUENTIAL else VARIABLE_TYPE.sample_from(G),
    )

    A.save_losses_plot(f"{OUT_DIR}/losses_A.png")
    B.save_losses_plot(f"{OUT_DIR}/losses_B.png")

    A.save_optim_trace(f"{OUT_DIR}/trace_A.png")
    B.save_optim_trace(f"{OUT_DIR}/trace_B.png")

    Interpolator(G).interpolate(
        A.final_variable,
        B.final_variable,
    ).save(OUT_DIR + f"/interpolation_A_to_B.png")
