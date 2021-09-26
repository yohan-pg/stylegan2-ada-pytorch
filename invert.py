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

# todo bring back jittering
# todo bring back lr schedule
# todo try PULSE-like spherical optimization
# todo try joint optimization -> minimize distance?

# ? try adaconv 3x3?
# ? try training without a mapper for z optim
# ? try training without style mixing, see if interpolation is better/worse with/without w+
# ? try training on 256 so the VGG loss looks right
# ? eventually bring back noise + noise optim

# METHOD = "adaconv"
# G_PATH = f"pretrained/{METHOD}-church128-001600.pkl"

METHOD = "adaconv"
G_PATH = "training-runs/linear_mapper_adaconv/00000-church64-auto01-gamma100-kimg5000-batch8/network-snapshot-000200.pkl"

OUT_DIR = f"out/{METHOD}-church"
NUM_STEPS = 1_000
SEQUENTIAL = False
# VARIABLE_TYPE = ZVariableInitAtMean if METHOD == "adaconv" else WPlusVariableInitAtMean
VARIABLE_TYPE = ZVariable
CRITERION_TYPE = VGGCriterion
SNAPSHOT_FREQ = 20
OPTIMIZER_CTOR = lambda params: torch.optim.Adam(
    params, lr=0.05, betas=(0.0, 0.0)
)

AIM_FOR_FAKES = False
TARGET_A_PATH = f"/home-local2/yopog.extra.nobkp/stylegan2-ada-pytorch/datasets/samples/churches/a.webp" # ./datasets/samples/cats/00000/img00000013.png
TARGET_B_PATH = f"/home-local2/yopog.extra.nobkp/stylegan2-ada-pytorch/datasets/samples/churches/c.webp" # ./datasets/samples/cats/00000/img00000003.png
SEED = 11

if __name__ == "__main__":
    shutil.rmtree(OUT_DIR, ignore_errors=True)

    G = open_generator(G_PATH)
    criterion = CRITERION_TYPE()

    def invert_target(target: torch.Tensor, name: str, variable: Variable):
        return invert(
            G,
            target=target,
            variable=variable,
            out_path=f"{OUT_DIR}/optim_progress_{name}.png",
            num_steps=NUM_STEPS,
            criterion=criterion,
            snapshot_frequency=SNAPSHOT_FREQ,
            optimizer_constructor=OPTIMIZER_CTOR,
            constraints=[
                # StyleJittering(variable)
            ]
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

    interpolation = Interpolator(G).interpolate(
        A.final_variable,
        B.final_variable,
    )
    
    interpolation.save(OUT_DIR + f"/interpolation_A_to_B.png")

    print(interpolation.ppl(criterion))
    print(interpolation.endpoint_distance(criterion))

    print(interpolation.latent_distance(criterion).item())
