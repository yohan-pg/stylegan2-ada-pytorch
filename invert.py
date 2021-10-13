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

# ? implement PULSE as a comparaison?
# todo bring back jittering
# todo bring back LR schedule

DISABLE_NORMALIZE = False
FORCE_NORMALIZE = False
FORCE_LERP = False
FINE_TUNE_G = False 
TRANSFORM_TARGETS = False

# if False:
#     METHOD = "adain"
#     G_PATH = "training-runs/cfg_auto_large_res_adain/00004-afhq256cat-auto2-gamma10-kimg5000-batch8/network-snapshot-001200.pkl"
# else:
#     METHOD = "adaconv"
#     G_PATH = "training-runs/cfg_auto_large_res_adaconv/00000-afhq256cat-auto2-gamma10-kimg5000-batch8/network-snapshot-001200.pkl"

# METHOD = "adaconv"
# G_PATH = "training-runs/cfg_linear_mapper_large_res_adaconv/00000-afhq256cat-auto12-gamma10-kimg5000-batch8/network-snapshot-001200.pkl"

METHOD = "adaconv"
G_PATH = "training-runs/cfg_auto_large_res_adaconv_fixed_lr_mult/00000-afhq256cat-auto2-gamma10-kimg5000-batch8/network-snapshot-001200.pkl"

NUM_STEPS = 2_000
SEQUENTIAL = False
CRITERION_TYPE = VGGCriterion
SNAPSHOT_FREQ = 20
OPTIMIZER_CTOR = lambda params: torch.optim.Adam(
    params, 
    lr=0.01 / (math.sqrt(512) if (issubclass(VARIABLE_TYPE, ZVariable)) else 1),
    betas=(0.0, 0.0) 
)

VARIABLE_TYPES = [
    WVariable
]

AIM_FOR_FAKE_A = False
AIM_FOR_FAKE_B = False

TARGET_A_PATH = "./datasets/afhq2/train/cat/flickr_cat_000004.png"
TARGET_B_PATH = "./datasets/afhq2/train/cat/flickr_cat_000007.png"

# TARGET_A_PATH = "./datasets/afhq2/train/cat/flickr_cat_000006.png"
# TARGET_B_PATH = "./datasets/afhq2/train/cat/flickr_cat_000018.png"

# TARGET_A_PATH = "datasets/afhq2/train/cat/pixabay_cat_000077.png"
# TARGET_B_PATH = "datasets/afhq2/train/cat/pixabay_cat_004220.png"

# TARGET_A_PATH = "./datasets/afhq2/train/cat/flickr_cat_000436.png"
# TARGET_B_PATH = "./datasets/afhq2/train/cat/pixabay_cat_004436.png"

SEED = 11

for VARIABLE_TYPE in VARIABLE_TYPES:
    print(VARIABLE_TYPE.__name__)
    OUT_DIR = f"out/invert/{METHOD}/{VARIABLE_TYPE.__name__}"

    if __name__ == "__main__":
        if not os.path.exists(OUT_DIR):
            os.makedirs(OUT_DIR)
        else:
            for file in os.listdir(OUT_DIR):
                # * Removes files within the directory so that vscode refresh doesn't get confused
                shutil.rmtree(file, ignore_errors=True)

        G = open_generator(G_PATH)

        # U, S, V = G.mapping.fc0.weight.svd()
        # U, S, V = G.synthesis.b128.conv0.affine.weight.svd()
        # plt.plot(S.detach().cpu())
        # plt.savefig("tmp.png")
        # breakpoint()
        # quit()

        if DISABLE_NORMALIZE:
            G.mapping.normalize = False
        if FORCE_NORMALIZE:
            G.mapping.normalize = True
        if FORCE_LERP:
            ZVariable.interpolate = WVariable.interpolate

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
                fine_tune_G=FINE_TUNE_G,
                constraints=[]
            )

        torch.manual_seed(SEED)

        target_A = (
            sample_image(G) if AIM_FOR_FAKE_A else open_target(G, TARGET_A_PATH)
        )
        target_B = (
            sample_image(G) if AIM_FOR_FAKE_B else open_target(G, TARGET_B_PATH)
        )   

        if TRANSFORM_TARGETS:
            with torch.no_grad():
                target_A = target_A.roll(20, dims=[3])

        A = invert_target(target_A, "A", VARIABLE_TYPE.sample_from(G))
        
        B = invert_target(
            target_B,
            "B",
            A.final_variable.copy() if SEQUENTIAL else VARIABLE_TYPE.sample_from(G), 
        )

        Inversion.save_losses_plot(dict(A=A, B=B), f"{OUT_DIR}/losses.png")

        z_mean = ZVariableInitAtMean.sample_from(G)
        save_image(
            z_mean.to_image(), f"{OUT_DIR}/mean_z.png"
        )
        w_mean = WVariable.sample_from(G)
        save_image(
            w_mean.to_image(), f"{OUT_DIR}/mean_w.png"
        )

        A.save_optim_trace(f"{OUT_DIR}/trace_A.png")
        B.save_optim_trace(f"{OUT_DIR}/trace_B.png")

        if VARIABLE_TYPE is WVariable:
            Interpolator(G).interpolate(
                A.final_variable,
                w_mean,
            ).save(f"{OUT_DIR}/interpolation_A_to_mean_w.png")
            Interpolator(G).interpolate(
                B.final_variable,
                w_mean,
            ).save(f"{OUT_DIR}/interpolation_B_to_mean_w.png")

        interpolation = Interpolator(G).interpolate(
            A.final_variable,
            B.final_variable,
        )

        interpolation.save(f"{OUT_DIR}/interpolation_A_to_B.png")

        latent_distance = interpolation.latent_distance(nn.MSELoss()).item()

        ppl = interpolation.ppl(criterion)
        endpoint_distance = interpolation.endpoint_distance(criterion)

        ppl_l2 = interpolation.ppl(nn.MSELoss())
        endpoint_distance_l2 = interpolation.endpoint_distance(nn.MSELoss())

        a_norm = A.final_variable.data.norm(dim=-1).mean().item()
        b_norm = B.final_variable.data.norm(dim=-1).mean().item()

        idxs = list(range(A.final_variable.data.ndim))[1:]
        a_matrix_norm = A.final_variable.data.norm(dim=idxs).item()
        b_matrix_norm = B.final_variable.data.norm(dim=idxs).item()

        from PIL import Image, ImageDraw

        montage = Image.new("RGB", (840, 600), color=(255, 255, 255))
        draw = ImageDraw.Draw(montage)
        montage.paste(Image.open(f"{OUT_DIR}/optim_progress_A.png"), (200, 75))
        montage.paste(
            Image.open(f"{OUT_DIR}/optim_progress_B.png"), (200, 75 + 6 * 66)
        )
        draw.text((200, 60), "Target A", (0, 0, 0))
        draw.text((200 + 68, 60), "Recon A", (0, 0, 0))
        draw.text((200 + 2 * 68, 60), "Error A", (0, 0, 0))

        draw.text((200, 458), "Target B", (0, 0, 0))
        draw.text((200 + 68, 458), "Recon B", (0, 0, 0))
        draw.text((200 + 2 * 68, 458), "Error B", (0, 0, 0))

        draw.text((550 - 50, 60), "Recon A", (0, 0, 0))
        draw.text((550 - 50, 445 + 68 + 28), "Recon B", (0, 0, 0))

        O = 80
        H = -100
        draw.text((550 + O, 185 + H), "Mean in Z", (0, 0, 0))
        montage.paste(Image.open(f"{OUT_DIR}/mean_z.png"), (550 + O, 200 + H))
        montage.paste(Image.open(f"{OUT_DIR}/mean_w.png"), (550 + O, 300 + H))
        draw.text((550 + O, 285 + H), "Mean in W (empiric)", (0, 0, 0))

        draw.text(
            (600, 485 - 50),
            f"Mean Norm A: {round(a_norm, 8)}",
            (0, 0, 0),
        )
        draw.text(
            (600, 485 - 35),
            f"Mean Norm B: {round(b_norm, 8)}",
            (0, 0, 0),
        )

        draw.text(
            (600, 485 - 50 - 50),
            f"Matrix Norm A: {round(a_matrix_norm, 8)}",
            (0, 0, 0),
        )
        draw.text(
            (600, 485 - 35 - 50),
            f"Matrix Norm B: {round(b_matrix_norm, 8)}",
            (0, 0, 0),
        )

        draw.text(
            (600, 485), f"Latent Distance: {round(latent_distance, 8)}", (0, 0, 0)
        )
        draw.text(
            (600, 500 + 10), f"Percep. Path Length: {round(ppl, 8)}", (0, 0, 0)
        )
        draw.text(
            (600, 515 + 10),
            f"Endpoint Distance: {round(endpoint_distance, 8)}",
            (0, 0, 0),
        )
        draw.text(
            (600, 530 + 20),
            f"Percep. Path Length L2: {round(ppl_l2, 8)}",
            (0, 0, 0),
        )
        draw.text(
            (600, 545 + 20),
            f"Endpoint Distance L2: {round(endpoint_distance_l2, 8)}",
            (0, 0, 0),
        )

        montage.paste(
            Image.open(f"{OUT_DIR}/losses.png").resize((350, 250)), (125, 170)
        )
        montage.paste(
            Image.open(f"{OUT_DIR}/interpolation_A_to_B.png"), (550 - 50, 75)
        )
        montage.save(f"{OUT_DIR}/montage.png")
