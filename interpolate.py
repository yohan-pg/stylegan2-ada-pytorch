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

DISABLE_NORMALIZE = False 
FORCE_NORMALIZE = False
FORCE_LERP = False
SAME_SEED = True
SEQUENTIAL = False
TRANSFORM_TARGETS = False

# VARIABLE_TYPE = make_ZVariableWithRelativeDropoutOnZ(0.5)
# VARIABLE_TYPE = make_ZVariableWithFeatureAlphaDropoutOnZ(0.9, 0.995) 
VARIABLE_TYPE = WPlusVariableRandomInit
NUM_STEPS = 1000
CRITERION_TYPE = VGGCriterion
SNAPSHOT_FREQ = 10

AIM_FOR_FAKE_A = False
AIM_FOR_FAKE_B = False

# TARGET_A_PATH = "./datasets/afhq2/train/cat/flickr_cat_000006.png"
# TARGET_B_PATH = "./datasets/afhq2/train/cat/flickr_cat_000018.png"

TARGET_A_PATH = "datasets/samples/my_face_a.png"
TARGET_B_PATH = "datasets/samples/my_face_b.png"

SEED = 2

print(VARIABLE_TYPE.__name__)
OUT_DIR = f"out/invert/{METHOD}/{VARIABLE_TYPE.__name__}"

if __name__ == "__main__":
    fresh_dir(OUT_DIR)

    G = open_generator(G_PATH)
    G.requires_grad_(True)
    G.train()

    if DISABLE_NORMALIZE:
        G.mapping.normalize = False
    if FORCE_NORMALIZE:
        G.mapping.normalize = True
    if FORCE_LERP:
        ZVariable.interpolate = WVariable.interpolate

    criterion = CRITERION_TYPE()

    def invert_target(target: torch.Tensor, name: str, variable_type: Variable):
        inverter = Inverter(
            G, 
            variable_type=variable_type,
            num_steps=NUM_STEPS,
            criterion=100 * criterion,
            create_optimizer=lambda params: torch.optim.Adam(params, lr=0.1),
            # create_optimizer=lambda params: torch.optim.SGD(params, lr=0.01),
            # create_optimizer=lambda params: SGDWithNoise(params, lr=0.1),
            create_schedule=lambda optimizer: lr_scheduler.LambdaLR(
                optimizer, lambda epoch: min(1.0, epoch / 100.0) * 0.998**epoch
            ),
            snapshot_frequency=SNAPSHOT_FREQ,
            seed=SEED if SAME_SEED else None,
        )
        try:
            for inversion, snapshot_iter in tqdm.tqdm(inverter.all_inversion_steps(target)):
                if snapshot_iter:
                    inversion.snapshot(f"{OUT_DIR}/optim_progress_{name}.png")
        except KeyboardInterrupt:
            pass

        return inversion

    torch.manual_seed(SEED)

    target_A = (
        sample_image(G)
        if AIM_FOR_FAKE_A
        else (open_target(G, TARGET_A_PATH))
    )

    target_B = (
        sample_image(G)
        if AIM_FOR_FAKE_B
        else (open_target(G, TARGET_B_PATH))
    )

    if TRANSFORM_TARGETS:
        with torch.no_grad():
            target_A = target_A.roll(20, dims=[3])

    try:
        z_mean = ZVariableInitAtMean.sample_from(G)
        save_image(z_mean.to_image(), f"{OUT_DIR}/mean_z.png")
        w_mean = WVariable.sample_from(G)
        save_image(w_mean.to_image(), f"{OUT_DIR}/mean_w.png")
        s_mean = SVariable.sample_from(G)
        save_image(s_mean.to_image(), f"{OUT_DIR}/mean_s.png")
    except:
        print("Failed to save mean images")
        pass

    A = invert_target(target_A, "A", VARIABLE_TYPE)
    B = invert_target(
        target_B,
        "B",
        A.final_variable.copy() if SEQUENTIAL else VARIABLE_TYPE,
    )

    # Inversion.save_losses_plot(dict(A=A, B=B), f"{OUT_DIR}/losses.png")

    # if VARIABLE_TYPE != WPlusVariable:
    #     Interpolation.from_variables(
    #         w_mean.to_W_plus(), A.final_variable.to_W(), gain=2, num_steps=14
    #     ).save(f"{OUT_DIR}/interpolation_mean_W_to_A.png")
    #     Interpolation.from_variables(
    #         w_mean.to_W_plus(), B.final_variable.to_W(), gain=2, num_steps=14
    #     ).save(f"{OUT_DIR}/interpolation_mean_w_to_B.png")
    
    interpolation = Interpolation.from_variables(
        A.final_variable.disable_noise(), B.final_variable.disable_noise(), gain=1.0
    )
    interpolation.save(f"{OUT_DIR}/interpolation_A_to_B.png", target_A, target_B)

    if hasattr(A.final_variable, 'to_W'):
        interpolation = Interpolation.from_variables(
            A.final_variable.to_W(), B.final_variable.to_W(), gain=1.0
        )

    if False:
        interpolation.save(f"{OUT_DIR}/interpolation_A_to_B_on_W.png", target_A, target_B)
    
        A.save_optim_trace(f"{OUT_DIR}/trace_A.png")
        B.save_optim_trace(f"{OUT_DIR}/trace_B.png")

        latent_distance = interpolation.latent_distance(nn.MSELoss()).item()

        ppl = interpolation.ppl(criterion)
        endpoint_distance = interpolation.endpoint_distance(criterion)
        print("PPL:", ppl, endpoint_distance)

        ppl_l2 = interpolation.ppl(nn.MSELoss())
        endpoint_distance_l2 = interpolation.endpoint_distance(nn.MSELoss())
        print("L2:", ppl_l2, endpoint_distance_l2)

        a_norm = A.final_variable.data.norm(dim=-1).mean().item()
        b_norm = B.final_variable.data.norm(dim=-1).mean().item()
        print("Norm:", a_norm, b_norm)

        idxs = list(range(A.final_variable.data.ndim))[1:]
        a_matrix_norm = A.final_variable.data.norm(dim=idxs).item()
        b_matrix_norm = B.final_variable.data.norm(dim=idxs).item()

        
        from PIL import Image, ImageDraw

        montage = Image.new("RGB", (840, 600), color=(255, 255, 255))
        draw = ImageDraw.Draw(montage)
        montage.paste(Image.open(f"{OUT_DIR}/optim_progress_A.png"), (200, 75))
        montage.paste(Image.open(f"{OUT_DIR}/optim_progress_B.png"), (200, 75 + 6 * 66))
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

        draw.text((600, 485), f"Latent Distance: {round(latent_distance, 8)}", (0, 0, 0))
        draw.text((600, 500 + 10), f"Percep. Path Length: {round(ppl, 8)}", (0, 0, 0))
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

        montage.paste(Image.open(f"{OUT_DIR}/losses.png").resize((350, 250)), (125, 170))
        montage.paste(Image.open(f"{OUT_DIR}/interpolation_A_to_B.png"), (550 - 50, 75))
        montage.save(f"{OUT_DIR}/montage.png")
