# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Project given image to the latent space of pretrained network pickle."""

from inversion import *


def open_network_pkl(pkl_path: str):
    print(f"Loading networks from {pkl_path}...")

    with dnnlib.util.open_url(pkl_path) as fp:
        return legacy.load_network_pkl(fp)["G_ema"].requires_grad_(False).cuda()  # type: ignore


def open_target(G, path: str):
    target_pil = PIL.Image.open(path).convert("RGB")
    w, h = target_pil.size
    s = min(w, h)
    target_pil = target_pil.crop(
        ((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2)
    )
    target_pil = target_pil.resize(
        (G.img_resolution, G.img_resolution), PIL.Image.LANCZOS
    )
    target_uint8 = np.array(target_pil, dtype=np.uint8)
    return torch.tensor(target_uint8.transpose([2, 0, 1])).cuda()


def save_video(G, target_uint8, outdir: str, projected_w_steps: list):
    video = imageio.get_writer(
        f"{outdir}/proj.mp4", mode="I", fps=10, codec="libx264", bitrate="16M"
    )
    print(f'Saving optimization progress video "{outdir}/proj.mp4"')
    for projected_w in projected_w_steps:
        synth_image = G.synthesis(projected_w.unsqueeze(0), noise_mode="const")
        synth_image = (synth_image + 1) * (255 / 2)
        synth_image = (
            synth_image.permute(0, 2, 3, 1)
            .clamp(0, 255)
            .to(torch.uint8)[0]
            .cpu()
            .numpy()
        )
        video.append_data(np.concatenate([target_uint8, synth_image], axis=1))
    video.close()


def save_frame(G, outdir: str, target_uint8: list, projected_w_steps: list):
    # Save final projected frame and W vector.
    target_uint8.save(f"{outdir}/target.png")  #!! was target_pil
    projected_w = projected_w_steps[-1]
    synth_image = G.synthesis(projected_w.unsqueeze(0), noise_mode="const")
    synth_image = (synth_image + 1) * (255 / 2)
    synth_image = (
        synth_image.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()
    )
    PIL.Image.fromarray(synth_image, "RGB").save(f"{outdir}/proj.png")
    np.savez(f"{outdir}/projected_w.npz", w=projected_w.unsqueeze(0).cpu().numpy())
    return outdir


def invert(
    G,
    target_path: str,
    outdir: str,
    num_steps: int = 1000,
    save_video: bool = False,
    seed: int = 0,
):
    os.makedirs(outdir, exist_ok=True)
    np.random.seed(seed)
    torch.manual_seed(seed)

    target = open_target(G, target_path)
    start_time = perf_counter()

    # w_out = torch.zeros(
    #     [num_steps] + list(z_opt.shape[1:]), dtype=torch.float32, device=device
    # )

    # # Save projected W for each optimization step.
    # w_out[step] = w_opt.detach()[0]

    # return w_out if w_plus else w_out.repeat([1, G.mapping.num_ws, 1])

    variable = WVariable(G)
    for i, (loss, pred) in enumerate(
        Inverter(G).invert(target, variable, VGGCriterion(), num_steps)
    ):
        if outdir is not None and i % 100 == 0:
            A = target / 255.0
            B = (pred + 1) / 2
            save_image(torch.cat((A, B, (A - B).abs())), f"{outdir}/optim_progress.png")

    # projected_w_steps = project(
    #     G,
    #     target=,  # pylint: disable=not-callable
    #     num_steps=num_steps,
    #     device=device,
    #     verbose=True,
    #     outdir=outdir,
    #     **kwargs,
    # )

    print(f"Elapsed: {(perf_counter() - start_time):.1f} s")

    # if save_video:
    #     save_video(G, target, outdir, projected_w_steps)


if __name__ == "__main__":
    G = open_network_pkl("pretrained/alpha-adaconv-002600.pkl")
    outdir = "out/adaconv-cat"

    invert(G, "./datasets/samples/cats/00000/img00000003.png", outdir)
    
