from .prelude import *
from .variables import *

import pickle

# todo move this into a class which opens G than has IO methods?


def save_pickle(dict_of_models: dict, path: str):
    snapshot_data = {}

    for name, module in dict_of_models.items():
        if module is not None:
            # todo
            # if num_gpus > 1:
            #     misc.check_ddp_consistency(module, ignore_regex=r".*\.w_avg")
            module = copy.deepcopy(module).eval().requires_grad_(False).cpu()
        snapshot_data[name] = module
        del module

    # if rank == 0: #todo
    with open(path, "wb") as f:
        pickle.dump(snapshot_data, f)


def open_generator_and_discriminator(pkl_path: str):
    print(f"Loading {pkl_path}...")

    with dnnlib.util.open_url(pkl_path) as fp:
        pkl = legacy.load_network_pkl(fp)
        return (pkl["G_ema"].cuda().eval(), pkl["D"].cuda().eval())


def open_encoder(pkl_path: str):
    print(f"Loading {pkl_path}...")

    with dnnlib.util.open_url(pkl_path) as fp:
        pkl = legacy.load_network_pkl(fp)
        return pkl["E"].cuda().eval()


def open_generator(pkl_path: str):
    print(f"Loading generator from {pkl_path}...")

    with dnnlib.util.open_url(pkl_path) as fp:
        return legacy.load_network_pkl(fp)["G_ema"].cuda().eval()


# todo fuse with open-gen
def open_discriminator(pkl_path: str):
    print(f"Loading discriminator from {pkl_path}...")

    with dnnlib.util.open_url(pkl_path) as fp:
        return legacy.load_network_pkl(fp)["D"].cuda().eval()


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
    return torch.tensor(target_uint8.transpose([2, 0, 1])).cuda().unsqueeze(0) / 255


@torch.no_grad()
def sample_image(G, batch_size: int = 1):
    return (G.synthesis(ZVariable.sample_from(G, batch_size).to_styles()) + 1) / 2


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