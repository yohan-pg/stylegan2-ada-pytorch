from .prelude import *
from .variables import *

# import training.networks

# todo move this into a class which opens G than has IO methods?
# class Generator(training.networks.Generator):
#     pass

def open_pkl(pkl_path: str):
    print(f"Loading networks from {pkl_path}...")

    with dnnlib.util.open_url(pkl_path) as fp:
        return legacy.load_network_pkl(fp)["G_ema"].cuda().eval().requires_grad_(False)  # type: ignore

        
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