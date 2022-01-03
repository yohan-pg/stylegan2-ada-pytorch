from .prelude import *
from .variables import *

import pickle
import shutil


from training.dataset import ImageFolderDataset
import training.networks as networks

# todo move this into a class which opens G than has IO methods?


@torch.no_grad()
def save_pickle(dict_of_models: dict, path: str):
    snapshot_data = {}

    for name, module in dict_of_models.items():
        # if module is not None:
        #     # todo
        #     # if num_gpus > 1:
        #     #     misc.check_ddp_consistency(module, ignore_regex=r".*\.w_avg")
        #     module = copy.deepcopy(module).eval().requires_grad_(False).cpu()
        snapshot_data[name] = module
        del module

    # if rank == 0: #todo
    with open(path, "wb") as f:
        pickle.dump(snapshot_data, f)


def open_generator_and_discriminator(pkl_path: str):
    print(f"Loading {pkl_path}...")

    with dnnlib.util.open_url(pkl_path) as fp:
        pkl = legacy.load_network_pkl(fp)
        return (pkl["G_ema"].cuda().eval(), pkl["D"].cuda().eval(), pkl["training_set_kwargs"])


def open_encoder(pkl_path: str):
    print(f"Loading {pkl_path}...")

    with dnnlib.util.open_url(pkl_path) as fp:
        pkl = legacy.load_network_pkl(fp)
        return pkl["E"].cuda().eval()


def open_generator(pkl_path: str):
    print(f"Loading generator from {pkl_path}...")

    with dnnlib.util.open_url(pkl_path) as fp:
        return legacy.load_network_pkl(fp)["G_ema"].cuda().eval().float()

# todo fuse with open-gen
def open_discriminator(pkl_path: str):
    print(f"Loading discriminator from {pkl_path}...")

    with dnnlib.util.open_url(pkl_path) as fp:
        return legacy.load_network_pkl(fp)["D"].cuda().eval()


import shelve


def cache(key, f, *args):
    key = repr((key, args))
    with shelve.open(f"cache/shelve_{f.__name__}") as db:
        if key not in db:
            db[key] = f(*args)
        return db[key]


def to_G(G_or_E):
    if G_or_E.__class__.__name__ == "Encoder":
        return G_or_E.G[0]
    else:
        return G_or_E

def open_target(G_or_E, *paths: str):
    G = to_G(G_or_E)
    images = []
    for path in paths:
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
        images.append(
            torch.tensor(target_uint8.transpose([2, 0, 1])).cuda().unsqueeze(0) / 255
        )
    return torch.cat(images)


@torch.no_grad()
def sample_image(G, batch_size: int = 1):
    var = ZVariable.sample_from(G, batch_size)
    return (G.synthesis(var.to_styles()) + 1) / 2


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


def latest_snapshot(name: str) -> Optional[str]:
    run_path = sorted(os.listdir(f"training-runs/{name}"))[-1]
    snapshot_path = sorted(
        [
            path
            for path in os.listdir(f"training-runs/{name}/{run_path}")
            if path.startswith("network-snapshot")
        ]
    )[-1]
    return f"training-runs/{name}/{run_path}/{snapshot_path}"


def fresh_dir(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)


class InversionDataloader:
    pass


@dataclass(eq=False)
class RealDataloader(InversionDataloader):
    name = "real"

    dataset_path: str
    batch_size: int
    num_images: int
    seed: int = 0
    fid_data_path: str = None

    def __post_init__(self):
        torch.manual_seed(self.seed)
        dataset = ImageFolderDataset(self.dataset_path)
        self.dataset = torch.utils.data.Subset(
            dataset, torch.randperm(len(dataset))[: self.num_images]
        )

    def __len__(self):
        return self.num_images // self.batch_size

    def subset(self, num_images: int) -> "RealDataloader":
        return RealDataloader(
            self.dataset_path, self.batch_size, self.num_images, self.seed
        )

    def __iter__(self):
        inner_loader = torch.utils.data.DataLoader(
            self.dataset, batch_size=self.batch_size, shuffle=False, drop_last=True
        )

        def loader_transform():
            for img, _ in inner_loader:
                yield img.cuda() / 255

        return loader_transform()


@dataclass(eq=False)
class FakeDataloader(InversionDataloader):
    name = "fake"

    G: networks.Generator
    batch_size: int
    num_images: int

    def __iter__(self):
        for _ in range(self.num_images):
            with torch.no_grad():
                image = ZVariable.sample_from(self.G, self.batch_size).to_image()
            yield image


import itertools


def grouper(iterable, n, fillvalue=None):
    "Collect data into non-overlapping fixed-length chunks or blocks"
    args = [iter(iterable)] * n
    return itertools.zip_longest(*args, fillvalue=fillvalue)


class InvertedDataloader:
    def __init__(self, target_dataloader: InversionDataloader, inverter):
        self.inversions = []
        self.inverter = inverter
        self.target_dataloader = target_dataloader

        for target in tqdm.tqdm(target_dataloader):
            inversion = inverter(target).purge()
            self.inversions.append(inversion)

        self.num_images = self.target_dataloader.num_images

    def __len__(self):
        return len(self.target_dataloader)

    def __iter__(self) -> Iterable:
        for i, _ in enumerate(self.target_dataloader):
            inversion = self.inversions[i]
            inversion.move_to_cuda()
            yield inversion
            inversion.move_to_cuda()


class Parallelize:
    def __init__(self, module: nn.Module):
        self.module = module
        
        if module.__class__.__name__ == "Encoder":
            self.data_parallel = module
            if not isinstance(module.network, nn.DataParallel):
                module.network = nn.DataParallel(module.network)
        elif module.__class__.__name__ == "Generator":
            self.data_parallel = nn.DataParallel(module)
        
    def __getattr__(self, name: str):
        return getattr(self.module, name)

    def __call__(self, *args, **kwargs):
        return self.data_parallel(*args, **kwargs)
    