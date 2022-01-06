from ..prelude import *

from .eval import *

sys.path.append("vendor/FID_IS_infinity")

from pytorch_fid.fid_score import calculate_fid_given_paths

# from score_infinity import calculate_FID_infinity_path


def compute_fid(dataloader: InvertedDataloader, path: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_avail_cpus = len(os.sched_getaffinity(0))
    num_workers = min(num_avail_cpus, 8)
    return torch.tensor(
        calculate_fid_given_paths(
            [dataloader.target_dataloader.fid_data_path, path],
            50,
            device,
            2048,
            num_workers,
        )
    )
