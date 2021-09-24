from inversion import *
import dnnlib
from torch_utils import misc

import itertools

class EncodingDataset:
    def __init__(self, path: str):
        self.dataset = dnnlib.util.construct_class_by_name(
            class_name="training.dataset.ImageFolderDataset",
            path=path,
            use_labels=False,
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i: int):
        targets, _ = self.dataset.__getitem__(i)
        return torch.tensor(targets) / 255

    def to_loader(
        self, batch_size: int, subset_size: Optional[int] = None, step: int = 5, offset: int = 1
    ): 
        subset = (
            torch.utils.data.Subset(
                self, [((x * step) + offset) % len(self) for x in range(subset_size)]
            )
            if subset_size is not None
            else self
        )
        return torch.utils.data.DataLoader(
            subset,
            sampler=misc.InfiniteSampler(dataset=subset, shuffle=False),
            batch_size=batch_size,
            pin_memory=True,
            num_workers=0,
        )