from inversion import *

from training.dataset import ImageFolderDataset
import training.networks as networks 


def measure_image_editing_consistency(
    name: str,
    dataloader: Iterable[torch.Tensor],
    inverter: Inverter,
    edit_strength: float = 0.5,
):
    def sample_random_edit():
        return (ZVariable(inverter.G).to_w().data - ZVariable(inverter.G).to_w().data) * 0.5

    def edit_consistency(target, edit):
        first_inversion = inverter(target)
        edited_image = first_inversion.final_variable.edit(edit).to_image()
        second_inversion = inverter(edited_image)
        unedited_image = second_inversion.final_variable.data.edit(-edit).to_image()
        inverter.criterion(unedited_image, target)

    losses = []

    for i, target in enumerate(dataloader):
        edit = sample_random_edit()

        result = edit_consistency(target, edit)
        
        # first_inversion.final_variable.data += edit
        # first_reconstruction = first_inversion.final_variable.to_image()
        # first_inversion.final_variable.data -= edit

        # all_losses.append(inversion.losses)

    return losses
    # return all_losses

def run():
    pass

if __name__ == "__main__":
    pass