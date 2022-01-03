from prelude import *

from colorize import Colorize
from denoise import Denoise
from deartifact import Deartifact
from upsample import Upsample
from inpaint import Inpaint

#! we need an alternative to refinement
#! implement independant solve as a baseline compariason => overload solve, disable refinement?
#! The pasting does not seem to work so well. How can we fix it?
#! colorization fucks everything up. Why?
# ? how to benefit from the encoding weight and truncation weight for individual tasks?

@dataclass(eq=False)
class Joint(Edit):
    edits: List[Edit] = None

    encoding_weight: ClassVar[float] = 0.25
    truncation_weight: ClassVar[float] = 0.05
    
    def parameters(self, recurse: bool = True) -> Iterator[nn.Parameter]:
        return sum([list(edit.parameters(recurse=recurse)) for edit in self.edits], [])

    def initialize(self, image_shape: torch.Size):
        for edit in self.edits:
            edit.initialize(image_shape) 

    def f(self, pred):
        for edit in self.edits:
            pred = edit.f(pred)

        return pred

    def f_ground_truth(self, ground_truth):
        for edit in self.edits:
            ground_truth = edit.f_ground_truth(ground_truth)

        return ground_truth
    
    def penalty(self, var, pred, target):
        return sum([edit.penalty(var, pred, target) for edit in self.edits])

if __name__ == "__main__":
    run_edit_on_examples(
        Joint(
            edits=[
                Upsample(scale=4),
                # Denoise(),
                # Deartifact(quality=10),
                # Inpaint(),
                Colorize(),
            ]
        ),
        num_steps=2000
    )
