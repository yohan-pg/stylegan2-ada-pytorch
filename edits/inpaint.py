from prelude import *

#! the mean image having a black background is what gives black cats? or maybe its what the encoder learned 
#! try to truncate towards the initial cat instead of towards the mean 
#! the left half looks good initially. what happens to it?


# todo generic method for fading a mask (gaussian blur?) => only if refinement remains
# todo try on weird mask patterns

@dataclass(eq=False)
class Inpaint(Edit):
    encoding_weight: ClassVar[float] = 0.00 #10.0 
    truncation_weight: ClassVar[float] = 0.0 #0.2

    faded_mask_fraction: float = 0.1

    def select_right_half(self, x):
        return x[:, :, :, x.shape[-1] // 2 :]

    def pad_left_half(self, x):
        x = torch.cat((torch.zeros_like(x), x), dim=3)

        if self.faded_mask_fraction > 0.0:
            l = x.shape[-1] // 2
            r = round(l * (1 + self.faded_mask_fraction))
            x[:, :, :, l:r] *= torch.arange(0.0, 1.0, 1 / (r - l), device=x.device)
        
        return x

    def f(self, pred):
        return self.pad_left_half(self.select_right_half(pred))

if __name__ == "__main__":
    run_edit_on_examples(Inpaint())

