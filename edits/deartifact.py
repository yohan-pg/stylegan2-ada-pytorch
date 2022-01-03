from prelude import *

sys.path.append("vendor/DiffJPEG")
from DiffJPEG import DiffJPEG, DiffJPEGMock  # type: ignore
import io

#!!! the clamping was removed from the diffjpeg library
#!!! the rounding was modified in the diffjpeg library

#! the pasting looks bad. what gives?
#! uses the diff approx for the target as well. This needs to be adressed.
#! assumes knowledge of quantization tables. Maybe we can optimize over the quality level for starters?
#! i need to check what quantization table PIL image uses
#! need to plot the quantization tables
#! tables are shared over batch elements, which may be undesired!

@dataclass(eq=False)
class Deartifact(Edit):
    encoding_weight: ClassVar[float] = 0.20
    truncation_weight: ClassVar[float] = 0.06
    
    quality: int = 10

    def parameters(self, recurse: bool = True) -> Iterator[nn.Parameter]: 
        return []  # * Used to avoid optimizing params in `compress_jpeg`

    def initialize(self, image_shape: torch.Size):
        self.diff_compress_jpeg = DiffJPEG( 
            height=image_shape[2],
            width=image_shape[3],
            differentiable=True,
            quality=self.quality,
        ).cuda()

    def f(self, pred):
        return self.diff_compress_jpeg(pred, skip_rounding=False) 

    # def f_ground_truth(self, pred):
    #     return self.diff_compress_jpeg(pred, skip_rounding=False) 

    def compress_to_real_jpeg(self, image_tensor: torch.Tensor) -> List:
        images = []
        
        for element in image_tensor:
            data = io.BytesIO()
            TF.functional.to_pil_image(element.cpu()).save(
                data, format="jpeg", quality=self.quality
            )
            images.append(PIL.Image.open(data))

        return images

    # def f_ground_truth(self, raw_target):
    #     result = torch.stack(
    #         [TF.functional.to_tensor(image) for image in self.compress_to_real_jpeg(raw_target)]
    #     ).to(raw_target.device)
    #     # save_image(torch.cat([self.f(raw_target), result]), "tmp/artifact.png" , nrow=len(result))
    #     return result

    # def log(self, raw_target):
    #     self.diff_compress_jpeg.compress.c_quantize.c_table
    #     self.diff_compress_jpeg.compress.y_quantize.y_table
    #     self.compress_to_real_jpeg(raw_target)[0].quantization

if __name__ == "__main__":
    deartifact = run_edit_on_examples(Deartifact(), num_steps=1000) #!!!
