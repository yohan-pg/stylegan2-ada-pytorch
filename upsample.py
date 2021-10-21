from inversion import *

METHOD = "adaconv"
# G_PATH = f"pretrained/adaconv-normalized.pkl"
G_PATH = f"pretrained/adaconv-slowdown-all.pkl"
PATH = "out/upsample.png"

# todo do this in vgg space
class GradientNormPenalty:
    def __init__(self, weight: float, vgg):
        self.weight = weight
        self.vgg = vgg

    def __call__(self, variable, styles, pred, target, loss):
        return (
            self.weight
            * torch.autograd.grad(
                self.vgg(pred.clone() * 255, resize_images=False, return_lpips=True).abs().sum(),
                # pred.abs().sum(),
                variable.data,
                create_graph=True,
            )[0].norm(dim=(1, 2), p=1)
        )
        
if __name__ == "__main__":
    G = open_generator(G_PATH)
    D = open_discriminator(G_PATH)

    criterion = DownsamplingVGGCriterion(downsample)
    target = open_target(G, "./datasets/afhq2/train/cat/pixabay_cat_004436.png"
    
    # "./datasets/afhq2/train/cat/flickr_cat_000436.png"
    )
    target_low_res = downsample(target)

    inverter = Inverter(
        G,
        1000,
        ZVariable,
        # make_ZVariableWithNoise(ZVariable.default_lr) if True else ZVariable,
        criterion=criterion,
        # make_ZVariableWithNoise(ZVariable.default_lr) if True else ZVariable
        penalties=[GradientNormPenalty(1e-7, criterion.vgg16)],
        learning_rate=0.01,
        snapshot_frequency=10
    )

    for inversion in inverter(target):
        target_resampled = upsample(downsample(target))
        pred_resampled = upsample(downsample(inversion.final_pred))
        save_image(
            torch.cat((target, target_resampled, inversion.final_pred, pred_resampled, (inversion.final_pred - target).abs())),
            PATH
        )
    