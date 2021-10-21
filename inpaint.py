from inversion import *

# METHOD = "adaconv"
# G_PATH = f"pretrained/adaconv-256-001200.pkl"
G_PATH = "pretrained/adain-their-params-003800.pkl"
# G_PATH = f"pretrained/adaconv-slowdown-all.pkl"

def make_mask(x):
    x = x.clone()
    mask = torch.ones_like(x)
    _, _, H, W = x.shape
    mask[:, :, :, W // 3 : W // 3*2] = 0.0
    return mask


if __name__ == "__main__":
    G = open_generator(G_PATH)

    target = open_target(
        G,
        "./datasets/afhq2/train/cat/flickr_cat_000018.png"
        # "datasets/samples/cats/00000/img00000014.png"
        # "./datasets/afhq2/train/cat/flickr_cat_000007.png"
        # "datasets/afhq2/test/cat/pixabay_cat_000542.png"
        # "datasets/afhq2/test/cat/pixabay_cat_002694.png"
        # "datasets/afhq2/test/wild/flickr_wild_001251.png"
        # "./datasets/afhq2/train/dog/flickr_dog_000021.png"
        # "./datasets/afhq2/train/dog/flickr_dog_000022.png"
    )
    mask = make_mask(target)

    # todo do this in vgg space
    class GradientNormPenalty:
        def __init__(self, weight: float, vgg):
            self.weight = weight
            self.vgg = vgg

        def __call__(self, variable, styles, pred, target):
            return (
                self.weight
                * torch.autograd.grad(
                    self.vgg(pred.clone() * 255, resize_images=False, return_lpips=True).sum(),
                    pred.abs().sum(),
                    variable.data,
                    create_graph=True,
                )[0].norm(dim=(1, 2), p=1)
            )

    criterion = MaskedVGGCriterion(mask)

    class FlippedCriterion(MaskedVGGCriterion):
        def forward(self, x):
            pass

    inverter = Inverter(
        G,
        1000,
        # make_ZVariableWithNoise(ZVariable.default_lr) if True else ZVariable,
        WPlusVariable,
        learning_rate=0.03,
        criterion=criterion,
        # learning_rate=ZVariable.default_lr / 3,
        snapshot_frequency=10,
    )

    for inversion in tqdm.tqdm(inverter(target * mask, out_path="out/inpaint.png")):
        save_image(
            torch.cat(
                (
                    target,
                    mask,
                    target * mask,
                    inversion.final_pred,
                    (inversion.final_pred * (1.0 - mask)) + target * mask,
                )
            ),
            "out/inpainting_result.png",
        )
