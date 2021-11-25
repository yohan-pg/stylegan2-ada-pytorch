from inversion import *


METHOD = "adaconv"
PATH = "out/style_transfer.png"
G_PATH = "pretrained/tmp.pkl"

BLUR = True


def select_left_region(x):
    return x[:, :, :, x.shape[-1] // 2 :]


class StyleTransferCriterion(InversionCriterion):
    def __init__(self, criterion, style_target, style_weight=100_000_000.0):
        super().__init__()
        self.criterion = criterion
        self.style_target = style_target
        self.style_distance = nn.MSELoss()
        self.style_gram = self.gram_matrix(self.style_target)
        self.style_weight = style_weight

    def gram_matrix(self, image):
        features = self.criterion.extract_features(image).reshape(len(image), -1, 128 * 128)
        return features.bmm(features.transpose(1, 2))

    def forward(self, pred: ImageTensor, target: ImageTensor):
        return self.criterion(pred, target) + self.style_weight * self.style_distance(
            self.gram_matrix(pred),
            self.style_gram,
        )

if __name__ == "__main__":
    G = open_generator(G_PATH).eval()
    E = open_encoder("out/encoder_W/encoder-snapshot-000150.pkl")

    SCALE = 64

    def f(x):
        return x

    def fi(x):
        return x

    def paste(target):
        def do_paste(x):
            return x

        return do_paste

    target = open_target(
        G,
        "datasets/afhq2/test/cat/pixabay_cat_002905.png",
        "datasets/afhq2/test/cat/pixabay_cat_002997.png",
    )
    style_target = open_target(G, "datasets/samples/starry_night_full.jpeg").repeat_interleave(len(target), dim=0)
    criterion = 100 * StyleTransferCriterion(
        VGGCriterion(), 
        style_target
    )

    inverter = Inverter(
        G,
        5000,
        variable_type=add_encoder_constraint(WVariable, E, 0.1, 0.05, paste(target)),
        create_optimizer=lambda params: AdamWithNoise(params, lr=0.1),
        create_schedule=lambda optimizer: lr_scheduler.LambdaLR(
            optimizer, lambda epoch: min(1.0, epoch / 100.0)
        ),
        criterion=criterion,
        snapshot_frequency=50,
        seed=7,
    )

    for i, (inversion, did_snapshot) in enumerate(
        tqdm.tqdm(inverter.all_inversion_steps(target), total=len(inverter))
    ):
        if i == 0:
            save_image(inversion.variables[0].to_image(), "out/style_transfer_init.png")

        if did_snapshot:
            with torch.no_grad():
                target_resampled = f(target)
                pred_resampled = f(inversion.final_pred)
                trunc = inversion.final_variable.to_image(truncation=0.75)
                save_image(
                    torch.cat(
                        (
                            target,
                            style_target,
                            inversion.final_pred,
                            pred_resampled,
                            (target - inversion.final_pred).abs(),
                            (target_resampled - pred_resampled).abs(),
                            trunc,
                            (target_resampled - fi(f(trunc))).abs(),
                        )
                    ),
                    PATH,
                    nrow=len(target) * 2,
                )
