from inversion import *

from inversion.radam import RAdam


METHOD = "adaconv"
PATH = "out/scribbles.png"
G_PATH = "pretrained/tmp.pkl"

# todo: what about 


if __name__ == "__main__":
    G = open_generator(G_PATH).eval()
    D = open_discriminator(G_PATH)
    E = open_encoder("out/encoder_W/encoder-snapshot-000150.pkl")

    SCALE = 64 

    def scribble(x):
        return x.min(edits)

    def paste(target):
        def do_paste(x):
            return scribble(x)
        return do_paste

    criterion = 100 * VGGCriterion()

    clean_target = open_target(
        G,
        "datasets/samples/not_angry_cat.png",
        "datasets/samples/spotless_cat.png"
    )
    edits = open_target(
        G,
        "datasets/samples/scribles.png",
        "datasets/samples/spots.png"
    )
    target = scribble(clean_target)

    inverter = Inverter(
        G,
        5000,
        variable_type=add_encoder_constraint(WVariable, E, 0.5, 0.05, paste(target)),
        create_optimizer=lambda params: AdamWithNoise(params, lr=0.1),
        create_schedule=lambda optimizer: lr_scheduler.LambdaLR(
            optimizer, lambda epoch: min(1.0, epoch / 100.0) 
        ),
        criterion=criterion,
        snapshot_frequency=50,
        seed=7,
    )

    for i, (inversion, did_snapshot) in enumerate(
        tqdm.tqdm(inverter.all_inversion_steps(clean_target), total=len(inverter))
    ):
        if i == 0:
            save_image(inversion.variables[0].to_image(), "out/scribbles_init.png")

        if did_snapshot:
            with torch.no_grad():
                target_resampled = scribble(target)
                pred_resampled = scribble(inversion.final_pred)
                trunc = inversion.final_variable.to_image(truncation=0.75)
                save_image(
                    torch.cat(
                        (
                            target,
                            target_resampled,
                            inversion.final_pred,
                            (target_resampled - pred_resampled).abs(),
                            trunc,
                            (target_resampled - scribble(trunc)).abs()
                        )
                    ),
                    PATH,
                    nrow=len(target),
                )