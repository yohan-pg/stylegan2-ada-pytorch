from inversion import *
from edits import *
PATH = "out/denoise.png"


if __name__ == "__main__":
    G = open_generator("pretrained/tmp.pkl").eval()
    E = open_encoder("out/encoder_0.1/encoder-snapshot-000050.pkl")
    target = open_target(
        G,
        # "datasets/afhq2/test/cat/flickr_cat_000176.png",
        "datasets/afhq2/test/cat/flickr_cat_000368.png",
        "datasets/afhq2/test/cat/pixabay_cat_002905.png",
        "datasets/afhq2/test/cat/pixabay_cat_002997.png"
    )
    
    def add_noise(x):
        return x + torch.randn_like(x) * 0.15

    def lift(x):
        if x is target:
            return x 
        else:
            return add_noise(x)

    def present(x): #!! fail, contains residual oise
        return x

    target_no_noise = target
    target = add_noise(target) 
    
    

    edit(
        "denoise",
        G, 
        E, 
        target, 
        lift=lift,
        present=present,
        encoding_weight=0.15,
        truncation_weight=0.1
    )


    # for i, (inversion, did_snapshot) in enumerate(
    #     tqdm.tqdm(inverter.all_inversion_steps(target), total=len(inverter))
    # ):
    #     if i == 0:
    #         save_image(inversion.variables[0].to_image(), "out/denoise_init.png")

    #     if did_snapshot:
    #         with torch.no_grad():
    #             target_resampled = target_no_noise
    #             pred_resampled = fi(lift(inversion.final_pred))
    #             trunc = inversion.final_variable.to_image(truncation=0.75)
    #             save_image(
    #                 torch.cat(
    #                     (
    #                         target_resampled,
    #                         target,
    #                         inversion.final_pred,
    #                         (target_resampled - pred_resampled).abs(),
    #                         trunc,
    #                         (target_resampled - fi(lift(trunc))).abs()
    #                     )
    #                 ),
    #                 PATH,
    #                 nrow=len(target),
    #             )