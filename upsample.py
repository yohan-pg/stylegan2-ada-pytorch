from inversion import *

METHOD = "adaconv"
G_PATH = f"pretrained/adaconv-normalized.pkl"
OUT_DIR = f"out"

if __name__ == "__main__":
    G = open_generator(G_PATH)
    D = open_discriminator(G_PATH)


# TARGET_A_PATH = 
# TARGET_B_PATH = "./datasets/afhq2/train/cat/pixabay_cat_004436.png"
    criterion = DownsamplingVGGCriterion(downsample)
    target = open_target(G, "./datasets/afhq2/train/cat/pixabay_cat_004436.png"
    
    # "./datasets/afhq2/train/cat/flickr_cat_000436.png"
    )
    target_low_res = downsample(target)

    inverter = Inverter(
        G,
        500,
        ZVariable,
        criterion=criterion,
        learning_rate=0.03
    )

    inversion = inverter(target, out_path="out/upsample.png")

    target_resampled = upsample(downsample(target))
    pred_resampled = upsample(downsample(inversion.final_pred))
    save_image(
        torch.cat((target, target_resampled, inversion.final_pred, pred_resampled, (inversion.final_pred - target).abs())),
        "out/upsample.png"
    )
    