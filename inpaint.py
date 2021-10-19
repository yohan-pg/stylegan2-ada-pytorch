from inversion import *

METHOD = "adaconv"
G_PATH = f"pretrained/adaconv-256-001200.pkl"

def make_mask(x):
    x = x.clone()
    mask = torch.ones_like(x)
    _, _, H, W = x.shape
    mask[:, :, :, W//3*2:] = 0.0
    return mask

if __name__ == "__main__":
    G = open_generator(G_PATH)
    
    target = open_target(G, 
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
    
    criterion = MaskedVGGCriterion(mask)
    inverter = Inverter(
        G,
        500,
        ZVariableConstrainToTypicalSetAllVecs,
        criterion=criterion,
        learning_rate=0.03
    )

    inversion =inverter(target * mask, out_path="out/inpaint.png")
    save_image(torch.cat((target, mask, target * mask, inversion.final_pred, (inversion.final_pred * (1.0 - mask)) + target * mask )), "out/inpainting_result.png")