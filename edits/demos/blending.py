from prelude import *


def select_left_quater(x):
    return x[:, :, :, : x.shape[-1] // 4]


def select_right_quater(x):
    return x[:, :, :, x.shape[-1] // 4 * 3 :]


def select_left_half(x):
    return x[:, :, :, : x.shape[-1] // 2]


def select_right_half(x):
    return x[:, :, :, x.shape[-1] // 2 :]


def mask_out_center(x):
    result = torch.zeros_like(x)
    select_left_quater(result).copy_(select_left_quater(x))
    select_right_quater(result).copy_(select_right_quater(x))
    return result


def paste(x):
    result = x.clone()
    select_left_quater(result).copy_(select_left_quater(left_target))
    select_right_quater(result).copy_(select_right_quater(right_target))
    return result


if __name__ == "__main__":
    G = open_generator("pretrained/tmp.pkl")
    E = open_encoder("out/encoder_0.1/encoder-snapshot-000050.pkl")

    left_target = open_target(
        G,
        "datasets/afhq2/test/cat/flickr_cat_000176.png",
        "datasets/afhq2/test/cat/flickr_cat_000236.png",
        "datasets/afhq2/test/cat/flickr_cat_000368.png",
        "datasets/afhq2/test/cat/pixabay_cat_000117.png",
    )
    right_target = open_target(
        G,
        "datasets/afhq2/test/cat/pixabay_cat_002488.png",
        "datasets/afhq2/test/cat/pixabay_cat_002860.png",
        "datasets/afhq2/test/cat/pixabay_cat_002905.png",
        "datasets/afhq2/test/cat/pixabay_cat_002997.png",
    )
    target = left_target.clone()
    select_right_half(target).copy_(select_right_half(right_target))

    # edit(
    #     "blend",
    #     G,
    #     E,
    #     target,
    #     f=mask_out_center,
    #     paste=paste,
    #     encoding_weight=0.1,
    #     truncation_weight=0.02,
    # )
