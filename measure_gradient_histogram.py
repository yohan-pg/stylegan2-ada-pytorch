from inversion import *

# todo slerp
# todo clean up image/target image range
# todo double check that VGG loss inputs are the right size and range and everything

import matplotlib.pyplot as plt 

OUT_DIR = f"out"
BATCH_SIZE = 1

TARGET_PATH = "./datasets/afhq2/train/cat/flickr_cat_000006.png"
# G_PATH = f"pretrained/adaconv-slowdown-all.pkl"
# G_PATH = "pretrained/no_torgb_adaconv_untrained.pkl"
G_PATH = "pretrained/no_torgb_adaconv_tmp.pkl"
# G_PATH = f"pretrained/adain-their-params-003800.pkl"
# G_PATH = "pretrained/adaconv_latent_dropout_002200.pkl"
# G_PATH = "pretrained/adain-dropout.pkl"
CRITERION = nn.MSELoss

if __name__ == "__main__":
    G = open_generator(G_PATH)
    
    target = open_target(G, TARGET_PATH)

    var = SVariable.sample_from(G, batch_size=BATCH_SIZE)
    criterion = CRITERION()
    init = var.copy()

    (criterion(var.to_image(), target)).mean().backward()

    print(var.data.grad.abs().max())

    plt.hist(var.data.grad.flatten().cpu().numpy(), bins=100)
    plt.yscale('log', nonpositive='clip')

    plt.savefig("out/grad_hist.png")

        
