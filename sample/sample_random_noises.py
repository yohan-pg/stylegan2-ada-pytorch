from inversion import *

# todo slerp
# todo clean up image/target image range
# todo double check that VGG loss inputs are the right size and range and everything

from torchvision.utils import save_image

METHOD = "adaconv"
G_PATH = f"pretrained/alpha-{METHOD}-002600.pkl"
OUT_DIR = f"out"

if __name__ == "__main__":
    G = open_generator(G_PATH)

    z = torch.randn(1, G.num_required_vectors(), G.w_dim).squeeze(1).repeat(32, 1, 1).cuda()

    images = (G(z, None) + 1) / 2
    save_image(images, "out/same_style_random_noises.png")
