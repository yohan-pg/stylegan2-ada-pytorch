from inversion import *

# todo slerp
# todo clean up image/target image range
# todo double check that VGG loss inputs are the right size and range and everything

from torchvision.utils import save_image

# todo adaconv

METHOD = "adaconv"
G_PATH = f"pretrained/alpha-{METHOD}-002600.pkl"
OUT_DIR = f"out"
BATCH_SIZE = 12

if __name__ == "__main__":
    G = open_pkl(G_PATH)

    style_A = G.mapping(
        torch.randn(BATCH_SIZE, G.num_required_vectors(), G.w_dim).squeeze(1).cuda(), None
    )
    style_B = G.mapping(
        torch.randn(BATCH_SIZE, G.num_required_vectors(), G.w_dim).squeeze(1).cuda(), None
    )

    images = []  
    for i in reversed(range(G.num_ws + 1)):
        print(i * G.num_required_vectors())
        images.append(
            (G.synthesis(
                torch.cat(
                    (
                        style_A[:, : i * G.num_required_vectors(), :],
                        style_B[:, i * G.num_required_vectors() :, :],
                    ),
                    dim=1,
                ),
                noise_mode="const",
            ) + 1) / 2
        )

    save_image(torch.cat(images), "out/random_mix.png", nrow=BATCH_SIZE)
