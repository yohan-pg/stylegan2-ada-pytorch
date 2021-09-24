from inversion import *

# todo slerp
# todo clean up image/target image range
# todo double check that VGG loss inputs are the right size and range and everything

from torchvision.utils import save_image


METHOD = "adaconv"
G_PATH = "pretrained/resnet-adaconv-000800.pkl"
# G_PATH = f"pretrained/resnet-{METHOD}-000800.pkl"


OUT_DIR = f"out"
BATCH_SIZE = 12


if __name__ == "__main__":
    with torch.no_grad():
        G = open_generator(G_PATH)

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
