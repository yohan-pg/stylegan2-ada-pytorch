from inversion import *


METHOD = "adaconv"
G_PATH = "pretrained/ffhq.pkl"

NUM_BATCHES = 8
NUM_DIRS = 10


def compute_ganspace(G):
    def f(num_batches):
        mu = G.mapping.w_avg.unsqueeze(0)
        ws = (WVariable.sample_random_from(G, num_batches).data - mu).reshape(
            -1, G.mapping.w_dim
        )
        U, S, V = ws.svd()
        return U, S, V.t()

    return f


if __name__ == "__main__":
    with torch.no_grad():
        G = open_generator(G_PATH).eval()
        D = open_discriminator(G_PATH)

        U, S, V = cache(G_PATH, compute_ganspace(G), NUM_BATCHES)
        origins = WVariable.sample_random_from(G, 8, truncation_psi=0.5).data
        
        imgs = [WVariable(G, origins).to_image()]
        for i in range(NUM_DIRS):
            imgs.append(
                WVariable(
                    G, origins + (V[i] * S[i] / math.sqrt(512)).reshape(1, 1, 512)
                ).to_image()
            )

        save_image(torch.cat(imgs), "out/ganspace.png", nrow=len(origins))
