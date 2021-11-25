from inversion import *

# todo consider gradient truncation as per https://research.cs.cornell.edu/langevin-mcmc/data/paper.pdf
# todo try an LR rampup?

from inversion.radam import RAdam

METHOD = "adaconv"
G_PATH = "pretrained/tmp.pkl"


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

        U, S, V = cache(G_PATH, compute_ganspace(G), 17)
        origins = WVariable.sample_random_from(G, 8, truncation_psi=0.5).data
        imgs = [WVariable(G, origins).to_image()]
        for i in range(10):
            imgs.append(
                WVariable(G, origins + (V[i] * S[i] / 10).reshape(1, 1, 512)).to_image()
            )

        save_image(torch.cat(imgs), "out/ganspace.png", nrow=len(origins))
