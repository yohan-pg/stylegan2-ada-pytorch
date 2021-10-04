import torch 

import tqdm

def sample_mix(G, batch_size):
    style_A = G.mapping(
        torch.randn(batch_size, G.num_required_vectors(), G.w_dim).squeeze(1).cuda(), None
    )
    style_B = G.mapping(
        torch.randn(batch_size, G.num_required_vectors(), G.w_dim).squeeze(1).cuda(), None
    )

    images = []  
    for i in tqdm.tqdm(list(reversed(range(G.num_ws + 1)))):
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

    return torch.cat(images)


# todo: untrained network 
# todo: what parts of the net has small # of channels?

# ???? why is it the first few vectors in the matrix???
# ? maybe the cutting up into chunks is bad? vectors that keep their same position get used more or something?

# ? add a positional encoding or something?


def sample_latent_perturbations(G, batch_size):
    latent_A = torch.randn(batch_size, G.num_required_vectors(), G.w_dim).squeeze(1).cuda()
    latent_B = torch.randn(batch_size, G.num_required_vectors(), G.w_dim).squeeze(1).cuda()

    images = []
    for i in tqdm.tqdm(list(range(G.num_required_vectors() + 1))):
        images.append(
            (G.synthesis(
                G.mapping(torch.cat(
                        (
                            latent_A[:, : i, :].clone(),
                            latent_B[:, i :, :].clone(),
                        ),
                        dim=1,
                    ),None),
                noise_mode="const",
            ) + 1) / 2
        )

    return torch.cat(images)


def sample_style_pertubations(G, batch_size):
    style_A = G.mapping(
        torch.randn(batch_size, G.num_required_vectors(), G.w_dim).squeeze(1).cuda(), None
    )
    style_B = G.mapping(
        torch.randn(batch_size, G.num_required_vectors(), G.w_dim).squeeze(1).cuda(), None
    )

    images = []
    for i in tqdm.tqdm(list(range(G.num_required_vectors() + 1))):
        images.append(
            (G.synthesis(
                torch.cat(
                        (
                            style_A[:, : G.w_dim, :][:, : i, :].clone(),
                            style_B[:, : G.w_dim, :][:, i :, :].clone(),
                        ),
                        dim=1,
                    ).repeat(1, G.num_ws, 1),
                noise_mode="const",
            ) + 1) / 2
        )

    return torch.cat(images)