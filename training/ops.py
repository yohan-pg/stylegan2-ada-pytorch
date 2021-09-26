import torch 


def sample_mix(G, batch_size):
    style_A = G.mapping(
        torch.randn(batch_size, G.num_required_vectors(), G.w_dim).squeeze(1).cuda(), None
    )
    style_B = G.mapping(
        torch.randn(batch_size, G.num_required_vectors(), G.w_dim).squeeze(1).cuda(), None
    )

    images = []  
    for i in reversed(range(G.num_ws + 1)):
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