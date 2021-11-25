from prelude import *


if __name__ == "__main__":
    G = open_generator("pretrained/tmp.pkl")


    for i in range(100):
        torch.manual_seed(i)
        save_image(
            ZVariable.sample_random_from(G, batch_size=1).to_image(),
            f"figures/out/generation_quality_{i}.png",
            nrow=2
        )