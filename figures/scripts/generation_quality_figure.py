from prelude import *

NUM_SAMPLES = 100

# -----------------------------

print("🛣 Generation Quality")

G = open_generator("pretrained/tmp.pkl")

os.makedirs("figures/out/generation_quality/", exist_ok=True)

for i in range(NUM_SAMPLES):
    torch.manual_seed(i)
    save_image(
        ZVariable.sample_random_from(G, batch_size=1).to_image(),
        f"figures/out/generation_quality/{i}.png",
        nrow=2
    )

