from inversion import *

OUT_DIR = f"out"
BATCH_SIZE = 4
NUM_SAMPLES = 500
GAIN = 0.2
METHOD_NAME, G_PATH = "adaconv-slow", "pretrained/adaconv-slowdown-all.pkl"

if __name__ == "__main__":
    with torch.no_grad():
        G = open_generator(G_PATH)

        z = ZVariable.sample_from(G, batch_size=BATCH_SIZE)
        
        imgs = []
        init_data = z.data.clone()

        for i in tqdm.tqdm(range(NUM_SAMPLES)):
            z.data.copy_(init_data + torch.randn_like(init_data) * GAIN)
            imgs.append(z.to_image())

        save_image(torch.stack(imgs).mean(dim=0), "out/texture_sticking.png")
