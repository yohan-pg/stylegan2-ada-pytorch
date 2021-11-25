from inversion import *



OUT_DIR = f"out"
VARIABLE_TYPE = ZVariable
BATCH_SIZE = 12
METHOD_NAME, G_PATH = "adaconv-slow", "pretrained/adaconv-slowdown-all.pkl"

if __name__ == "__main__":
    with torch.no_grad():
        G = open_generator(G_PATH)

        images = []

        ZVariable.interpolate = WVariable.interpolate
        
        for i in range(1, 11):
            torch.manual_seed(0)
            Interpolation.from_variables(
                ZVariable.sample_random_from(G, batch_size=BATCH_SIZE) * i,
                ZVariable.sample_random_from(G, batch_size=BATCH_SIZE) * i,
            ).save(OUT_DIR + f"/fake_interpolation_{METHOD_NAME}_{i}.png")

