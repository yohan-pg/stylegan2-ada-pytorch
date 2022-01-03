from inversion import *

# todo slerp
# todo clean up image/target image range
# todo double check that VGG loss inputs are the right size and range and everything



OUT_DIR = f"out"
BATCH_SIZE = 12

if __name__ == "__main__":
    # for METHOD_NAME, G_PATH in {
    #     "adain": "pretrained/adain-their-params-003800.pkl",
    #     "adaconv": "pretrained/adaconv-gamma-20-003800.pkl"
    # }.items():
    METHOD_NAME, G_PATH = "adaconv-slow", "pretrained/adaconv-slowdown-all.pkl"
    with torch.no_grad():
        G = open_generator(G_PATH)

        images = []

        Interpolation.from_variables(
            # ZVariableInitAtMean.sample_from(G, batch_size=BATCH_SIZE).to_W(),
            WVariable.sample_from(G, batch_size=BATCH_SIZE),
            WVariable.sample_random_from(G, batch_size=BATCH_SIZE),
            gain=2.0,
            num_steps=14
        ).save(OUT_DIR + f"/fake_interpolation_{METHOD_NAME}.png")
