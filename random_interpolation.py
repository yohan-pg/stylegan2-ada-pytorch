from inversion import *

# todo slerp
# todo clean up image/target image range
# todo double check that VGG loss inputs are the right size and range and everything

METHOD = "adain"
G_PATH = "pretrained/resnet-adain-000800.pkl"
# G_PATH = f"pretrained/resnet-{METHOD}-000800.pkl"

OUT_DIR = f"out"
VARIABLE_TYPE = ZVariable
BATCH_SIZE = 12

if __name__ == "__main__":
    with torch.no_grad():
        G = open_generator(G_PATH)

        images = []
        
        Interpolator(G).interpolate(
            VARIABLE_TYPE.sample_from(G, batch_size=BATCH_SIZE),
            VARIABLE_TYPE.sample_from(G, batch_size=BATCH_SIZE),
        ).save(OUT_DIR + f"/fake_interpolation_{METHOD}.png")
