from inversion import *

# todo slerp
# todo clean up image/target image range
# todo double check that VGG loss inputs are the right size and range and everything

METHOD = "adain"
G_PATH = f"pretrained/alpha-{METHOD}-002600.pkl"
OUT_DIR = f"out"
VARIABLE_TYPE = ZVariable

if __name__ == "__main__":
    G = open_generator(G_PATH)
    
    images = []

    Interpolator(G).interpolate(
        VARIABLE_TYPE.sample_from(G, batch_size = 4), 
        VARIABLE_TYPE.sample_from(G, batch_size = 4)
    ).save(OUT_DIR + f"/fake_interpolation_{METHOD}.png")
