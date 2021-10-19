from inversion import *

# todo slerp
# todo clean up image/target image range
# todo double check that VGG loss inputs are the right size and range and everything



OUT_DIR = f"out"
VARIABLE_TYPE = ZVariable
BATCH_SIZE = 12

if __name__ == "__main__":
    for METHOD_NAME, G_PATH in {
        "adain": "pretrained/adain-their-params-003800.pkl",
        "adaconv": "pretrained/adaconv-gamma-20-003800.pkl"
    }.items():
        with torch.no_grad():
            G = open_generator(G_PATH)

            images = []
            
            ZVariable.interpolate = WVariable.interpolate #!!!

            Interpolation.from_variables(
                ZVariableInitAtMean.sample_from(G, batch_size=BATCH_SIZE),
                # VARIABLE_TYPE.sample_random_from(G, batch_size=BATCH_SIZE),
                VARIABLE_TYPE.sample_random_from(G, batch_size=BATCH_SIZE),
                gain=3
            ).save(OUT_DIR + f"/fake_interpolation_{METHOD_NAME}.png")
