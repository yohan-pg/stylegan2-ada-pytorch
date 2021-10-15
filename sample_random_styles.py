from inversion import *

# todo slerp
# todo clean up image/target image range
# todo double check that VGG loss inputs are the right size and range and everything

from torchvision.utils import save_image

G_PATH = latest_snapshot("cfg_auto_large_res_no_div_slow_mapper_and_affine")
OUT_DIR = f"out"
VARIABLE_TYPE = ZVariable
BATCH_SIZE = 16

if __name__ == "__main__":
    G = open_generator(G_PATH)
    
    with torch.no_grad():
        images = VARIABLE_TYPE.sample_from(G, BATCH_SIZE).to_image()
    save_image(images, f"out/random_styles_{VARIABLE_TYPE.__name__}.png")
