from inversion import *

# todo slerp
# todo clean up image/target image range
# todo double check that VGG loss inputs are the right size and range and everything



OUT_DIR = f"out"
BATCH_SIZE = 1

TARGET_PATH = "./datasets/afhq2/train/cat/flickr_cat_000006.png"
METHOD_NAME, G_PATH = "adaconv", "pretrained/baseline-final.pkl"
CRITERION = VGGCriterion

if __name__ == "__main__":
    G = open_generator(G_PATH)
    breakpoint()
    target = open_target(G, TARGET_PATH)

    var = WVariable.sample_from(G, batch_size=BATCH_SIZE)
    criterion = CRITERION()
    init = var.copy()
    
    if False:
        optim = torch.optim.Adam(var.parameters(), lr=0.05)
    else:
        optim = torch.optim.SGD(var.parameters(), lr=500.0)
    
    for i in range(100):
        optim.zero_grad()
        criterion(var.to_image(), target).mean().backward()
        optim.step()

        Interpolation.from_variables(
            init, 
            var,
            # var.from_data(var.data - var.data.grad * DISTANCE),
            # var.from_data(var.data + var.data.grad * DISTANCE),
            gain=1.0,
            num_steps=13
        ).save(OUT_DIR + f"/step_{i}.png")

        init = var.copy()

        
