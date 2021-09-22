from inversion import *

# todo slerp
# todo clean up image/target image range
# todo double check that VGG loss inputs are the right size and range and everything

from torchvision.utils import save_image

# todo adaconv

METHOD = "adaconv"
G_PATH = f"pretrained/alpha-{METHOD}-002600.pkl"
OUT_DIR = f"out"
RESOLUTION = 32
LEARNING_RATE = 0.01

def downsample(x):
    return F.interpolate(x, size=(RESOLUTION, RESOLUTION), mode="bilinear", align_corners=False)

if __name__ == "__main__":
    G = open_generator(G_PATH)
    D = open_discriminator(G_PATH)

    variable = WVariableInitAtMean.sample_from(G)
    optimizer = torch.optim.SGD(variable.parameters(), lr=LEARNING_RATE * 100)
    # optimizer = torch.optim.LBFGS(variable.parameters(), lr=1.0, line_search_fn="strong_wolfe")
    
    criterion = VGGCriterion()

    target = open_target(G, "datasets/samples/cats/00000/img00000014.png")
    target_low_res = downsample(target)

    for i in range(1_000_000):
        def eval():
            optimizer.zero_grad()
            pred = variable.to_image()
            loss = criterion(downsample(pred), target_low_res) 
            loss.backward()
            return loss
    
        loss = optimizer.step(eval)
        
        if i % 100 == 0:
            pred = variable.to_image()
            print(loss.item())
            save_image(torch.cat((pred, target)), "out/pulse_result.png")
            save_image(torch.cat((downsample(pred), target_low_res)), "out/pulse_optim.png")