from inversion import *

METHOD = "adaconv"
G_PATH = f"pretrained/alpha-{METHOD}-002600.pkl"
OUT_DIR = f"out"
RESOLUTION = 32
LEARNING_RATE = 0.1

def downsample(x):
    return F.interpolate(x, size=(RESOLUTION, RESOLUTION), mode="bilinear", align_corners=False)

if __name__ == "__main__":
    G = open_generator(G_PATH)
    D = open_discriminator(G_PATH)

    variable = WConvexCombinationVariable.sample_from(G)
    # optimizer = torch.optim.Adam(variable.parameters(), lr=LEARNING_RATE)
    optimizer = torch.optim.SGD(variable.parameters(), lr=LEARNING_RATE)
    
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