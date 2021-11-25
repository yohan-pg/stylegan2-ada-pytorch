from inversion import *

#todo add an inversion step and sample real images

OUT_DIR = f"out"
BATCH_SIZE = 3
METHOD_NAME, G_PATH = "adaconv", "pretrained/tmp.pkl"

if __name__ == "__main__":
    torch.manual_seed(0)

    with torch.no_grad():
        G = open_generator(G_PATH)

        low_variables = WVariable.sample_random_from(
            G, batch_size=BATCH_SIZE
        ).to_W_plus()
        high_variables = WVariable.sample_random_from(
            G, batch_size=BATCH_SIZE
        ).to_W_plus()

        low_images = low_variables.to_image()
        high_images = high_variables.to_image()

        mixes = []

        for i, high_variable in enumerate(
            high_variables.split_into_individual_variables()
        ):
            high_image = high_variable.to_image()
            high_variable.data = high_variable.data.repeat(BATCH_SIZE, 1, 1)
            mixes.append(torch.cat((
                high_image, 
                low_variables.mix(high_variable, 1).to_image()
            )))
            print(len(mixes[i]))

        save_image(
            make_grid(
                torch.cat(
                    (
                        torch.cat((torch.ones_like(low_images)[0:1], low_images)),
                        torch.cat(mixes)
                    )
                ),
                nrow=BATCH_SIZE + 1,
                padding=0
            ),
            OUT_DIR + f"/style_mixing_{METHOD_NAME}.png",
        )
