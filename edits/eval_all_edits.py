from edits import *


if __name__ == "__main__":
    E = open_encoder("encoder-training-runs/encoder_wplus_0.1/encoder-snapshot-000100.pkl")

    run_edit_on_examples(Inpaint(), E)
    run_edit_on_examples(Upsample(), E)
    run_edit_on_examples(Colorize(), E)
    run_edit_on_examples(Denoise(), E)
    run_edit_on_examples(Deartifact(), E)


# for edit in []:
#     target_dataloader = RealDataloader(
#         "datasets/afhq2_cat256_test.zip",
#         batch_size=4,
#         num_images=32,
#         fid_data_path="datasets/afhq2_cat256",
#     )

# Colorize(self.E, self.target),
# Denoise(self.E, self.target),
# Deartifact(self.E, self.target),
# Upsample(self.E, self.target),
# Inpaint(self.E, self.target),