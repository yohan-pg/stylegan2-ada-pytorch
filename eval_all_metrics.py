from inversion.eval import *

# todo cross product for interpolation determinism
# todo cross product for editing consistency
# todo two pass: generate data, then plots and tables
# todo move root code functions into the folder as well
# todo double check that it is deterministic for a given seed, & stable for another (low variance)
# todo run fully to check "slow" trainings


if __name__ == "__main__":
    run_eval(
        label="hi",
        target_dataloader=RealDataloader(
            "datasets/afhq2_cat128_test.zip",
            batch_size=4,
            num_images=20,
        ),
        variable_types=[ZVariable, ZPlusVariable, WVariable, WPlusVariable],
        num_steps=1,
        methods={
            "AdaConv": open_generator("pretrained/adaconv-gamma-20-003800.pkl"),
            "AdaConvSlow": open_generator("pretrained/adaconv-slowdown-all.pkl"),
        },
        evaluations=[
            EvalReconstructionQuality,
            EvalInterpolationDeterminism,
            EvalImageEditingConsistency,
        ],
    )
