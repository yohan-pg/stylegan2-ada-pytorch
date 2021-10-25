from inversion.eval import *

# todo plots are missing stuff!
# todo recon quality plot is off (weird values - maybe wasn't transposed)
# todo make histogram start at 0 on the left *
# todo merge all images into a single image with the best metrics in bold
# todo two pass: generate data, then plots and tables
# todo run fully to check "slow" trainings
# todo double check that it is deterministic for a given seed, & stable for another (low variance)

# todo decide on including the reruns or not in the reconstruction eval 
# todo what to do with second rereun for interpolation determinism?
# todo rename timestamp to something more appropriate (its a name)
# todo understand how to nest tqdm better
# todo avoid such extreme # of pairs in interpolation determinism. subset the second loader?


if __name__ == "__main__":
    evaluations = [
        EvalReconstructionQuality,
        EvalInterpolationDeterminism,
        EvalImageEditingConsistency,
    ]

    target_dataloader = RealDataloader(
        "datasets/afhq2_cat128_test.zip",
        batch_size=4,
        num_images=1,
    )
    
    timestamp = run_eval(
        label="eval_slow",
        target_dataloader=target_dataloader,
        variable_types=[ZVariable, ZPlusVariable, WVariable, WPlusVariable],
        num_steps=1,
        methods={
            "AdaConv": open_generator("pretrained/adaconv-gamma-20-003800.pkl"),
            "AdaConvSlow": open_generator("pretrained/adaconv-slowdown-all.pkl"),
        },
        evaluations=evaluations,
    )

    create_artifacts(timestamp, target_dataloader, evaluations)
