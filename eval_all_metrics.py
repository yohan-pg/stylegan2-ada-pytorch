from inversion import *
from inversion.eval import *


RESUME_FROM_TIMESTAMP = None


if __name__ == "__main__":
    evaluations = [
        EvalReconstructionQuality,
        # EvalReconstructionRealism,
        # EvalInterpolationRealism,
    ]

    target_dataloader = RealDataloader(
        "datasets/afhq2_cat256_test.zip",
        batch_size=4,
        max_images=100,
        fid_data_path="datasets/afhq2_cat256",
        seed=0,
    )

    if RESUME_FROM_TIMESTAMP is not None:
        create_artifacts(RESUME_FROM_TIMESTAMP, target_dataloader, evaluations)
    else:
        E1 = open_encoder(
            "encoder-training-runs/encoder_0.1/encoder-snapshot-000100.pkl"
        )
        timestamp = run_eval(
            label="",
            evaluations=evaluations,
            perform_dry_run=True,
            target_dataloader=target_dataloader,
            num_steps=0,
            experiments={
                "AdaConv/W+": (
                    E1,
                    add_hard_encoder_constraint(WPlusVariable, 0.0, 0.0),
                ),
                "AdaIN/W": (
                    E1,
                    add_hard_encoder_constraint(WVariable, 0.0, 0.0),
                ),
                "AdaIN/W+": (
                    E1,
                    add_hard_encoder_constraint(WPlusVariable, 0.0, 0.0),
                ),
            },
            criterion=VGGCriterion(),
            create_optimizer=lambda params: torch.optim.Adam(params, lr=0.02),
        )
        create_artifacts(timestamp, target_dataloader, evaluations)


## code quality
# todo change prefix to creating a subfolder

## interpolation realism
# todo fix interpolation realism tqdm
# todo avoid such extreme no. of pairs in interpolation realism. subset the second loader?
#!!! interpolation generates WAY too many images, I don't understand

## artifact generation
# todo delete all artifacts when creating new ones. maybe split measurements from artiact production in output folders?
# todo should we really pass in target_dataloader into create_artifacts? in reality, the dataloader info we need should be saved into the pickle files
# todo understand how to cache eval methods -> we want to pull creating the FID metric outside of the first pass

## determinism
# todo does not appear to be deterministic for a given seed
# todo verify that it is stable for 2 seeds (low variance)

## folder structure
# todo uniformize realism folder structure (recon & interpolation)
# todo make sure we don't need to split images into subfolders to avoid lag
