from inversion import *
from inversion.eval import *


RESUME_FROM_TIMESTAMP = None


if __name__ == "__main__":
    evaluations = [
        EvalReconstructionRealism,
        EvalReconstructionQuality,
        EvalInterpolationRealism,
    ]

    target_dataloader = RealDataloader(
        "datasets/afhq2_cat256_test.zip",
        batch_size=4,
        num_images=8,
        fid_data_path="datasets/afhq2_cat256",
        seed=0
    )

    if RESUME_FROM_TIMESTAMP is not None:
        create_artifacts(RESUME_FROM_TIMESTAMP, target_dataloader, evaluations)
    else:
        timestamp = run_eval(
            label="",
            evaluations=evaluations,
            peform_dry_run=False,
            target_dataloader=target_dataloader,
            variable_types=[
                add_hard_encoder_constraint(WPlusVariable, 0.0, 0.0)
            ],
            num_steps=0,
            methods={
                "AdaConv": open_encoder(
                    "encoder-training-runs/encoder_0.1/encoder-snapshot-000100.pkl"
                ),
            },
            criterion=VGGCriterion(),
            create_optimizer=lambda params: torch.optim.Adam(params, lr=0.02),
        )
        create_artifacts(timestamp, target_dataloader, evaluations)


#! dry run in broken
# todo add reconstruction quality tqdm total
# todo fix interpolation realism tqdm
# todo understand how to cache eval methods -> we want to pull creating the FID metric outside of the first pass
# todo consider splitting interpolation quality into subfolders to avoid lagging the FS
# todo refactor so that `run_eval` takes method-variable pairs [list out exactly which combination of parameter I want to use]
# todo does not appear to be deterministic for a given seed
# todo verify that it is stable for 2 seeds (low variance)
# todo delete all artifacts when creating new ones. maybe split measurements from artiact production in output folders?
# todo what to do with second rereun for interpolation determinism?
# todo understand how to nest tqdm better
# todo avoid such extreme no. of pairs in interpolation determinism. subset the second loader?
# todo should we really pass in target_dataloader into create_artifacts? in reality, the dataloader info we need should be saved into the pickle files
