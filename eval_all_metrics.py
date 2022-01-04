from inversion import *
from inversion.eval import *

if __name__ == "__main__":
    evaluations = [
        EvalReconstructionQuality,
        # EvalInterpolationQuality,
        # EvalImageEditingConsistency,
    ]

    target_dataloader = RealDataloader(
        "datasets/afhq2_cat256_test.zip",
        batch_size=4,
        num_images=8,
        fid_data_path="datasets/afhq2_cat256",
    )

    if True:
        timestamp = run_eval(
            inverter_type=Inverter,
            label="",
            evaluations=evaluations,
            peform_dry_run=False,
            target_dataloader=target_dataloader,
            variable_types=[
                add_soft_encoder_constraint(WPlusVariable, 0.0, 0.002),
            ],
            num_steps=1000,
            methods={
                "AdaConv": open_encoder(
                    "encoder-training-runs/encoder_0.1/encoder-snapshot-000100.pkl"
                ),
                "AdaIn": open_encoder(
                    "encoder-training-runs/encoder_0.1_baseline/encoder-snapshot-000100.pkl"
                ),
            },
            criterion=VGGCriterion(),
            create_optimizer=lambda params: torch.optim.Adam(params, lr=0.02),
        )
        create_artifacts(timestamp, target_dataloader, evaluations)
    else:
        create_artifacts("2021-11-25_20:01:36", target_dataloader, evaluations)


#! dry run in broken
#!!!!!!! interpolation quality is not deterministic?
# todo make it work with encoders
# todo understand how to cache eval methods
# todo fix progress for interpolation determinism
# todo consider splitting interpolation quality into folders
# todo refactor so that `run_eval` takes method-variable pairs [list out exactly which combination of parameter I want to use]
# todo add a flag for sequential optimization
# todo double check that it is deterministic for a given seed, & stable for another (low variance)
# todo delete all artifacts when creating new ones. maybe split measurements from artiact production in output folders?
# todo what to do with second rereun for interpolation determinism?
# todo understand how to nest tqdm better
# todo avoid such extreme # of pairs in interpolation determinism. subset the second loader?
# todo should we really pass in target_dataloader into create_artifacts? in reality, the dataloader info we need should be saved into the pickle files
# todo multigpu?
