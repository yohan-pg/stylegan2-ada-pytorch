from inversion import *
from inversion.eval import *

# todo save as svg
# todo compare to sequential?
# todo refactor so that `run_eval` takes method-variable pairs

if __name__ == "__main__":
    evaluations = [
        # EvalReconstructionQuality,
        EvalInterpolationQuality,
    ]

    target_dataloader = RealDataloader(
        "datasets/afhq2_cat256_test.zip",
        batch_size=4,
        num_images=4,
    )

    if True:
        timestamp = run_eval(
            inverter_type=Inverter,
            label="",
            evaluations=evaluations,
            peform_dry_run=False,
            target_dataloader=target_dataloader,
            variable_types=[
                WPlusVariable,
                ZVariable
            ],
            num_steps=50,
            methods={
                "AdaConv": open_generator("pretrained/tmp.pkl"),
            },
            criterion=VGGCriterion(),
            create_optimizer=lambda params: torch.optim.Adam(params, lr=0.1),
            create_schedule=lambda optimizer: lr_scheduler.LambdaLR(
                optimizer, lambda epoch: min(1.0, epoch / 100.0)
            ),
        )
        create_artifacts(timestamp, target_dataloader, evaluations)
    else:
        create_artifacts(
            "2021-11-18_16:10:18", target_dataloader, evaluations
        )

# todo double check that it is deterministic for a given seed, & stable for another (low variance)
# todo delete all artifacts when creating new ones. maybe split measurements from artiact production in output folders?
# todo decide on including the reruns or not in the reconstruction eval
# todo what to do with second rereun for interpolation determinism?
# todo rename timestamp to something more appropriate (its a name)
# todo understand how to nest tqdm better
# todo avoid such extreme # of pairs in interpolation determinism. subset the second loader?
# todo should we really pass in target_dataloader into create_artifacts? in reality, the dataloader info we need should be saved into the pickle files
