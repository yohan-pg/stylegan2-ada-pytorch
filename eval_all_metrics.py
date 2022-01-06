from inversion.eval import *

evaluations = [
    EvalReconstructionQuality,
    EvalReconstructionRealism,
    EvalInterpolationRealism,
]

run_eval(
    label=None,
    evaluations=evaluations,
    perform_dry_run=True,
    target_dataloader=RealDataloader(
        "datasets/afhq2_cat256_test.zip",
        batch_size=4,
        max_images=8,
        fid_data_path="datasets/afhq2_cat256",
        seed=0,
    ),
    num_steps=0,
    experiments={
        "AdaConv/W+": (
            open_encoder(
                "encoder-training-runs/encoder_0.1/encoder-snapshot-000100.pkl"
            ),
            add_hard_encoder_constraint(WPlusVariable, 0.0, 0.0),
        ),
        # "AdaIN/W": (
        #     open_encoder(
        #         "encoder-training-runs/encoder_0.1/encoder-snapshot-000100.pkl"
        #     ),
        #     add_hard_encoder_constraint(WVariable, 0.0, 0.0),
        # ),
    },
    criterion=VGGCriterion(),
    create_optimizer=lambda params: torch.optim.Adam(params, lr=0.02),
)


## artifact generation
# todo create a script that resumes from compute_metrics

## code quality
# todo label the phases more clearly (right now there are a bunch of tqdm bars, not clear)
# todo clear memory between runs? with the encoder, memory usage gets quite high
# todo split regen script into a separate script
# todo review the need for table_stat

## interpolation realism
# todo fix interpolation realism tqdm
# todo avoid such extreme no. of pairs in interpolation realism. subset the second loader?
#!!! interpolation generates WAY too many images, I don't understand

## determinism
# todo does not appear to be deterministic for a given seed
# todo verify that it is stable for 2 seeds (low variance)

## folder structure
# todo uniformize realism folder structure (recon & interpolation)
# todo make sure we don't need to split images into subfolders to avoid lag

## reconstruction quality
# todo add a plot for for regularization penalty, or the L2 distance to mean
