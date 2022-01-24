from inversion.eval import *

run_eval(
    label="interpolation_enc_vs_hybrid",
    perform_dry_run=False,
    evaluations=[
        EvalReconstructionQuality,
        EvalInterpolationRealism,
        # EvalReconstructionRealism,
    ],
    target_dataloader=RealDataloader(
        "datasets/afhq2_cat256.zip",  #!!! _test
        batch_size=4,
        max_images=8,
        fid_data_path="datasets/afhq2_cat256",
        seed=0,
    ),
    num_steps=0,
    experiments={
        "AdaConv": (
            open_encoder(
                "encoder-training-runs/encoder_0.0/2022-01-05_12:03:21/encoder-snapshot-000150.pkl"
            ),
            add_hard_encoder_constraint(WVariable, 0.0, 0.00),
        ),
    },
    create_optimizer=lambda params: torch.optim.Adam(params, lr=0.02),
)

## bugs
#! dry run overwrites my real stuff now?

## artifact generation
# todo create the eval_again script

## code quality
# todo label the phases more clearly (right now there are a bunch of tqdm bars, not clear)
# todo clear memory between runs? with the encoder, memory usage gets quite high
# todo split regen script into a separate script
# todo review the need for table_stat

## evaluation determinism
# todo does not appear to be deterministic for a given seed
# todo verify that it is stable for 2 seeds (low variance)
# todo verify consistency with the eval_encoder script

## interpolation realism
# todo cap the max number of images -> we need to cap at, say, 50k. Is this realistic with batching or not?

## folder structure
# todo uniformize realism folder structure (recon & interpolation)
# todo make sure we don't need to split images into subfolders to avoid lag

## reconstruction quality
# todo add a plot for for regularization penalty, or the L2 distance to mean
