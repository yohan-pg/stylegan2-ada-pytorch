from inversion import *


def edit(
    name: str, # * Edit name used for saving artifacts
    G, # * Generator G
    E, # * Encoder E which must be trained using G
    target, # * Target batch y
    
    lift = lambda x: x, # * Lifting function f
    paste = lambda x: x, # * Pasting function g 
    present = lambda x: x, # * Presentation function which formats lifted images into something presentable (used for visualization only)
    
    encoding_weight: float = 0.15, # * Decay towards the encoder distribution psi 
    truncation_weight: float = 0.05, # * Truncation decay phi
    penalty = None # * Optional regularization penalty
):
    os.makedirs(f"out/{name}", exist_ok=True)

    inverter = Inverter(
        G,
        1000,
        variable_type=add_encoder_constraint(E.variable_type, E, encoding_weight, truncation_weight, paste),
        create_optimizer=lambda params: torch.optim.Adam(params, lr=0.1),
        create_schedule=lambda optimizer: lr_scheduler.LambdaLR(
            optimizer, lambda epoch: min(1.0, epoch / 100.0) 
        ),
        penalty=penalty,
        criterion=100 * VGGCriterion().transform(lift), # todo only lift target once
        snapshot_frequency=50,
        seed=7,
    )

    lifted_target = present(lift(target))

    for i, (inversion, did_snapshot) in enumerate(
        tqdm.tqdm(inverter.all_inversion_steps(target), total=len(inverter)) # todo remove target from inversion params
    ):
        if i == 0:
            save_image(inversion.variables[0].to_image(), f"out/{name}/init.png")

        if did_snapshot:
            with torch.no_grad():
                inversion.final_pred
                save_image(
                    torch.cat(
                        (
                            lifted_target,
                            inversion.final_pred,
                            (lifted_target - present(lift(inversion.final_pred))).abs(),
                        )
                    ),
                    f"out/{name}/result.png",
                    nrow=len(target),
                )





    