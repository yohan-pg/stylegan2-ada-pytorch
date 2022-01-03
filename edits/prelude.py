import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from inversion import *


#! add a distance to ground truth image

#! todo remove pasting function
#! godo include the paste in the image output
# todo refactor initialize method to another method which gives an example corruption
# todo make sure the scaling factor is required
# todo avoid passing a var by variable_type


@dataclass(eq=False)
class Edit(nn.Module):
    encoding_weight: ClassVar[float] = 0.0
    truncation_weight: ClassVar[float] = 0.0
    soft_constraint: bool = True
    learning_rate: float = 0.1
    numerical_stability_loss_scale: float = 1000.0

    def __new__(cls, *args, **kwargs):
        self = super().__new__(cls)
        nn.Module.__init__(self)
        return self

    def initialize(self, image_shape: torch.Size) -> None:
        pass

    @abstractmethod
    def f(self, pred: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def f_ground_truth(self, target: torch.Tensor) -> torch.Tensor:
        return self.f(target)  # * Different for certain tasks, e.g. denoising

    def paste(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = pred.detach() 
        return pred - self.f(pred) + target

    def present(self, x):
        return self.f(x)  # * Different for certain tasks, e.g. denoising

    def penalty(
        self, var: Variable, pred: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        return torch.tensor(0.0)

    @property
    def name(self) -> str:
        return self.__class__.__name__.lower()

    def log(self, ground_truth):
        pass  # todo

    def solve_from_ground_truth(self, E, ground_truth: torch.Tensor, **kwargs):
        self.initialize(ground_truth.shape)

        with torch.no_grad():
            target = self.f_ground_truth(ground_truth)

        for (inversion, snapshot_time) in self.solve(
            E, target, **kwargs, initialize=False
        ):
            if snapshot_time:
                self.log(ground_truth)

                save_image(
                    torch.cat(
                        (
                            ground_truth,
                            inversion.final_pred,
                            (ground_truth - self.present(inversion.final_pred)).abs(),
                        )
                    ),
                    f"out/{self.name}/ground_truth.png",
                    nrow=len(ground_truth),
                )

    def solve(
        self,
        E,
        target: torch.Tensor,
        num_steps: int = 1000,
        snapshot_frequency: int = 50,
        initialize: bool = True,
    ):
        if initialize:
            self.initialize(target.shape)

        fresh_dir(f"out/{self.name}")

        self.E = E 
        self.criterion = VGGCriterion().transform(self.f)

        inverter = Inverter(
            E,
            num_steps,
            variable_type=(
                add_soft_encoder_constraint
                if self.soft_constraint
                else add_hard_encoder_constraint
            )(
                E.variable_type,
                self.encoding_weight,
                self.truncation_weight,
                lambda pred: self.paste(pred, target),
                encoder_init=False,
                gamma=1.0
            ),
            # todo extract to shared method with refinement
            create_optimizer=lambda params: torch.optim.Adam(
                params, lr=self.learning_rate
            ),
            create_schedule=lambda optimizer: lr_scheduler.LambdaLR(
                optimizer, lambda t: min(1.0, t / 100.0)
            ), 
            criterion=self.criterion,
            snapshot_frequency=snapshot_frequency,
            penalty=lambda *args, **kwargs: self.penalty(*args, **kwargs),
            extra_params=self.parameters(),
            randomize=False,  #!!!
            parallel=False,  #!!!
        )
        for i, (inversion, snapshot_time) in enumerate(
            tqdm.tqdm(
                inverter.all_inversion_steps(target), total=len(inverter)
            )  # todo remove target from inversion params
        ):
            if i == 0:
                save_image(
                    inversion.variables[0].to_image(), f"out/{self.name}/init.png"
                )

            if snapshot_time:
                with torch.no_grad():
                    pasting = self.paste(inversion.final_pred, target)
                    fed = self.f(inversion.final_pred)

                with torch.no_grad():
                    inversion.final_pred
                    save_image(
                        torch.cat(
                            (
                                target,
                                inversion.final_pred,
                                (target - self.present(inversion.final_pred)).abs(),
                            )
                        ),
                        f"out/{self.name}/initial_fit.png",
                        nrow=len(target),
                    )
                    save_image(
                        torch.cat(
                            (
                                inversion.final_pred,
                                self.f(inversion.final_pred),
                                (fed - target).abs(),
                                target,
                                pasting,
                            )
                        ),
                        f"out/{self.name}/fing.png",
                        nrow=len(target),
                    )

                    # breakpoint()
                    # x = E(pasting)
                    # for i in range(30):
                    #     with torch.no_grad():
                    #         x = E(self.paste(x.to_image(), target))
                    # save_image(
                    #     torch.cat(
                    #         (
                    #             E(pasting).to_image(),
                    #             x.to_image(),
                    #         )
                    #     ),
                    #     f"out/{self.name}/npasting.png",
                    #     nrow=len(target),
                    # )

                    save_image(
                        torch.cat(
                            (
                                pasting,
                                E(pasting).to_image(),
                                (target - pasting).abs(),
                            )
                        ),
                        f"out/{self.name}/pasting.png",
                        nrow=len(target),
                    )

            yield (inversion, snapshot_time)

        inverter = Inverter(
            E,
            num_steps,
            variable_type=E.variable_type.from_variable(inversion.final_variable),
            create_optimizer=lambda params: torch.optim.Adam(
                params, lr=self.learning_rate
            ),
            create_schedule=lambda optimizer: lr_scheduler.LambdaLR(
                optimizer, lambda t: min(1.0, t / 100.0)
            ),
            criterion=1000 * VGGCriterion(),
            snapshot_frequency=50,
            penalty=self.penalty,
            extra_params=self.parameters(),
        )

        # with torch.no_grad():
        #     refinement_target = self.paste(inversion.final_pred, target)

        # for i, (refinement_inversion, snapshot_time) in enumerate(
        #     tqdm.tqdm(
        #         inverter.all_inversion_steps(refinement_target), total=len(inverter)
        #     )
        # ):
        #     if snapshot_time:
        #         with torch.no_grad():
        #             refinement_inversion.final_pred
        #             save_image(
        #                 torch.cat(
        #                     (
        #                         refinement_target,
        #                         refinement_inversion.final_pred,
        #                         (
        #                             target
        #                             - self.present(refinement_inversion.final_pred)
        #                         ).abs(),
        #                     )
        #                 ),
        #                 f"out/{self.name}/refinement.png",
        #                 nrow=len(target),
        #             )

        #     yield (refinement_inversion, snapshot_time)

        # save_image(
        #     torch.cat(
        #         (
        #             target,
        #             inversion.final_pred,
        #             refinement_inversion.final_pred,
        #         )
        #     ),
        #     f"out/{self.name}/before_and_after_refinement.png",
        #     nrow=len(target),
        # )

        # save_image(
        #     torch.cat(
        #         (
        #             target,
        #             refinement_inversion.final_pred,
        #             (target - self.present(refinement_inversion.final_pred)).abs(),
        #         )
        #     ),
        #     f"out/{self.name}/final_result.png",
        #     nrow=len(target),
        # )


def run_edit_on_examples(edit, E=None, **kwargs):
    if E is None:
        E = open_encoder(
            "encoder-training-runs/encoder_0.1/encoder-snapshot-000100.pkl"
        )

    target = open_target(
        E,
        "datasets/afhq2/test/cat/flickr_cat_000176.png",
        "datasets/afhq2/test/cat/flickr_cat_000236.png",
        "datasets/afhq2/test/cat/pixabay_cat_000117.png",
        "datasets/afhq2_cat256_test/00000/img00000162.png",
        "datasets/afhq2_cat256_test/00000/img00000195.png",
    )

    edit.solve_from_ground_truth(E, target, **kwargs)

    return edit
