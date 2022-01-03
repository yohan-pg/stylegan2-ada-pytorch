from .prelude import *

from .variables import *


class Inversion:
    def __init__(
        self,
        target: nn.Module,
        variables: List[Variable],
        losses: List[float],
        preds: List[torch.Tensor],
        eval=False
    ):
        self.variables = variables
        self.target = target
        self.losses = losses
        self.preds = preds

        if eval:
            for var in self.variables:
                var.eval()

        self.final_variable = self.variables[-1]
        self.final_pred = self.preds[-1]

    @staticmethod
    def save_losses_plot(inversions: List["Inversion"], out_path: str) -> None:
        plt.figure()
        plt.title("Reconstruction loss per step")

        for name, inversion in reversed(inversions.items()):
            for loss_list in torch.stack(inversion.losses, dim=1):
                plt.plot(loss_list.detach().cpu(), label=name)
        plt.legend()

        plt.ylim(0, 0.5)
        plt.xlabel("Optimization steps")
        plt.ylabel("Reconstruction loss")
        plt.savefig(out_path)

    def move_to_cuda(self):
        self.final_pred = self.final_pred.cuda()
        self.final_variable = self.final_variable.cuda()

    def move_to_cpu(self):
        self.final_pred = self.final_pred.cpu()
        self.final_variable = self.final_variable.cpu()

    def save_optim_trace(self, out_path: str) -> None:
        save_image(torch.cat(self.preds), out_path)

    def purge(self):
        self = copy.copy(self)

        del self.preds
        del self.variables
        self.final_pred = self.final_pred.cpu()
        self.final_variable = self.final_variable.cpu()

        return self

    def snapshot(self, out_path: str):
        save_image(
            torch.cat(
                (self.target, self.final_pred, (self.target - self.final_pred).abs())
            ),
            out_path,
            nrow=(3 if len(self.final_pred) == 1 else len(self.final_pred)),
        )

    @staticmethod
    def save_to_video(outpath: str, inversions: List["Inversion"]):
        def convert(x):
            return (
                (x.clamp(min=0, max=1.0) * 255)
                .to(torch.uint8)
                .transpose(1, 3)
                .transpose(1, 2)
                .cpu()
            )

        joinded_videos = torch.cat(
            [torch.cat(inversion.preds) for inversion in inversions], dim=3
        )
        write_video(outpath, convert(joinded_videos), fps=25)
