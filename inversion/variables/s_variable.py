from .variable import *
from .w_variable import *
from .plus_variables import *

from training.networks import *


class Style(torch.Tensor):
    pass


def forward_if_not_style(self, x):
    if isinstance(x, Style):
        return x[:, : self.weight.shape[0]]
    else:
        return self._forward(x)


class SVariable(Variable):
    space_name = "S"
    
    @staticmethod
    def prepare_G(G: nn.Module):
        num_modules = 0

        for module in G.synthesis.modules():
            if module.__class__.__name__ == "FullyConnectedLayer":
                num_modules += 1
                if not hasattr(module, "_forward"):
                    module._forward = module.forward
                    module.forward = forward_if_not_style.__get__(module)

        assert num_modules > 0

    @classmethod
    def sample_from(cls, G: nn.Module, batch_size: int = 1):
        SVariable.prepare_G(G)
        affines = []
        for module in G.synthesis.modules():
            if module.__class__.__name__ == "FullyConnectedLayer":
                affines.append(module)
        assert len(affines) == G.num_ws - 1 #?
        data = WPlusVariable.sample_from(G, batch_size).data
        k = G.num_required_vectors()
        with torch.no_grad():
            #! ew
            for i, affine in enumerate(affines):
                datum = data[:, i * k: (i + 1) * k]
                style = affine(datum.reshape(-1, G.w_dim)).reshape(
                    datum.shape[0], datum.shape[1], affine.weight.shape[0]
                )
                datum[:, :, :affine.weight.shape[0]].copy_(style)

        
        return SVariable(
            G,
            data
        )  #!!! need to pass through the affines...

    @classmethod
    def sample_random_from(cls, G: nn.Module, batch_size: int = 1):
        raise NotImplementedError

    def to_W(self):
        return WPlusVariable(self.G[0], self.data)

    interpolate = WVariable.interpolate

    def to_styles(self):
        return Style(self.data.cpu()).cuda()
