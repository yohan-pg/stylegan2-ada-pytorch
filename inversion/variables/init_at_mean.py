from .w_variable import *
from .z_variable import *


# class WVariableInitAtMean(WVariable):
#     @classmethod
#     def sample_from(cls, G: nn.Module, batch_size: int = 1):
#         return cls(
#             G,
#             nn.Parameter(
#                 G.mapping.w_avg.reshape(1, 1, G.w_dim).repeat(
#                     batch_size, G.num_required_vectors(), 1
#                 )
#             ),
#         )


class ZVariableInitAtMean(ZVariable):
    # * Note that normalize_2nd_moment does preserve zeros, 
    # * it just projects to the typical set right after the first step. 
    # * So initializing z at the mean is compatible with normalized mappers in a weird way.
    @classmethod
    def sample_from(cls, G: nn.Module, batch_size: int = 1):
        return cls(
            G,
            nn.Parameter(ZVariable.sample_from(G, batch_size=batch_size).data * 0.0),
        )


class ZVariableInitAtMeanWithNoise(ZVariableInitAtMean):
    noise_level = 1.0 

    def to_styles(self):
        return self.G[0].mapping(self.data + torch.randn_like(self.data) * self.noise_level, None)


class ZVariableInitAtMeanWithMutatingNoise(ZVariableInitAtMean):
    noise_level = 1.0 

    def to_styles(self):
        if self.training:
            with torch.no_grad():
                self.data += torch.randn_like(self.data) * self.data.norm(dim=-1, keepdim=True) / math.sqrt(self.data.shape[-1]) * self.noise_level
        return self.G[0].mapping(self.data, None)



class ZVariableInitAtMeanMatchStat(ZVariableInitAtMean):
    noise_level = 1.0 

    def to_styles(self):
        data = self.data - self.data.mean(dim=2, keepdim=True)
        data = data / (data.std(dim=2) + 1e-10)
        return self.G[0].mapping(data, None)