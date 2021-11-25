from inversion import *


@dataclass(eq=False)
class Edit:
    from_point: torch.Tensor
    to_point: torch.Tensor

    def apply(self, point: torch.Tensor, alpha: float):
        raise NotImplementedError


@dataclass(eq=False)
class LerpEdit(Edit):
    def apply(self, point: torch.Tensor, alpha: float):
        diff = self.to_point - self.from_point
        return alpha * diff + point


@dataclass(eq=False)
class SlerpEdit(Edit):
    def apply(self, point: torch.Tensor, alpha: float):
        plane_axis = torch.cross(self.from_point, self.to_point)
        component_off_plane = point @ plane_axis.t()

        projection_on_plane = None
        rotation_matrix = torch.tensor(
            [
                [
                    torch.cos(alpha),
                    torch.sin(-alpha),
                    torch.sin(alpha),
                    torch.cos(alpha),
                ]
            ]
        )
        rotated_vector_on_plane = rotation_matrix @ projection_on_plane
        return (
            component_off_plane
            + rotated_vector_on_plane[:, 0] @ self.from_point
            + rotated_vector_on_plane[:, 1] @ self.to_point
        )
