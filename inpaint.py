from inversion import *

# METHOD = "adaconv"
# G_PATH = f"pretrained/adaconv-256-001200.pkl"
# G_PATH = "pretrained/adain-their-params-003800.pkl"
# G_PATH = "pretrained/adain-slowdown-halftrain.pkl"
G_PATH = f"pretrained/adaconv-slowdown-all.pkl"


def make_mask(x):
    x = x.clone()
    mask = torch.ones_like(x)
    _, _, H, W = x.shape
    mask[:, :, :, W // 4 : W // 4 * 3] = 0.0
    return mask


if __name__ == "__main__":
    G = open_generator(G_PATH)

    target = open_target(
        G,
        "./datasets/afhq2/train/cat/flickr_cat_000018.png"
        # "datasets/samples/cats/00000/img00000014.png"
        # "./datasets/afhq2/train/cat/flickr_cat_000007.png"
        # "datasets/afhq2/test/cat/pixabay_cat_000542.png"
        # "datasets/afhq2/test/cat/pixabay_cat_002694.png"
        # "datasets/afhq2/test/wild/flickr_wild_001251.png"
        # "./datasets/afhq2/train/dog/flickr_dog_000021.png"
        # "./datasets/afhq2/train/dog/flickr_dog_000022.png"
    )
    mask = make_mask(target)

    # todo do this in vgg space
    class GradientNormPenalty:
        def __init__(self, weight: float, vgg):
            self.weight = weight
            self.vgg = vgg

        def __call__(self, variable, styles, pred, target, loss):
            return (
                self.weight
                * torch.autograd.grad(
                    self.vgg(
                        pred.clone() * 255, resize_images=False, return_lpips=True
                    ).sum(),
                    pred.abs().sum(),
                    variable.data,
                    create_graph=True,
                )[0].norm(dim=(1, 2), p=1)
            )

    class PPLPenalty:
        def __init__(self, weight: float, vgg):
            self.weight = weight
            self.vgg = vgg

        def __call__(self, variable, styles, pred, target, loss):
            pred = criterion.vgg16(pred.clone() * 255, resize_images=False, return_lpips=True)
            return (
                self.weight
                * torch.autograd.grad(
                    pred,
                    variable.data,
                    torch.randn_like(pred) / math.sqrt(pred.shape[1]),
                    create_graph=True,
                )[0].norm(dim=(1, 2), p=1)
            )

    def power_iteration(
            matrix: torch.Tensor, num_iterations: int = 5, rayleigh=False, cache_id=None
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            assert matrix.shape[1] == matrix.shape[2]
            batch_size, dim, _ = matrix.shape
            query_vector = torch.randn(batch_size, dim, 1)

            for _ in range(num_iterations):
                query_vector = matrix @ query_vector
                norm = query_vector.norm(dim=1, keepdim=True)
                query_vector = query_vector / norm

            if rayleigh:
                def dot(a, b):
                    assert a.shape == b.shape
                    return (a * b).sum(dim=(1,2))
                norm = dot(query_vector, (matrix @ query_vector)) / dot(query_vector, query_vector)
            else:
                norm = norm.squeeze(2)

            return query_vector.squeeze(2), norm

    def largest_eigenvalue(matrix: torch.Tensor, **kwargs) -> torch.Tensor:
        return power_iteration(matrix, **kwargs)[1]
    

    class NetMetricMatrix:
        def __init__(self, net, point, shift=0):
            self.net = net
            self.point = point
            self.shape = (1, point.shape[1], point.shape[1])
            self.shift = shift

        def to_matrix(self, **kwargs):
            J = torch.autograd.functional.jacobian(self._f, self.point.squeeze(0), **kwargs)
            return (J @ J.t()).unsqueeze(0)

        def spectral_shift(self, amount):
            return NetMetricMatrix(self.net, self.point, self.shift + amount.squeeze())

        def __matmul__(self, dir):
            # *  This should be equivalent to a dot product against the metric tensor
            # *  (J @ J.t) @ X
            # *  == J @ (J.t @ X)
            # *  == J @ (X.t @ J).t
            y = torch.autograd.functional.vjp(self._f, self.point.flatten(), dir.flatten(), create_graph=True)[1]
            z = torch.autograd.functional.jvp(self._f, self.point.flatten(), y, create_graph=True)[1]
            return z.reshape(self.point.shape)

        def _f(self, x):
            return self.net(x).flatten()
            
    class SpectralPenalty:
        def __init__(self, weight):
            self.weight = weight 

        def __call__(self, variable, styles, pred, target, loss):
            def forward(z):
                return G(z.reshape(1, 512, 512), None)
            M = NetMetricMatrix(forward, variable.data) @ torch.randn_like(target)
            quit()

            largest_eigenvalue(M)

    criterion = MaskedVGGCriterion(mask)


    ZVariable.default_lr /= 3
    N = 1
    inverter = Inverter(
        G,
        400 * N,
        # make_ZVariableConstrainToTypicalSetAllVecsWithNoise(
        #     noise=ZVariable.default_lr * 50, truncation=1.0
        # )
        # make_WPlusVariableWithNoise(ZVariable.default_lr * 50)
        # if False
        # else 
        ZVariableConstrainToTypicalSetAllVecs,
        criterion=criterion,
        learning_rate=ZVariable.default_lr,
        snapshot_frequency=50,
        step_every_n=N,
        seed=11,
        penalties=[
            SpectralPenalty(1.0)
        ]
    )

    for inversion in tqdm.tqdm(inverter.all_inversion_steps(target * mask)):
        save_image(
            torch.cat(
                (
                    target,
                    mask,
                    target * mask,
                    inversion.final_pred,
                    # inversion.final_variable.disable_noise().to_image(),
                    (inversion.final_pred * (1.0 - mask)) + target * mask,
                )
            ),
            "out/inpainting_result.png",
        )
