from inversion import *


class Initialization:
    def initialize(self, variable: Variable) -> None:
        pass

class InitializeToRandomPoint(Initialization):
    seed: Optional[int]

    def initialize(self, variable: Variable) -> None:
        pass
        # variable.data.copy_(
        #     variable.__class__.sample_random_from(variable.G, )
        # )

class InitializeToSameRandomPoint(Initialization):
    seed: Optional[int]

    def initialize(self, variable: Variable) -> None:
        torch.manual_seed(self.seed) 

class InitializeToMean(Initialization):
    seed: Optional[int]
    pass


class InitializeToRandomNearestNeighbor(Initialization):
    seed: Optional[int]
    num_trials: int = 30

    pass

class InitializeToKDTreeNearestNeighbor(Initialization):
    seed: Optional[int]
    pass