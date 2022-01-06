from inversion.eval import *

if __name__ == "__main__":
    regenerate_artifacts("", [
        EvalReconstructionQuality,
        # EvalReconstructionRealism,
        # EvalInterpolationRealism,
    ])
