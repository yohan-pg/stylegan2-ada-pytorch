from inversion.eval import *

if __name__ == "__main__":
    regenerate_artifacts("evaluation-runs/2022-01-07_10:45:45", [
        EvalReconstructionQuality,
        # EvalReconstructionRealism,
        EvalInterpolationRealism,
    ])
