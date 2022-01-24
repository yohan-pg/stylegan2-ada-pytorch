import sys 
import os 
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from inversion import *

CRITERION = VGGCriterion()
NUM_OPTIM_STEPS = 1
LR = 0.02