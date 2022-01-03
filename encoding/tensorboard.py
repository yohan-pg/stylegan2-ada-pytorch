
import warnings
warnings.filterwarnings("ignore", module="tensorboard")
from torch.utils.tensorboard import SummaryWriter
from tensorboard import program

def launch_tensorboard(out_dir):
    writer = SummaryWriter(out_dir)
    tb = program.TensorBoard()
    tb.configure(argv=[None, "--logdir", out_dir])
    tb.launch()
    return writer