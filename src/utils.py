import os
import torch
import numpy as np


def seed_everything(seed=42):
    """This is a function to seed everything to help reproducibility of the algorithm.

    Args:
        seed (int, optional): integer value to seed . Defaults to 42.
    """
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True