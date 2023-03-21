import random 
import torch
import numpy as np

def set_cuda(use_cuda=False):

    use_cuda = use_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    return device


def set_random_seeds(random_seed=0):

    torch.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)
