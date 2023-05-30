import numpy as np
import thop
import torch
# from ..thop_profile import profile
from ..profile import profile

def ops_counter(model, device, test_loader, count_quantised_operations=True):
    """
    Computes Total Number of Operations 
    Args:
        model        : takes the loaded model.
        device       : device to load the model.
        test_loader  : dataset loader

    Returns total_ops 

    """
    for test_images, test_labels in test_loader:  
        sample_image_size = test_images[0].size()
        input_size = (1,sample_image_size[0], sample_image_size[1], sample_image_size[2])
        break
    
    model.to(device)
    model.eval()

    x = torch.rand(size=input_size).to(device)

    total_ops = profile(model, input_size, count_quantised_operations)
    total_ops = total_ops.item()

    return total_ops


def macs(model, device, test_loader, count_quantised_operations=True):
    """
    Computes Multiply–Accumulate Operations (MACs)
    One MAC contains one multiplication and one addition.  
    So one MAC can though as having has two operations.

    Args:
        model        : takes the loaded model.
        device       : device to load the model.
        test_loader  : dataset loader

    Returns MACs 

    """
    total_ops = ops_counter(model, device, test_loader, count_quantised_operations=True)  
    if total_ops != 0:
        macs = total_ops / 2  # One MAC can though as having has two operations (one multiplication and one addition).
    else:
        macs = 0
    return macs


def flops(model, device, test_loader):
    """
    Computes Floating Points Operations (FLOPs)

    Args:
        model        : takes the loaded model.
        device       : device to load the model.
        test_loader  : dataset loader

    Returns FLOPs 
    """
    total_ops = ops_counter(model, device, test_loader, count_quantised_operations=False)  
    
    if total_ops != 0:
        flops = total_ops  # / 1E9 
    else:
        flops = 0

    return flops
