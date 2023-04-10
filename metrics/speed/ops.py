import numpy as np
import thop
import torch

def macs(model,device,test_loader):
    """
    Computes Multiply–Accumulate Operation (MAC)
    Args:
        model        : takes the loaded model.
        device       : device to load the model.
        test_loader  : dataset loader

    Returns MACs 

    """
    for test_images, test_labels in test_loader:  
        sample_image_size = test_images[0].size()
        input_size = (1,sample_image_size[0], sample_image_size[1], sample_image_size[2])
        break
    
    model.to(device)
    model.eval()

    x = torch.rand(size=input_size).to(device)
    macs, params = thop.profile(model, inputs=(x,), verbose=False)
    return macs, params


def flops(model, device, test_loader):
    """
    Computes Floating Points Operations (FLOPS)

    One MAC contains one multiplication and one addition. 
    One multiplication or one addition can be seen as one FLOP. So one MAC has two FLOPS.

    Args:
        model        : takes the loaded model.
        device       : device to load the model.
        test_loader  : dataset loader

    Returns Gega FLOPS () GFLOPs

    """
    mac, params = macs(model,device,test_loader)  

    # if params are counted then there are floating points, otherwise model is quantised and there are no FLOPs.
    if params != 0:
        flops = mac / 1E9 * 2
    else:
        flops = 0

    return flops
