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

    # for test_images, test_labels in test_loader:  
    #     sample_image_size = test_images[0].size()
    #     input_size = (1,sample_image_size[0], sample_image_size[1], sample_image_size[2])
    #     break
    
    # model.to(device)
    # model.eval()

    # x = torch.rand(size=input_size).to(device)
    # # macs, params = profile(model, inputs=(x,), verbose=False)

    # macs = profile(model, input_size, count_quantised_operations)
    # macs = macs.item()
    # print("macs = ", macs)
    # input("entert to continue")

    # if qunatised_flag:
    #     for name, mod in model.named_modules():
    #         if isinstance(mod, nnq.Conv2d):                              
    #             weight, bias = mod._weight_bias()                        
    #             # print(name, 'weight', weight, 'bias', bias) 
    #             parameters_count += weight.numel()
    #             parameters_count += bias.numel()

    # for n, m in model.named_modules():#model.named_children():
    #     try: 
    #         print("m.total_ops.item() = ", m.total_ops.item())
    #         input("entert to continue")  
    #     except:
    #         pass
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
