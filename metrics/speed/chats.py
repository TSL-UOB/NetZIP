# import numpy as np
# import thop
import torch
# # from ..thop_profile import profile
# from ..profile import profile
from .ops import ops_counter
import torch.nn.quantized as nnq  

def chats(model, device, test_loader, count_quantised_operations=True):
    """
    Computes Total Number of Operations 
    Args:
        model        : takes the loaded model.
        device       : device to load the model.
        test_loader  : dataset loader

    Returns chats (Compression and Hardware Agnostic Theoretical Speed)

    """
    
    # Get total number of operations
    total_ops = ops_counter(model, device, test_loader, count_quantised_operations=True)  
    
    # Get bit width used in the model
    for param in model.parameters():
        element_size = param.element_size()
        break

    for name, mod in model.named_modules():
        if isinstance(mod, nnq.Conv2d):                              
            weight, bias = mod._weight_bias()                        
            element_size = weight.element_size()
            break

    bit_width = element_size * 8


    # Calculate chats
    chats = total_ops * bit_width

    return chats

