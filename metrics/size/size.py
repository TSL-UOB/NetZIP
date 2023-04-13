import os
import psutil
import torch
import numpy as np
import torch.nn.quantized as nnq  

def model_size(file_path):
    """
    Computes model file size in mega bytes (MB).
    Args:
        Model file path
    Returns:
        model file size in mega bytes (MB)
    """

    file_stats = os.stat(file_path)
    return file_stats.st_size / (1024 * 1024)

def gpu_mem_usage():
    """
    Returns the GPU memory usage in gega bytes(GB).
    """
    mem_usage_bytes = torch.cuda.max_memory_allocated()
    return mem_usage_bytes / 1024 ** 3

def cpu_mem_usage():
    """
    Compute the system memory (RAM) usage for the current device (GB).
    Returns:
        usage (float): used memory (GB).
        total (float): total memory (GB).
    """
    vram = psutil.virtual_memory()
    usage = (vram.total - vram.available) / 1024 ** 3
    total = vram.total / 1024 ** 3

    return usage, total

# from prettytable import PrettyTable

def parameters_count(model, count_non_zero_only = True):
    """
    Calculates the number of parameters in a model.
    Args:
        model: model.

    Return: 
        Number of parameters.

    """
     
    parameters_count = 0 
    
    # === Count model parameters
    for p in model.parameters():
        if count_non_zero_only:
            parameters_count += torch.count_nonzero(p) # Only counts non zeros
        else:
            parameters_count += p.numel() # Counts zeros or non-zeros 
        
    # === Count number of parameters if model is quantised, as the above will output zero for quantised models.
    for name, mod in model.named_modules():
        if isinstance(mod, nnq.Conv2d):                              
            weight, bias = mod._weight_bias()                        
            # print(name, 'weight', weight, 'bias', bias) 
            parameters_count += weight.numel()
            parameters_count += bias.numel()


    # === If parmaeters count is in tensor conver to numpy integer
    try :
        parameters_count.item()
    except:
        pass

    try:
        parameters_count = np.int(parameters_count)
    except:
        pass


    # return np.sum([p.numel() for p in model.parameters()]).item()
    return parameters_count