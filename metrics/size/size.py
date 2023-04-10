import os
import psutil
import torch
import numpy as np

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

def parameters_count(model):
    """
    Calculates the number of parameters in a model.
    Args:
        model: model.

    Return: 
        Number of parameters.

    """
    # table = PrettyTable(["Modules", "Parameters"])
    # total_params = 0
    # for name, parameter in model.named_parameters():
    #     if not parameter.requires_grad: continue
    #     params = parameter.numel()
    #     table.add_row([name, params])
    #     total_params+=params
    # print(table)
    # print(f"Total Trainable Params: {total_params}")

    # Parameters count for quantised models is zero. Could this be due the parameters not being floats, therefore not detected?
    
    return np.sum([p.numel() for p in model.parameters()]).item()