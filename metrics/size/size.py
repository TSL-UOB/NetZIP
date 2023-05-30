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

def gpu_utilisation():
    """
    Returns the GPU memory usage in gega bytes(GB).
    """
    mem_usage_bytes = torch.cuda.max_memory_allocated()
    return mem_usage_bytes / 1024 ** 3

def cpu_utilisation(model, device,test_loader, initial_machine_status):
    """
    Returns the CPU utilisation in gega bytes(GB).
    """
    
    # == Measure before utilsiation before inference.
    utilisation_before = initial_machine_status['CPU_utilisation']
    
    # == Measure utilistion during inference.
    num_samples=10000
    num_warmups=1000

    utilisation_during = 0

    for test_images, test_labels in test_loader:  
        sample_image_size = test_images[0].size()
        input_size = (1,sample_image_size[0], sample_image_size[1], sample_image_size[2])
        break
    
    model.to(device)
    model.eval()

    x = torch.rand(size=input_size).to(device)

    with torch.no_grad():
        for _ in range(num_warmups):
            _ = model(x)
    torch.cuda.synchronize()

    with torch.no_grad():
        psutil.cpu_percent(interval=0, percpu=False) # Running command which should be the reference meaurement for next time the command is called.
        for _ in range(num_samples):
            _ = model(x)
            torch.cuda.synchronize()
            
            # Measure utilsiation
            utilisation_during += psutil.cpu_percent(interval=0, percpu=False)

    utilisation_during = utilisation_during/num_samples

    # == Subtract to get utilisation due to inference
    cpu_utilisation_by_model = utilisation_during - utilisation_before
    print("cpu_utilisation_by_model = ",cpu_utilisation_by_model)
    return cpu_utilisation_by_model


def ram_usage(model, device,test_loader,initial_machine_status):
    """
    Compute the system memory (RAM) usage for the current device (GB).
    Returns:
        usage (float): used memory (GB).
        total (float): total memory (GB).
    """

    # == Measure before utilsiation before inference.
    usage_before = initial_machine_status['ram_usage']
    
    # == Measure utilistion during inference.
    num_samples=10000
    num_warmups=1000

    usage_during = 0

    for test_images, test_labels in test_loader:  
        sample_image_size = test_images[0].size()
        input_size = (1,sample_image_size[0], sample_image_size[1], sample_image_size[2])
        break
    
    model.to(device)
    model.eval()

    x = torch.rand(size=input_size).to(device)

    with torch.no_grad():
        for _ in range(num_warmups):
            _ = model(x)
    torch.cuda.synchronize()

    with torch.no_grad():
        # start_time = time.time()
        for _ in range(num_samples):
            _ = model(x)
            torch.cuda.synchronize()
            
            # Measure utilsiation
            vram  = psutil.virtual_memory()
            usage_during = max(usage_during,((vram.total - vram.available) / 1024 ** 3))

    # == Subtract to get utilisation due to inference
    ram_usage_by_model = usage_during - usage_before

    return ram_usage_by_model

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