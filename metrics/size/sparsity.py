import torch

def get_module_sparsity(module, weight=True, bias=False):
    num_zeros = 0
    num_elements = 0

    for buffer_name, buffer in module.named_buffers():
        if "weight_mask" in buffer_name and weight == True:
            num_zeros += torch.sum(buffer == 0).item()
            num_elements += buffer.nelement()
        if "bias_mask" in buffer_name and bias == True:
            num_zeros += torch.sum(buffer == 0).item()
            num_elements += buffer.nelement()

    for param_name, param in module.named_parameters():
        if "weight" in param_name and weight == True:
            num_zeros += torch.sum(param == 0).item()
            num_elements += param.nelement()
        if "bias" in param_name and bias == True:
            num_zeros += torch.sum(param == 0).item()
            num_elements += param.nelement()
    
    if num_elements == 0:
        sparsity = float('inf') 
    else:
        sparsity = num_zeros / num_elements

    return num_zeros, num_elements, sparsity



def get_global_sparsity(model, weight=True, bias=False):

    num_zeros = 0
    num_elements = 0

    for module_name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):# or isinstance(module, torch.nn.Linear):
            module_num_zeros, module_num_elements, _ = get_module_sparsity(module, weight, bias)
            num_zeros += module_num_zeros
            num_elements += module_num_elements

    sparsity = num_zeros / num_elements

    return num_zeros, num_elements, sparsity