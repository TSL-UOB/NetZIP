# Post Training Quntisation (PTQ) method. Implemented based on Pytorch documentation.

import torch
import torch.nn as nn
import copy
from methods.common import QuantizedNN

def calibrate_model(model, loader, device=torch.device("cpu:0")):

    model.to(device)
    model.eval()

    for inputs, labels in loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        _ = model(inputs)


def PTQ(model, train_loader):
    # Move the model to CPU since static quantization does not support CUDA currently.
    cpu_device = torch.device("cpu:0")
    model.to(cpu_device)

    # Make a copy of the model.
    fused_model = copy.deepcopy(model)

    # The model has to be switched to evaluation mode for quantisation.
    fused_model.eval()

    
    # Fusing layers imporves speed and accuracy of Resnet quantized model performance. https://pytorch.org/blog/quantization-in-practice/
    # If model can be fused fuse it.Code below is set and tested for Resnet16.
    try: 
        fused_model = torch.quantization.fuse_modules(fused_model, [["conv1", "bn1", "relu"]], inplace=True)
        for module_name, module in fused_model.named_children():
            if "layer" in module_name:
                for basic_block_name, basic_block in module.named_children():
                    torch.quantization.fuse_modules(basic_block, [["conv1", "bn1", "relu1"], ["conv2", "bn2"]], inplace=True)
                    for sub_block_name, sub_block in basic_block.named_children():
                        if sub_block_name == "downsample":
                            torch.quantization.fuse_modules(sub_block, [["0", "1"]], inplace=True)
    except:
        print("Model did not fuse, so continuing without fusing.")


    compressed_model = QuantizedNN(model_fp32=fused_model)
    quantization_config = torch.quantization.get_default_qconfig("fbgemm")
    compressed_model.qconfig = quantization_config
    torch.quantization.prepare(compressed_model, inplace=True)
    
    # Use training data for calibration.
    calibrate_model(model=compressed_model, loader=train_loader, device=cpu_device)
    compressed_model = torch.quantization.convert(compressed_model, inplace=True)

    compressed_model.train()

    return compressed_model