import random 
import torch
import torch.nn as nn
import numpy as np
import copy
class QuantizedNN(nn.Module):
    def __init__(self, model_fp32):
        super(QuantizedNN, self).__init__()
        # QuantStub converts tensors from floating point to quantized. This is only used for inputs.
        self.quant = torch.quantization.QuantStub()
        # DeQuantStub converts tensors from quantized to floating point. This is only used for outputs.
        self.dequant = torch.quantization.DeQuantStub()
        # FP32 model
        self.model_fp32 = model_fp32

    def forward(self, x):
        x = self.quant(x)
        x = self.model_fp32(x)
        x = self.dequant(x)
        return x

def calibrate_model(model, loader, device=torch.device("cpu:0")):

    model.to(device)
    model.eval()

    for inputs, labels in loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        _ = model(inputs)

def compress_model(model, compression_technique, device, train_loader):
    if compression_technique == "PTQ":

        # Move the model to CPU since static quantization does not support CUDA currently.
        cpu_device = torch.device("cpu:0")
        model.to(cpu_device)


        print("I am in PTQ")
        # Make a copy of the model for layer fusion
        fused_model = copy.deepcopy(model)

        model.eval()
        # The model has to be switched to evaluation mode before any layer fusion.
        # Otherwise the quantization will not work correctly.
        fused_model.eval()

        # Fuse the model in place rather manually.
        fused_model = torch.quantization.fuse_modules(fused_model, [["conv1", "bn1", "relu"]], inplace=True)
        for module_name, module in fused_model.named_children():
            if "layer" in module_name:
                for basic_block_name, basic_block in module.named_children():
                    torch.quantization.fuse_modules(basic_block, [["conv1", "bn1", "relu1"], ["conv2", "bn2"]], inplace=True)
                    for sub_block_name, sub_block in basic_block.named_children():
                        if sub_block_name == "downsample":
                            torch.quantization.fuse_modules(sub_block, [["0", "1"]], inplace=True)


        compressed_model = QuantizedNN(model_fp32=model)
        quantization_config = torch.quantization.get_default_qconfig("fbgemm")
        compressed_model.qconfig = quantization_config
        torch.quantization.prepare(compressed_model, inplace=True)
        # Use training data for calibration.
        calibrate_model(model=compressed_model, loader=train_loader)#, device=device)
        compressed_model = torch.quantization.convert(compressed_model, inplace=True)

        model.train()
        compressed_model.train()
        
        pass
    
    elif compression_technique == "QAT":
        pass

    else:
        print("ERROR: An unknown compression method was selected.")
    return compressed_model


def PTQ():
    pass