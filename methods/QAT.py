import torch
import torch.nn as nn
import copy
from methods.common import QuantizedNN
from utils.model import train_model


def QAT(model, train_loader, test_loader, device, learning_rate=1e-3 , num_epochs=2):
    # Move the model to CPU since static quantization does not support CUDA currently.
    cpu_device = torch.device("cpu:0")
    model.to(cpu_device)

    # Make a copy of the model.
    fused_model = copy.deepcopy(model)

    # The model has to be switched to evaluation mode for quantisation.
    fused_model.eval()

    # # Fusing layers  imporves speed and accuracy of quantized model performance. https://pytorch.org/blog/quantization-in-practice/
    # # Code below is set and tested for Resnet16. This does not generlise, so commented out.
    # fused_model = torch.quantization.fuse_modules(fused_model, [["conv1", "bn1", "relu"]], inplace=True)
    # for module_name, module in fused_model.named_children():
    #     if "layer" in module_name:
    #         for basic_block_name, basic_block in module.named_children():
    #             torch.quantization.fuse_modules(basic_block, [["conv1", "bn1", "relu1"], ["conv2", "bn2"]], inplace=True)
    #             for sub_block_name, sub_block in basic_block.named_children():
    #                 if sub_block_name == "downsample":
    #                     torch.quantization.fuse_modules(sub_block, [["0", "1"]], inplace=True)


    compressed_model = QuantizedNN(model_fp32=fused_model)
    quantization_config = torch.quantization.get_default_qconfig("fbgemm")
    compressed_model.qconfig = quantization_config
    torch.quantization.prepare(compressed_model, inplace=True)
    
    # Training quantised model.
    print("Training QAT ...")
    compressed_model.train()
    train_model(model=compressed_model, train_loader=train_loader, test_loader=test_loader, device=device, learning_rate=learning_rate, num_epochs=num_epochs)
    compressed_model.to(cpu_device)
   

    compressed_model = torch.quantization.convert(compressed_model, inplace=True)

    compressed_model.train()

    return compressed_model