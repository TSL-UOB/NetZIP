# Quantisation Aware Training (QAT) method. Implemented based on Pytorch documentation and https://github.com/1adrianb/binary-networks-pytorch.


import torch
import torch.nn as nn
import copy
from methods.common import QuantizedNN
from utils.model import train_model

# from bnn.ops import BasicInputBinarizer,XNORWeightBinarizer
# from bnn import BConfig, prepare_binary_model, Identity
# from bnn.models.resnet import resnet18
# from nni.algorithms.compression.pytorch.quantization import BNNQuantizer
# import torch.optim as optim

def BNN(model, train_loader, test_loader, device, learning_rate=1e-5 , num_epochs=10):
    
    print("I am in BNN,but BNN method not implemented yet. Waiting for Pytorch to implement it. For now use LARQ library.")

    # # Move the model to CPU.
    # cpu_device = torch.device("cpu:0")
    # model.to(cpu_device)

    # # Make a copy of the model.
    # to_compress_model = copy.deepcopy(model)

    # print(to_compress_model)

    # # Binarize
    # configure_list = [{
    #     'quant_types': ['weight'],
    #     'quant_bits': 1,
    #     'op_types': ['Conv2d', 'Linear'],
    #     # 'op_names': ['features.3', 'features.7', 'features.10', 'features.14', 'classifier.0', 'classifier.3']
    # }, {
    #     'quant_types': ['output'],
    #     'quant_bits': 1,
    #     'op_types': ['Hardtanh'],
    #     'op_names': ['features.6', 'features.9', 'features.13', 'features.16', 'features.20', 'classifier.2', 'classifier.5']
    # }]

    # optimizer = optim.SGD(to_compress_model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-5)

    # quantizer = BNNQuantizer(to_compress_model, configure_list, optimizer)
    # compressed_model = quantizer.compress()

    # # Train

    return compressed_model