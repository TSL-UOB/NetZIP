import torch
print(torch.__version__)

import torchvision

import torchvision.datasets as datasets

cifar10_trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=None)
cifar10_testset  = datasets.CIFAR10(root='./data', train=False, download=True, transform=None)

print(len(cifar10_trainset))
print(len(cifar10_testset))