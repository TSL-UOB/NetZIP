import torch
print(torch.__version__)

import torchvision

import torchvision.datasets as datasets

cifar100_trainset = datasets.CIFAR100(root='./data', train=True, download=True, transform=None)
cifar100_testset  = datasets.CIFAR100(root='./data', train=False, download=True, transform=None)

print(len(cifar100_trainset))
print(len(cifar100_testset))