import os 
import wget
from zipfile import ZipFile

import torch
import torchvision
from torchvision import datasets, transforms


def pytorch_dataloader(dataset_name="",num_workers=8, train_batch_size=128, eval_batch_size=256):

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    # Check which dataset to load.
    if dataset_name == "CIFAR10":
        train_set = torchvision.datasets.CIFAR10(root="../../datasets/CIFAR10", train=True, download=True, transform=train_transform) 
        test_set = torchvision.datasets.CIFAR10(root="../../datasets/CIFAR10", train=False, download=True, transform=test_transform)
    
    elif dataset_name =="CIFAR100": 
        train_set = torchvision.datasets.CIFAR10(root="../../datasets/CIFAR10", train=True, download=True, transform=train_transform) 
        test_set = torchvision.datasets.CIFAR10(root="../../datasets/CIFAR10", train=False, download=True, transform=test_transform)

    elif dataset_name == "TinyImageNet":
        train_set = torchvision.datasets.ImageFolder(root="../../datasets/TinyImageNet/tiny-imagenet-200/train", transform=train_transform)
        test_set = torchvision.datasets.ImageFolder(root="../../datasets/TinyImageNet/tiny-imagenet-200/val", transform=test_transform)

    else:
        print("ERROR: dataset name is not integrated into NETZIP yet.")


    train_sampler = torch.utils.data.RandomSampler(train_set)
    test_sampler = torch.utils.data.SequentialSampler(test_set)

    train_loader = torch.utils.data.DataLoader(
        dataset=train_set, batch_size=train_batch_size,
        sampler=train_sampler, num_workers=num_workers)

    test_loader = torch.utils.data.DataLoader(
        dataset=test_set, batch_size=eval_batch_size,
        sampler=test_sampler, num_workers=num_workers)

    return train_loader, test_loader