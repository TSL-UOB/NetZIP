import os 
import wget
from zipfile import ZipFile

import torch
import torchvision
from torchvision import datasets, transforms

from torch.utils.data import Dataset
from PIL import Image
import json


def pytorch_dataloader(dataset_name="",num_workers=16, train_batch_size=32, eval_batch_size=32):

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

    elif dataset_name =="ImageNet":
        dataset_root = "../../datasets/ImageNet/imagenet-object-localization-challenge/ILSVRC/Data/CLS-LOC"
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)

        transform = transforms.Compose(
                    [
                        transforms.Resize(256),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        transforms.Normalize(mean, std),
                    ]
                )

        train_set  = torchvision.datasets.ImageFolder(root=dataset_root+"/train", transform=transform)#ImageNetKaggle(dataset_root, "train", transform)
        test_set   = torchvision.datasets.ImageFolder(root=dataset_root+"/val", transform=transform)#ImageNetKaggle(dataset_root, "val", transform)

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