import os 
import wget
from zipfile import ZipFile

import torch
import torchvision
from torchvision import datasets, transforms

from torch.utils.data import Dataset
from PIL import Image
import json

class ImageNetKaggle(Dataset):
    def __init__(self, root, split, transform=None):
        self.samples = []
        self.targets = []
        self.transform = transform
        self.syn_to_class = {}
        with open(os.path.join(root, "imagenet_class_index.json"), "rb") as f:
                    json_file = json.load(f)
                    for class_id, v in json_file.items():
                        self.syn_to_class[v[0]] = int(class_id)
        with open(os.path.join(root, "ILSVRC2012_val_labels.json"), "rb") as f:
                    self.val_to_syn = json.load(f)
        samples_dir = os.path.join(root, "Data/CLS-LOC", split)
        for entry in os.listdir(samples_dir):
            if split == "train":
                syn_id = entry
                target = self.syn_to_class[syn_id]
                syn_folder = os.path.join(samples_dir, syn_id)
                for sample in os.listdir(syn_folder):
                    sample_path = os.path.join(syn_folder, sample)
                    self.samples.append(sample_path)
                    self.targets.append(target)
            elif split == "val":
                syn_id = self.val_to_syn[entry]
                target = self.syn_to_class[syn_id]
                sample_path = os.path.join(samples_dir, entry)
                self.samples.append(sample_path)
                self.targets.append(target)
    def __len__(self):
            return len(self.samples)
    def __getitem__(self, idx):
            x = Image.open(self.samples[idx]).convert("RGB")
            if self.transform:
                x = self.transform(x)
            return x, self.targets[idx]


def pytorch_dataloader(dataset_name="",num_workers=8, train_batch_size=32, eval_batch_size=32):

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
        dataset_root = "../../datasets/ImageNet/imagenet-object-localization-challenge/ILSVRC"
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

        train_set  = ImageNetKaggle(dataset_root, "train", transform)
        test_set   = ImageNetKaggle(dataset_root, "val", transform)

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