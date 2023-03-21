# NetZIP
This is the official repository for our paper: [*NetZIP: A Standardised Bench for DNNs Compression*](https://link-url-here.org)

NetZIP is an open-source bench marking framework for evaluating the performance of model compression techniques on deep neural networks. It provides a standardized set of benchmarks and evaluation metrics for comparing different compression methods. The prominance of NetZIP is in the range of standardised evaluation metrics it provides to assess different aspects of performance affected by comrpession; these are broken into four main categories: accuracy, speed, size, and energy.


# Installation
1) Clone this repository: `git clone` 
2) Clone our docker image and setup container: `docker run -t --runtime=nvidia --shm-size 8G -d --name netzip -v ~/gits:/home -p 5000:80 abanoubg/netzip:latest`.

# Running Experiments.
Before starting with running experiments, setup datasets through the instructions listed
[here](readme/preparing_datasets.md).

We provide scripts for running experiments on the following VGG, Resnet and YOLOv5; using different these types of compression ----
