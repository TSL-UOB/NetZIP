# NetZIP
This is the official repository for our paper: [*NetZIP: A Standardised Bench for DNNs Compression*](https://link-url-here.org)

NetZIP is an open-source benchmarking framework, based mainly on PyTorch, for evaluating the performance of model compression techniques on deep neural networks. It provides a standardized set of benchmarks and evaluation metrics for comparing different compression methods. The prominance of NetZIP is in the range of standardised evaluation metrics it provides to assess different aspects of performance affected by comrpession; these are broken into four main categories: accuracy, speed, size, and energy.

## List of currently supported frameworks:
Neural Networks (see [methods](methods)) :
- Object Classification
-- ResNet
-- VGG
- Object Detection
-- YOLOv5

Compression Methods (see [methods](methods)):
- Pruning
-- Global Unstructure Pruning (GUP)
-- Global Structure Pruning (GSP)

- Quantisation:
-- Post Training Quantisation (PTQ)
-- Quantisation Aware Training (QAT)

# Installation
1) Clone this repository: `git clone` 
2) Clone our docker image and setup container: `docker run -t --runtime=nvidia --shm-size 8G -d --name netzip -v ~/gits:/home -p 5000:80 abanoubg/netzip:latest`.

# Running Experiments.
Before starting with running experiments, setup datasets through the instructions listed
[here](readme/preparing_datasets.md).

We provide scripts to [train](scripts/object_classificaiton/train.py), [compress](scripts/object_classificaiton/compress.py) and [compare](scripts/object_classificaiton/compare.py) using different metrics reviewed in our paper.

For object classificaiton:
- First use [train.py](scripts/object_classificaiton/train.py) to train your chosen nerual network. 

- Second use [compress.py](scripts/object_classificaiton/compress.py) to compress the trained neural network using the different compression methods provided. Note: The script will need to be run independently to generate copmressed models using different chosen compression technqiues. 

- Third use [compare.py](scripts/object_classificaiton/compare.py) to compare between the different compressed models. You can choose which metrics you wish to use. 

For object detection follow the same approach for object classificaiton but uses a different set of [train](scripts/object_detection/train.py), [compress](scripts/object_detection/compare.py) and [compare](scripts/object_detection/compare.py) scripts. Note, since we currently only use YOLOv5 for object detection experiments the current implementation of compression methods for object detection is reliant on the builtin compression techniques provied by [Ultralytics](https://github.com/ultralytics/yolov5), which are limited to a version of tflite [Post Training Quantisation](models/yolov5/export.py) and [Global Unstructured Pruning](models/yolov5/utils/torch_utils.py). 

