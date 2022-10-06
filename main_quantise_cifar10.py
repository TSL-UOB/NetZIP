
import torch
import torchvision
import torchvision.transforms as transforms

import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim

import matplotlib.pyplot as plt
import numpy as np

# ======================================================================
# == Check GPU is connected
# ======================================================================

print("======================")
print("Check GPU is info")
print("======================")
print("How many GPUs are there? Answer:",torch.cuda.device_count())
print("The Current GPU:",torch.cuda.current_device())
print("The Name Of The Current GPU",torch.cuda.get_device_name(torch.cuda.current_device()))
# Is PyTorch using a GPU?
print("Is Pytorch using GPU? Answer:",torch.cuda.is_available())
print("======================")

# switch to False to use CPU
use_cuda = True

use_cuda = use_cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu");

# =====================================================
# == Load and normalize CIFAR10
# =====================================================

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

batch_size = 4

trainset = torchvision.datasets.CIFAR10(root='./datasets/CIFAR10', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./datasets/CIFAR10', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
    'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


# functions to show an image
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.savefig("temp.png")


# == Let us show some of the training images, for fun.

# # get some random training images
# dataiter = iter(trainloader)
# images, labels = dataiter.next()

# # show images
# imshow(torchvision.utils.make_grid(images))
# # print labels
# print(' '.join(f'{classes[labels[j]]:5s}' for j in range(batch_size)))



# =====================================================
# == Define a Convolutional Neural Network
# =====================================================

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 20, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(20, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# =====================================================
# == Load saved model
# =====================================================

PATH = './models/trained_models/CIFAR10/cifar_net.pth'

# Load saved model 
model = Net().to(device)
model.load_state_dict(torch.load(PATH))



# =====================================================
# == Quantise
# =====================================================
from collections import namedtuple
QTensor = namedtuple('QTensor', ['tensor', 'scale', 'zero_point'])


def calcScaleZeroPoint(min_val, max_val,num_bits=8):
    # Calc Scale and zero point of next 
    qmin = 0.
    qmax = 2.**num_bits - 1.

    scale = (max_val - min_val) / (qmax - qmin)

    initial_zero_point = qmin - min_val / scale

    zero_point_next = 0
    if initial_zero_point < qmin:
        zero_point = qmin
    elif initial_zero_point > qmax:
        zero_point = qmax
    else:
        zero_point = initial_zero_point

    zero_point = int(zero_point)

    return scale, zero_point

def quantize_tensor(x, num_bits=8, min_val=None, max_val=None):
    
    if not min_val and not max_val: 
        min_val, max_val = x.min(), x.max()

    qmin = 0.
    qmax = 2.**num_bits - 1.

    scale, zero_point = calcScaleZeroPoint(min_val, max_val, num_bits)

    q_x = zero_point + x / scale
    q_x.clamp_(qmin, qmax).round_()
    q_x = q_x.round().byte()

    return QTensor(tensor=q_x, scale=scale, zero_point=zero_point)


def dequantize_tensor(q_x):
    return q_x.scale * (q_x.tensor.float() - q_x.zero_point)



# === Modified forward pass for Net
  
def quantizeLayer(x, layer, stat, scale_x, zp_x):
    # for both conv and linear layers

    # cache old values
    W = layer.weight.data
    B = layer.bias.data

    # quantise weights, activations are already quantised
    w = quantize_tensor(layer.weight.data) 
    b = quantize_tensor(layer.bias.data)

    layer.weight.data = w.tensor.float()
    layer.bias.data = b.tensor.float()


    # This is Quantisation Artihmetic
    scale_w = w.scale
    zp_w = w.zero_point

    scale_b = b.scale
    zp_b = b.zero_point

    scale_next, zero_point_next = calcScaleZeroPoint(min_val=stat['min'], max_val=stat['max'])

    # Perparing input by shifting
    X = x.float() - zp_x
    layer.weight.data = scale_x * scale_w*(layer.weight.data - zp_w)
    layer.bias.data = scale_b*(layer.bias.data + zp_b)

    # All int

    x = (layer(X)/ scale_next) + zero_point_next 

    # Perform relu too
    x = F.relu(x)

    # Reset
    layer.weight.data = W
    layer.bias.data = B

    return x, scale_next, zero_point_next

# Get Min and max of x tensor, and stores it
def updateStats(x, stats, key):
    max_val, _ = torch.max(x, dim=1)
    min_val, _ = torch.min(x, dim=1)

    if key not in stats:
        stats[key] = {"max": max_val.sum(), "min": min_val.sum(), "total": 1}
    else:
        stats[key]['max'] += max_val.sum().item()
        stats[key]['min'] += min_val.sum().item()
        stats[key]['total'] += 1

    return stats

# Reworked Forward Pass to access activation Stats through updateStats function
def gatherActivationStats(model, x, stats):

    stats = updateStats(x.clone().view(x.shape[0], -1), stats, 'conv1')
    x = model.pool(F.relu(model.conv1(x)))

    stats = updateStats(x.clone().view(x.shape[0], -1), stats, 'conv2')
    x = model.pool(F.relu(model.conv2(x)))

    x = torch.flatten(x, 1) # flatten all dimensions except batch

    stats = updateStats(x, stats, 'fc1')
    x = F.relu(model.fc1(x))

    stats = updateStats(x, stats, 'fc2')
    x = F.relu(model.fc2(x))

    stats = updateStats(x, stats, 'fc3')
    x = model.fc3(x)

    return stats

# Entry function to get stats of all functions.
def gatherStats(model, test_loader):
    device = 'cuda'
    
    model.eval()
    test_loss = 0
    correct = 0
    stats = {}
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            stats = gatherActivationStats(model, data, stats)
    
    final_stats = {}
    for key, value in stats.items():
      final_stats[key] = { "max" : value["max"] / value["total"], "min" : value["min"] / value["total"] }
    return final_stats


def quantForward(model, x, stats):

    # pool = nn.MaxPool2d(2, 2)

    x = quantize_tensor(x, min_val=stats['conv1']['min'], max_val=stats['conv1']['max'])
    
    x, scale_next, zero_point_next = quantizeLayer(x.tensor, model.conv1, stats['conv2'], x.scale, x.zero_point)

    x = model.pool(x)
    
    x, scale_next, zero_point_next = quantizeLayer(x, model.conv2, stats['fc1'], scale_next, zero_point_next)

    x = model.pool(x)

    x = torch.flatten(x, 1) # flatten all dimensions except batch

    x, scale_next, zero_point_next = quantizeLayer(x, model.fc1, stats['fc2'], scale_next, zero_point_next)
    
    x, scale_next, zero_point_next = quantizeLayer(x, model.fc2, stats['fc3'], scale_next, zero_point_next)

    # Back to dequant for final layer
    x = dequantize_tensor(QTensor(tensor=x, scale=scale_next, zero_point=zero_point_next))

    x = model.fc3(x)
    return x


# =====================================================
# == Testing Function for Quantisation
# =====================================================
def testQuant(model, test_loader, quant=False, stats=None):
    device = 'cuda'
    
    model.eval()
    # test_loss = 0
    total = 0
    correct = 0
    with torch.no_grad():
        # for data, target in test_loader:
        #     data, target = data.to(device), target.to(device)
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            if quant:
              output = quantForward(model, images, stats)
            else:
              output = model(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(output.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            # test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            # pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            # correct += pred.eq(target.view_as(pred)).sum().item()

    # test_loss /= len(test_loader.dataset)

    print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')
    # print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    #     test_loss, correct, len(test_loader.dataset),
    #     100. * correct / len(test_loader.dataset)))

# correct = 0
# total = 0
# # since we're not training, we don't need to calculate the gradients for our outputs
# with torch.no_grad():
#     for data in testloader:
#         images, labels = data[0].to(device), data[1].to(device)
#         # calculate outputs by running images through the network
#         outputs = net(images)
#         # the class with the highest energy is what we choose as prediction
#         _, predicted = torch.max(outputs.data, 1)
#         total += labels.size(0)
#         correct += (predicted == labels).sum().item()

# print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')



# =====================================================
# == Get Accuracy of Non Quantised Model
# =====================================================
import copy
q_model = copy.deepcopy(model)
print("== Non Quantised model accuracy ==")
testQuant(q_model, testloader, quant=False)

# =====================================================
# == Gather Stats of Activations
# =====================================================
stats = gatherStats(q_model, testloader)
print("== Printing stats ==")
print(stats)

# =====================================================
# == Test Quantised Inference Of Model
# =====================================================
print("== Quantised model accuracy ==")
testQuant(q_model, testloader, quant=True, stats=stats)

# tutorial followed: https://karanbirchahal.medium.com/how-to-quantise-an-mnist-network-to-8-bits-in-pytorch-no-retraining-required-from-scratch-39f634ac8459
# https://colab.research.google.com/drive/1oDfcLRz2AIgsclkXJHj-5wMvbylr4Nxz#scrollTo=f6duGNF_uoZB



