
import os
import random

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms

import time
import copy
import numpy as np

from resnet import resnet18

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
# == Set random seeds
# =====================================================
def set_random_seeds(random_seed=0):

    torch.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)

# =====================================================
# == Load and normalize TinyImageNet
# =====================================================
def prepare_dataloader(num_workers=8, train_batch_size=128, eval_batch_size=256):
    url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
    path = "./datasets/TinyImageNet/"

    if not os.path.isdir(path):
        os.makedirs(path)

    if len(os.listdir(path))==0:
        wget.download(url,out = path)
        print("Downloaded TinyImagnet zip file.")
    
    if len(os.listdir(path))<2:
        with ZipFile(path+"tiny-imagenet-200.zip", "r") as file:
            file.extractall(path)
        print("Extracted TinyImagnet files.")
    else:
        print("TinyImagnet files already exist.")

    train_dir  = os.path.join(path, 'tiny-imagenet-200/train')
    val_dir    = os.path.join(path, 'tiny-imagenet-200/val')

    # Help from this link: https://towardsdatascience.com/pytorch-ignite-classifying-tiny-imagenet-with-efficientnet-e5b1768e5e8f#:~:text=There%20are%20two%20ways%20to,from%20the%20official%20Stanford%20site

    # ==== Organize validation data folder in Tiny Imagenet to make it compatible with pytorch.
    # Create separate validation subfolders for the validation images based on
    # their labels indicated in the val_annotations txt file
    if os.path.exists(val_dir+"/images"):
        val_img_dir = os.path.join(val_dir, 'images')

        # Open and read val annotations text file
        fp = open(os.path.join(val_dir, 'val_annotations.txt'), 'r')
        data = fp.readlines()

        # Create dictionary to store img filename (word 0) and corresponding
        # label (word 1) for every line in the txt file (as key value pair)
        val_img_dict = {}
        for line in data:
            words = line.split('\t')
            val_img_dict[words[0]] = words[1]
        fp.close()

        # Create subfolders (if not present) for validation images based on label,
        # and move images into the respective folders
        for img, folder in val_img_dict.items():
            newpath_imgs       = (os.path.join(val_dir, folder,"images"))
    
            if not os.path.exists(newpath_imgs):
                os.makedirs(newpath_imgs)
            
            if os.path.exists(os.path.join(val_img_dir, img)):
                os.rename(os.path.join(val_img_dir, img), os.path.join(newpath_imgs, img))
            
        # Delete old images folder after finishing oraginsiign the images
        if os.path.exists(val_img_dir):
            os.rmdir(val_img_dir)
        print("Re-oragnised TinyImagenet val to Pytorch format.")

    else:
        print("TinyImagenet already oragnised to Pytorch format. Assumed because images folder does not exist.")


    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_set = torchvision.datasets.ImageFolder(root=train_dir, transform=train_transform)

    test_set = torchvision.datasets.ImageFolder(root=val_dir, transform=test_transform)

    train_sampler = torch.utils.data.RandomSampler(train_set)
    test_sampler = torch.utils.data.SequentialSampler(test_set)

    train_loader = torch.utils.data.DataLoader(
        dataset=train_set, batch_size=train_batch_size,
        sampler=train_sampler, num_workers=num_workers)

    test_loader = torch.utils.data.DataLoader(
        dataset=test_set, batch_size=eval_batch_size,
        sampler=test_sampler, num_workers=num_workers)

    return train_loader, test_loader

# =====================================================
# == Metric (Model Accuracy Evaluation)
# =====================================================
def evaluate_model(model, test_loader, device, criterion=None):

    model.eval()
    model.to(device)

    running_loss = 0
    running_corrects = 0

    for inputs, labels in test_loader:

        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)

        if criterion is not None:
            loss = criterion(outputs, labels).item()
        else:
            loss = 0

        # statistics
        running_loss += loss * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

    eval_loss = running_loss / len(test_loader.dataset)
    eval_accuracy = running_corrects / len(test_loader.dataset)

    return eval_loss, eval_accuracy

# =====================================================
# == Model Training
# =====================================================
def train_model(model, train_loader, test_loader, device, learning_rate, num_epochs):

    # The training configurations were not carefully selected.
    # learning_rate = 1e-2
    # num_epochs = 200

    criterion = nn.CrossEntropyLoss()

    model.to(device)

    # It seems that SGD optimizer is better than Adam optimizer for ResNet18 training on CIFAR100.
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-5)
    # optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)

    for epoch in range(num_epochs):

        # Training
        model.train()

        running_loss = 0
        running_corrects = 0

        for inputs, labels in train_loader:

            inputs = inputs.to(device)
            labels = labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        train_loss = running_loss / len(train_loader.dataset)
        train_accuracy = running_corrects / len(train_loader.dataset)

        # Evaluation
        model.eval()
        eval_loss, eval_accuracy = evaluate_model(model=model, test_loader=test_loader, device=device, criterion=criterion)

        print("Epoch: {:02d} Train Loss: {:.3f} Train Acc: {:.3f} Eval Loss: {:.3f} Eval Acc: {:.3f}".format(epoch, train_loss, train_accuracy, eval_loss, eval_accuracy))

    return model

# =====================================================
# == Model Callibration (Quantisation)
# =====================================================
def calibrate_model(model, loader, device=torch.device("cpu:0")):

    model.to(device)
    model.eval()

    for inputs, labels in loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        _ = model(inputs)

# =====================================================
# == Metric (Inference Latency) This link back's up this way of measuring the inference latency https://deci.ai/blog/measure-inference-time-deep-neural-networks/
# =====================================================
def measure_inference_latency(model,
                              device,
                              input_size=(1, 3, 32, 32),
                              num_samples=100,
                              num_warmups=10):

    model.to(device)
    model.eval()

    x = torch.rand(size=input_size).to(device)

    with torch.no_grad():
        for _ in range(num_warmups):
            _ = model(x)
    torch.cuda.synchronize()

    with torch.no_grad():
        start_time = time.time()
        for _ in range(num_samples):
            _ = model(x)
            torch.cuda.synchronize()
        end_time = time.time()
    elapsed_time = end_time - start_time
    elapsed_time_ave = elapsed_time / num_samples

    return elapsed_time_ave

# =====================================================
# == Model Save and Load
# =====================================================
def save_model(model, model_dir, model_filename):

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    model_filepath = os.path.join(model_dir, model_filename)
    torch.save(model.state_dict(), model_filepath)

def load_model(model, model_filepath, device):

    model.load_state_dict(torch.load(model_filepath, map_location=device))

    return model

def save_torchscript_model(model, model_dir, model_filename):

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    model_filepath = os.path.join(model_dir, model_filename)
    torch.jit.save(torch.jit.script(model), model_filepath)

def load_torchscript_model(model_filepath, device):

    model = torch.jit.load(model_filepath, map_location=device)

    return model

def create_model(num_classes=10):

    # The number of channels in ResNet18 is divisible by 8.
    # This is required for fast GEMM integer matrix multiplication.
    # model = torchvision.models.resnet18(pretrained=False)
    model = resnet18(num_classes=num_classes, pretrained=False)

    # We would use the pretrained ResNet18 as a feature extractor.
    # for param in model.parameters():
    #     param.requires_grad = False
    
    # Modify the last FC layer
    # num_features = model.fc.in_features
    # model.fc = nn.Linear(num_features, 10)

    return model


# =====================================================
# == Quantise ResNet18
# =====================================================

class QuantizedResNet18(nn.Module):
    def __init__(self, model_fp32):
        super(QuantizedResNet18, self).__init__()
        # QuantStub converts tensors from floating point to quantized.
        # This will only be used for inputs.
        self.quant = torch.quantization.QuantStub()
        # DeQuantStub converts tensors from quantized to floating point.
        # This will only be used for outputs.
        self.dequant = torch.quantization.DeQuantStub()
        # FP32 model
        self.model_fp32 = model_fp32

    def forward(self, x):
        # manually specify where tensors will be converted from floating
        # point to quantized in the quantized model
        x = self.quant(x)
        x = self.model_fp32(x)
        # manually specify where tensors will be converted from quantized
        # to floating point in the quantized model
        x = self.dequant(x)
        return x

def model_equivalence(model_1, model_2, device, rtol=1e-05, atol=1e-08, num_tests=100, input_size=(1,3,32,32)):

    model_1.to(device)
    model_2.to(device)

    for _ in range(num_tests):
        x = torch.rand(size=input_size).to(device)
        y1 = model_1(x).detach().cpu().numpy()
        y2 = model_2(x).detach().cpu().numpy()
        if np.allclose(a=y1, b=y2, rtol=rtol, atol=atol, equal_nan=False) == False:
            print("Model equivalence test sample failed: ")
            print(y1)
            print(y2)
            return False

    return True


def main():

    random_seed = 0
    num_classes = 200
    cuda_device = torch.device("cuda:0")
    cpu_device = torch.device("cpu:0")

    model_dir = "models/trained_models/TinyImageNet"
    model_filename = "resnet18_tinyimagenet.pt"
    quantized_model_filename = "resnet18_QAT_quantized_tinyimagenet.pt"
    model_filepath = os.path.join(model_dir, model_filename)
    quantized_model_filepath = os.path.join(model_dir, quantized_model_filename)

    set_random_seeds(random_seed=random_seed)

    train_loader, test_loader = prepare_dataloader(num_workers=8, train_batch_size=128, eval_batch_size=256)
    
    # Create an untrained model.
    model = create_model(num_classes=num_classes)
    # Load a pretrained model.
    model = load_model(model=model, model_filepath=model_filepath, device=cuda_device)
    # Move the model to CPU since static quantization does not support CUDA currently.
    model.to(cpu_device)

    # Make a copy of the model for layer fusion
    fused_model = copy.deepcopy(model)

    model.train()
    # The model has to be switched to training mode before any layer fusion.
    # Otherwise the quantization aware training will not work correctly.
    fused_model.train()

    # Fuse the model in place rather manually.
    fused_model = torch.quantization.fuse_modules(fused_model, [["conv1", "bn1", "relu"]], inplace=True)
    for module_name, module in fused_model.named_children():
        if "layer" in module_name:
            for basic_block_name, basic_block in module.named_children():
                torch.quantization.fuse_modules(basic_block, [["conv1", "bn1", "relu1"], ["conv2", "bn2"]], inplace=True)
                for sub_block_name, sub_block in basic_block.named_children():
                    if sub_block_name == "downsample":
                        torch.quantization.fuse_modules(sub_block, [["0", "1"]], inplace=True)

    # Print FP32 model.
    print(model)
    # Print fused model.
    print(fused_model)

    # Model and fused model should be equivalent.
    model.eval()
    fused_model.eval()
    assert model_equivalence(model_1=model, model_2=fused_model, device=cpu_device, rtol=1e-03, atol=1e-06, num_tests=100, input_size=(1,3,32,32)), "Fused model is not equivalent to the original model!"

    # Prepare the model for quantization aware training. This inserts observers in
    # the model that will observe activation tensors during calibration.
    quantized_model = QuantizedResNet18(model_fp32=fused_model)
    # Using un-fused model will fail.
    # Because there is no quantized layer implementation for a single batch normalization layer.
    # quantized_model = QuantizedResNet18(model_fp32=model)
    # Select quantization schemes from 
    # https://pytorch.org/docs/stable/quantization-support.html
    quantization_config = torch.quantization.get_default_qconfig("fbgemm")
    # Custom quantization configurations
    # quantization_config = torch.quantization.default_qconfig
    # quantization_config = torch.quantization.QConfig(activation=torch.quantization.MinMaxObserver.with_args(dtype=torch.quint8), weight=torch.quantization.MinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_tensor_symmetric))

    quantized_model.qconfig = quantization_config
    
    # Print quantization configurations
    print(quantized_model.qconfig)

    # https://pytorch.org/docs/stable/_modules/torch/quantization/quantize.html#prepare_qat
    torch.quantization.prepare_qat(quantized_model, inplace=True)

    # # Use training data for calibration.
    print("Training QAT Model...")
    quantized_model.train()
    train_model(model=quantized_model, train_loader=train_loader, test_loader=test_loader, device=cuda_device, learning_rate=1e-3, num_epochs=10)
    quantized_model.to(cpu_device)

    # Using high-level static quantization wrapper
    # The above steps, including torch.quantization.prepare, calibrate_model, and torch.quantization.convert, are also equivalent to
    # quantized_model = torch.quantization.quantize_qat(model=quantized_model, run_fn=train_model, run_args=[train_loader, test_loader, cuda_device], mapping=None, inplace=False)

    quantized_model = torch.quantization.convert(quantized_model, inplace=True)

    quantized_model.eval()

    # Print quantized model.
    print(quantized_model)

    # Save quantized model.
    save_torchscript_model(model=quantized_model, model_dir=model_dir, model_filename=quantized_model_filename)

    # Load quantized model.
    quantized_jit_model = load_torchscript_model(model_filepath=quantized_model_filepath, device=cpu_device)

    _, fp32_eval_accuracy = evaluate_model(model=model, test_loader=test_loader, device=cpu_device, criterion=None)
    _, int8_eval_accuracy = evaluate_model(model=quantized_jit_model, test_loader=test_loader, device=cpu_device, criterion=None)

    # Skip this assertion since the values might deviate a lot.
    # assert model_equivalence(model_1=model, model_2=quantized_jit_model, device=cpu_device, rtol=1e-01, atol=1e-02, num_tests=100, input_size=(1,3,32,32)), "Quantized model deviates from the original model too much!"

    print("FP32 evaluation accuracy: {:.3f}".format(fp32_eval_accuracy))
    print("INT8 evaluation accuracy: {:.3f}".format(int8_eval_accuracy))

    fp32_cpu_inference_latency = measure_inference_latency(model=model, device=cpu_device, input_size=(1,3,32,32), num_samples=100)
    int8_cpu_inference_latency = measure_inference_latency(model=quantized_model, device=cpu_device, input_size=(1,3,32,32), num_samples=100)
    int8_jit_cpu_inference_latency = measure_inference_latency(model=quantized_jit_model, device=cpu_device, input_size=(1,3,32,32), num_samples=100)
    fp32_gpu_inference_latency = measure_inference_latency(model=model, device=cuda_device, input_size=(1,3,32,32), num_samples=100)
    
    print("FP32 CPU Inference Latency: {:.2f} ms / sample".format(fp32_cpu_inference_latency * 1000))
    print("FP32 CUDA Inference Latency: {:.2f} ms / sample".format(fp32_gpu_inference_latency * 1000))
    print("INT8 CPU Inference Latency: {:.2f} ms / sample".format(int8_cpu_inference_latency * 1000))
    print("INT8 JIT CPU Inference Latency: {:.2f} ms / sample".format(int8_jit_cpu_inference_latency * 1000))

if __name__ == "__main__":

    main()