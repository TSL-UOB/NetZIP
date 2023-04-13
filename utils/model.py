import os

import importlib

import torch
import torch.nn as nn
import torch.optim as optim

from metrics.accuracy.topAccuracy import top1Accuracy

import time

from methods.common import QuantizedNN
import copy

def create_model(model_dir, model_choice, model_variant, num_classes=10):
    model_module_path = model_dir+"/"+model_choice+".py"
    model_module      = importlib.util.spec_from_file_location("",model_module_path).loader.load_module()
    model_function    = getattr(model_module, model_variant)
    model             = model_function(num_classes=num_classes, pretrained=False)

    return model

def load_model(model, model_filepath, device):
    model.load_state_dict(torch.load(model_filepath, map_location=device))
    return model

def save_model(model, model_dir, model_filename):
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    model_filepath = os.path.join(model_dir, model_filename)
    torch.save(model.state_dict(), model_filepath)

def model_selection(model_selection_flag=0, model_dir="", model_choice="", model_variant="", saved_model_filepath="",num_classes=1, device=""):
    
    if model_selection_flag == 0:
        if model_choice == "yolo":
            model = torch.hub.load('ultralytics/yolov5', model_variant, autoshape=False, pretrained=False)  
        else:
            # Create an untrained model.
            model = create_model(model_dir, model_choice, model_variant, num_classes)



    elif model_selection_flag == 1:
        if model_choice == "yolo":
            model = torch.hub.load('ultralytics/yolov5', model_variant, autoshape=False, pretrained=True)
        else:
            # Load a pretrained model from Pytorch.
            model = torch.hub.load('pytorch/vision:v0.10.0', model_variant, pretrained=True)
    
    elif model_selection_flag == 2:
        try:
            # Load a local pretrained model.
            model = create_model(model_dir, model_choice, model_variant, num_classes)
            model = load_model(model=model, model_filepath=saved_model_filepath, device=device)
        except:
            # If above fails then the model is quantised, and it has go through the same process of quantisation then load the state dictionary of values.
            model = create_model(model_dir, model_choice, model_variant, num_classes)

            # Move the model to CPU since static quantization does not support CUDA currently.
            cpu_device = torch.device("cpu:0")
            model.to(cpu_device)

            # Make a copy of the model.
            fused_model = copy.deepcopy(model)

            # The model has to be switched to evaluation mode for quantisation.
            fused_model.eval()

            # Fusing layers imporves speed and accuracy quantized model performance. https://pytorch.org/blog/quantization-in-practice/
            # If model can be fused, fuse it.
            try: 
                fused_model = torch.quantization.fuse_modules(fused_model, [["conv1", "bn1", "relu"]], inplace=True)
                for module_name, module in fused_model.named_children():
                    if "layer" in module_name:
                        for basic_block_name, basic_block in module.named_children():
                            torch.quantization.fuse_modules(basic_block, [["conv1", "bn1", "relu1"], ["conv2", "bn2"]], inplace=True)
                            for sub_block_name, sub_block in basic_block.named_children():
                                if sub_block_name == "downsample":
                                    torch.quantization.fuse_modules(sub_block, [["0", "1"]], inplace=True)
            except:
                # Model did not fuse, so continuing without fusing.
                pass


            compressed_model = QuantizedNN(model_fp32=fused_model)
            quantization_config = torch.quantization.get_default_qconfig("fbgemm")
            compressed_model.qconfig = quantization_config
            torch.quantization.prepare(compressed_model, inplace=True)
            compressed_model = torch.quantization.convert(compressed_model, inplace=True)
            compressed_model.train()

            # load the real state dict
            compressed_model.load_state_dict(torch.load(saved_model_filepath))
            return compressed_model

    model.to(device)
    return model


def train_model(model, train_loader, test_loader, device, learning_rate=1e-2, num_epochs=200 ):

    criterion = nn.CrossEntropyLoss()

    model.to(device)

    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-5)
    
    for epoch in range(num_epochs):

        # Time
        t0 = time.time()

        # Training
        model.train()

        running_loss = 0
        running_corrects = 0

        for inputs, labels in train_loader:
            # print("Model training..")
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
        eval_loss, eval_accuracy = top1Accuracy(model=model, test_loader=test_loader, device=device, criterion=criterion)
        t_end = time.time() - t0
        print("Epoch: {:02d} Train Loss: {:.3f} Train Acc: {:.3f} Eval Loss: {:.3f} Eval Acc: {:.3f} Time(s) {:.4f}".format(epoch, train_loss, train_accuracy, eval_loss, eval_accuracy, t_end))

    return model