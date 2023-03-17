import os

import importlib

import torch
import torch.nn as nn
import torch.optim as optim

from metrics.accuracy.topAccuracy import top1Accuracy


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
        # Create an untrained model.
        model = create_model(model_dir, model_choice, model_variant, num_classes)

    elif model_selection_flag == 1:
        # Load a pretrained model from Pytorch.
        model = torch.hub.load('pytorch/vision:v0.10.0', model_variant, pretrained=True)
    
    elif model_selection_flag == 2:
        # Load a local pretrained model.
        model = create_model(model_dir, model_choice, model_variant, num_classes)
        model = load_model(model=model, model_filepath=saved_model_filepath, device=device)

    return model


def train_model(model, train_loader, test_loader, device, learning_rate=1e-2, num_epochs=200 ):

    criterion = nn.CrossEntropyLoss()

    model.to(device)

    # It seems that SGD optimizer is better than Adam optimizer for ResNet18 training on CIFAR10.
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-5)
    # optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)

    
    for epoch in range(num_epochs):

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

        print("Epoch: {:02d} Train Loss: {:.3f} Train Acc: {:.3f} Eval Loss: {:.3f} Eval Acc: {:.3f}".format(epoch, train_loss, train_accuracy, eval_loss, eval_accuracy))

    return model