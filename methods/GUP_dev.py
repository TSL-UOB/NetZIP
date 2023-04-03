# Global Unstructured Pruning (GUP) method.

import os
import torch
torch.cuda.empty_cache()
import torch.nn as nn
import torch.optim as optim
import copy
import torch.nn.utils.prune as prune
import sklearn.metrics

from metrics.accuracy.topAccuracy import top1Accuracy
from metrics.size.sparsity import get_global_sparsity, get_module_sparsity

# == Summary of what is availble in  PyTorch
# RANDOMUNSTRUCTURED - Prune (currently unpruned) units in a tensor at random. (needs to define "amount" quantity of parameter to prune).

# L1UNSTRUCTURED - Prune (currently unpruned) units in a tensor by zeroing out the ones with the lowest L1-norm.

# RANDOMSTRUCTURED (Will not implement) - Prune entire (currently unpruned) channels in a tensor at random. (needs to define "amount" quantity of parameter to prune and "dim" index of dim along which we define channels to prune).

# LNSTRUCTURED (Will not implement)- Prune entire (currently unpruned) channels in a tensor based on their Ln-norm.

# GLOBAL_UNSTRUCTURED - Globally prunes tensors corresponding to all parameters in parameters by applying the specified pruning_method (needs to be combined with one of the unstructured methods above).

# == Implemented in NetZIP
# GLOBAL_RANDOM_UNSTRUCTURED -Globally prunes tensors corresponding to all parameters in parameters by applying random unstructured pruning.

# GLOBAL_L1_UNSTRUCTURED -Globally prunes tensors corresponding to all parameters in parameters by applying L1 unstructured pruning.

# def get_module_sparsity_v2(module, weight=True, bias=True):
#     num_zeros = 0
#     num_elements = 0

#     for buffer_name, buffer in module.named_buffers():
#         if "weight_mask" in buffer_name and weight == True:
#             num_zeros += torch.sum(buffer == 0).item()
#             num_elements += buffer.nelement()
#         if "bias_mask" in buffer_name and bias == True:
#             num_zeros += torch.sum(buffer == 0).item()
#             num_elements += buffer.nelement()

#     for param_name, param in module.named_parameters():
#         if "weight" in param_name and weight == True:
#             num_zeros += torch.sum(param == 0).item()
#             num_elements += param.nelement()
#         if "bias" in param_name and bias == True:
#             num_zeros += torch.sum(param == 0).item()
#             num_elements += param.nelement()

    
#     if num_elements == 0:
#         # print("========================")
#         # print(list(module.named_parameters()))
#         # print(module)
#         # input("Number of elements = 0")
#         sparsity = float('inf') 
#     else:
#         sparsity = num_zeros / num_elements

    

#     return num_zeros, num_elements, sparsity

def prune_and_finetune(model, train_loader, test_loader, device, method, prune_amount, 
    num_of_pruning_iterations, num_epochs_per_iteration, learning_rate, learning_rate_decay):

    for i in range(num_of_pruning_iterations):

        print("Pruning and Finetuning {}/{}".format(i + 1, num_of_pruning_iterations))

        print("Pruning...")

        # Global pruning
        parameters_to_prune = []
        for module_name, module in model.named_modules():
        #     print(module_name)
        #     # print(module)
        #     # print(list(module.named_parameters()))
            if isinstance(module, torch.nn.Conv2d):             
                parameters_to_prune.append((module, "weight"))
                # parameters_to_prune.append((module, "bias"))

            if isinstance(module, torch.nn.Linear):
                parameters_to_prune.append((module, "weight"))
                # parameters_to_prune.append((module, "bias"))


        #     # print(list(module.named_parameters()))
        #     # module_num_zeros, module_num_elements, _ = get_module_sparsity(module, weight=weight, bias=bias, use_mask=linear_use_mask)
        #     num_zeros, num_elements, sparsity = get_module_sparsity(module)
            
        #     print("num_zeros = ",num_zeros)
        #     print("num_elements = ",num_elements)
        #     print("sparsity = ",sparsity)
        #     input("press eneter to continue")
        num_zeros, num_elements, sparsity = get_global_sparsity(model)
        print("Global Sparsity before pruning:")
        print("{:.2f}".format(sparsity))

        _, eval_accuracy = top1Accuracy(model=model, test_loader=test_loader, device=device, criterion=None)
        print("Test Accuracy: {:.3f}".format(eval_accuracy))
        
        if method == "Random":
            prune.global_unstructured(parameters_to_prune,pruning_method=prune.RandomUnstructured,amount=prune_amount)
        elif method == "L1":
            prune.global_unstructured(parameters_to_prune,pruning_method=prune.L1Unstructured,amount=prune_amount)
        
        # classification_report = create_classification_report(model=model, test_loader=test_loader, device=device)

        num_zeros, num_elements, sparsity = get_global_sparsity(model)
        print("Global Sparsity after pruning:")
        print("{:.2f}".format(sparsity))

        _, eval_accuracy = top1Accuracy(model=model, test_loader=test_loader, device=device, criterion=None)
        print("Test Accuracy: {:.3f}".format(eval_accuracy))
        # print("Classification Report:")
        # print(classification_report)
        

        # print(model.conv1._forward_pre_hooks)

        print("Fine-tuning...")

        train_model(model=model, train_loader=train_loader, test_loader=test_loader,device=device,
                    learning_rate=learning_rate * (learning_rate_decay**i), num_epochs=num_epochs_per_iteration)

        _, eval_accuracy = top1Accuracy(model=model, test_loader=test_loader, device=device, criterion=None)

        # classification_report = create_classification_report(model=model, test_loader=test_loader, device=device)

        num_zeros, num_elements, sparsity = get_global_sparsity(model)

        print("Test Accuracy: {:.3f}".format(eval_accuracy))
        # print("Classification Report:")
        # print(classification_report)
        print("Global Sparsity:")
        print("{:.2f}".format(sparsity))

def GUP(model, train_loader, test_loader, device, method="Random", prune_amount=0.25, 
    num_of_pruning_iterations=1, num_epochs_per_iteration=10, 
    learning_rate = 1e-3, learning_rate_decay = 1):

    num_zeros, num_elements, sparsity = get_global_sparsity(model)
    print("Global Sparsity of unpruned model:")
    print("{:.2f}".format(sparsity))

    pruned_model = copy.deepcopy(model)

    prune_and_finetune(pruned_model, train_loader, test_loader, device, 
                        method, prune_amount, 
                        num_of_pruning_iterations, 
                        num_epochs_per_iteration, 
                        learning_rate, learning_rate_decay)

    # Apply mask to the parameters and remove the mask.
    remove_parameters(model=pruned_model)

    _, eval_accuracy = top1Accuracy(model=pruned_model,
                                      test_loader=test_loader,
                                      device=device,
                                      criterion=None)


    # classification_report = create_classification_report(
        # model=pruned_model, test_loader=test_loader, device=device)

    # Replace measure_global_sparsity with metric for meausring sparsity, speed, size (based on sparsity)

    num_zeros, num_elements, sparsity = get_global_sparsity(pruned_model)

    print("Test Accuracy: {:.3f}".format(eval_accuracy))
    # print("Classification Report:")
    # print(classification_report)
    print("Global Sparsity of pruned model:")
    print("{:.2f}".format(sparsity))

    pruned_model.train()

    return pruned_model

# def create_classification_report(model, device, test_loader):

#     model.eval()
#     model.to(device)

#     y_pred = []
#     y_true = []

#     with torch.no_grad():
#         for data in test_loader:
#             y_true += data[1].numpy().tolist()
#             images, _ = data[0].to(device), data[1].to(device)
#             outputs = model(images)
#             _, predicted = torch.max(outputs.data, 1)
#             y_pred += predicted.cpu().numpy().tolist()

#     classification_report = sklearn.metrics.classification_report(
#         y_true=y_true, y_pred=y_pred)

#     return classification_report


def remove_parameters(model):

    for module_name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            try:
                prune.remove(module, "weight")
            except:
                pass
            try:
                prune.remove(module, "bias")
            except:
                pass
        elif isinstance(module, torch.nn.Linear):
            try:
                prune.remove(module, "weight")
            except:
                pass
            try:
                prune.remove(module, "bias")
            except:
                pass

    return model


def train_model(model,
                train_loader,
                test_loader,
                device,
                learning_rate=1e-1,
                num_epochs=200):

    # The training configurations were not carefully selected.
    criterion = nn.CrossEntropyLoss()

    model.to(device)

    # Set SGD optimizer. 
    optimizer = optim.SGD(model.parameters(),
                          lr=learning_rate,
                          momentum=0.9,
                          weight_decay=1e-5)

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                     milestones=[100, 150],
                                                     gamma=0.1,
                                                     last_epoch=-1)
    
    # Evaluation
    model.eval()
    eval_loss, eval_accuracy = top1Accuracy(model=model,
                                              test_loader=test_loader,
                                              device=device,
                                              criterion=criterion)
    print("Epoch: {:03d} Eval Loss: {:.3f} Eval Acc: {:.3f}".format(0, eval_loss, eval_accuracy))

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
        eval_loss, eval_accuracy = top1Accuracy(model=model,
                                                  test_loader=test_loader,
                                                  device=device,
                                                  criterion=criterion)

        # Set learning rate scheduler
        scheduler.step()

        print(
            "Epoch: {:03d} Train Loss: {:.3f} Train Acc: {:.3f} Eval Loss: {:.3f} Eval Acc: {:.3f}"
            .format(epoch + 1, train_loss, train_accuracy, eval_loss,
                    eval_accuracy))

    return model