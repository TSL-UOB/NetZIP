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

    # Remove pruning re-parametrization to make pruning permenant.
    remove_parameters(model=pruned_model)

    _, eval_accuracy = top1Accuracy(model=pruned_model,
                                      test_loader=test_loader,
                                      device=device,
                                      criterion=None)

    num_zeros, num_elements, sparsity = get_global_sparsity(pruned_model)

    print("Test Accuracy: {:.3f}".format(eval_accuracy))
    print("Global Sparsity of pruned model:")
    print("{:.2f}".format(sparsity))

    pruned_model.train()

    return pruned_model

def prune_and_finetune(model, train_loader, test_loader, device, method, prune_amount, 
    num_of_pruning_iterations, num_epochs_per_iteration, learning_rate, learning_rate_decay):

    for i in range(num_of_pruning_iterations):

        print("Pruning and Finetuning {}/{}".format(i + 1, num_of_pruning_iterations))

        print("Pruning...")

        # Global pruning
        parameters_to_prune = []
        for module_name, module in model.named_modules():
            if isinstance(module, torch.nn.Conv2d):             
                parameters_to_prune.append((module, "weight"))
                # parameters_to_prune.append((module, "bias"))

            if isinstance(module, torch.nn.Linear):
                parameters_to_prune.append((module, "weight"))
                # parameters_to_prune.append((module, "bias"))

        num_zeros, num_elements, sparsity = get_global_sparsity(model)

        _, eval_accuracy = top1Accuracy(model=model, test_loader=test_loader, device=device, criterion=None)
        print("Test Accuracy: {:.3f}".format(eval_accuracy))
        
        if method == "Random":
            prune.global_unstructured(parameters_to_prune,pruning_method=prune.RandomUnstructured,amount=prune_amount)
        elif method == "L1":
            prune.global_unstructured(parameters_to_prune,pruning_method=prune.L1Unstructured,amount=prune_amount)
        
        # classification_report = create_classification_report(model=model, test_loader=test_loader, device=device)

        num_zeros, num_elements, sparsity = get_global_sparsity(model)
        print("Global Sparsity after pruning:")
        print("{:.2f}".format(sparsity*2))

        _, eval_accuracy = top1Accuracy(model=model, test_loader=test_loader, device=device, criterion=None)
        print("Test Accuracy before Finetuning: {:.3f}".format(eval_accuracy))

        print("Fine-tuning...")

        train_model(model=model, train_loader=train_loader, test_loader=test_loader,device=device,
                    learning_rate=learning_rate * (learning_rate_decay**i), num_epochs=num_epochs_per_iteration)

        _, eval_accuracy = top1Accuracy(model=model, test_loader=test_loader, device=device, criterion=None)

        num_zeros, num_elements, sparsity = get_global_sparsity(model)

        print("Test Accuracy after Finetuning: {:.3f}".format(eval_accuracy))


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