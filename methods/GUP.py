# Global Unstructured Pruning (GUP) method.

import os
import torch
import torch.nn as nn
import torch.optim as optim
import copy
import torch.nn.utils.prune as prune
import sklearn.metrics

from metrics.accuracy.topAccuracy import top1Accuracy


def GUP(model, train_loader, test_loader, device):
    l1_regularization_strength = 0
    l2_regularization_strength = 1e-4
    learning_rate = 1e-3
    learning_rate_decay = 1

    num_zeros, num_elements, sparsity = measure_global_sparsity(model)

    pruned_model = copy.deepcopy(model)

    iterative_pruning_finetuning(
        model=pruned_model,
        train_loader=train_loader,
        test_loader=test_loader,
        device=device,
        learning_rate=learning_rate,
        learning_rate_decay=learning_rate_decay,
        l1_regularization_strength=l1_regularization_strength,
        l2_regularization_strength=l2_regularization_strength,
        conv2d_prune_amount=0.98,
        linear_prune_amount=0,
        num_iterations=1,
        num_epochs_per_iteration=10,
        model_filename_prefix="pruned_model",
        model_dir="../../models/" + "resnet",
        grouped_pruning=True)

    # Apply mask to the parameters and remove the mask.
    remove_parameters(model=pruned_model)

    _, eval_accuracy = top1Accuracy(model=pruned_model,
                                      test_loader=test_loader,
                                      device=device,
                                      criterion=None)


    classification_report = create_classification_report(
        model=pruned_model, test_loader=test_loader, device=device)

    num_zeros, num_elements, sparsity = measure_global_sparsity(pruned_model)

    print("Test Accuracy: {:.3f}".format(eval_accuracy))
    print("Classification Report:")
    print(classification_report)
    print("Global Sparsity:")
    print("{:.2f}".format(sparsity))

    pruned_model.train()

    return pruned_model

def create_classification_report(model, device, test_loader):

    model.eval()
    model.to(device)

    y_pred = []
    y_true = []

    with torch.no_grad():
        for data in test_loader:
            y_true += data[1].numpy().tolist()
            images, _ = data[0].to(device), data[1].to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            y_pred += predicted.cpu().numpy().tolist()

    classification_report = sklearn.metrics.classification_report(
        y_true=y_true, y_pred=y_pred)

    return classification_report

def measure_module_sparsity(module, weight=True, bias=False, use_mask=False):

    num_zeros = 0
    num_elements = 0

    if use_mask == True:
        for buffer_name, buffer in module.named_buffers():
            if "weight_mask" in buffer_name and weight == True:
                num_zeros += torch.sum(buffer == 0).item()
                num_elements += buffer.nelement()
            if "bias_mask" in buffer_name and bias == True:
                num_zeros += torch.sum(buffer == 0).item()
                num_elements += buffer.nelement()
    else:
        for param_name, param in module.named_parameters():
            if "weight" in param_name and weight == True:
                num_zeros += torch.sum(param == 0).item()
                num_elements += param.nelement()
            if "bias" in param_name and bias == True:
                num_zeros += torch.sum(param == 0).item()
                num_elements += param.nelement()

    sparsity = num_zeros / num_elements

    return num_zeros, num_elements, sparsity

def measure_global_sparsity(model,
                            weight=True,
                            bias=False,
                            conv2d_use_mask=False,
                            linear_use_mask=False):

    num_zeros = 0
    num_elements = 0

    for module_name, module in model.named_modules():

        if isinstance(module, torch.nn.Conv2d):

            module_num_zeros, module_num_elements, _ = measure_module_sparsity(
                module, weight=weight, bias=bias, use_mask=conv2d_use_mask)
            num_zeros += module_num_zeros
            num_elements += module_num_elements

        elif isinstance(module, torch.nn.Linear):

            module_num_zeros, module_num_elements, _ = measure_module_sparsity(
                module, weight=weight, bias=bias, use_mask=linear_use_mask)
            num_zeros += module_num_zeros
            num_elements += module_num_elements

    sparsity = num_zeros / num_elements

    return num_zeros, num_elements, sparsity

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


def iterative_pruning_finetuning(model,
                                 train_loader,
                                 test_loader,
                                 device,
                                 learning_rate,
                                 l1_regularization_strength,
                                 l2_regularization_strength,
                                 learning_rate_decay=0.1,
                                 conv2d_prune_amount=0.4,
                                 linear_prune_amount=0.2,
                                 num_iterations=10,
                                 num_epochs_per_iteration=10,
                                 model_filename_prefix="pruned_model",
                                 model_dir="saved_models",
                                 grouped_pruning=False):

    for i in range(num_iterations):

        print("Pruning and Finetuning {}/{}".format(i + 1, num_iterations))

        print("Pruning...")

        if grouped_pruning == True:
            # Global pruning
            # I would rather call it grouped pruning.
            parameters_to_prune = []
            for module_name, module in model.named_modules():
                if isinstance(module, torch.nn.Conv2d):
                    parameters_to_prune.append((module, "weight"))
            prune.global_unstructured(
                parameters_to_prune,
                pruning_method=prune.L1Unstructured,
                amount=conv2d_prune_amount,
            )
        else:
            for module_name, module in model.named_modules():
                if isinstance(module, torch.nn.Conv2d):
                    prune.l1_unstructured(module,
                                          name="weight",
                                          amount=conv2d_prune_amount)
                elif isinstance(module, torch.nn.Linear):
                    prune.l1_unstructured(module,
                                          name="weight",
                                          amount=linear_prune_amount)

        _, eval_accuracy = top1Accuracy(model=model,
                                          test_loader=test_loader,
                                          device=device,
                                          criterion=None)

        classification_report = create_classification_report(
            model=model, test_loader=test_loader, device=device)

        num_zeros, num_elements, sparsity = measure_global_sparsity(
            model,
            weight=True,
            bias=False,
            conv2d_use_mask=True,
            linear_use_mask=False)

        print("Test Accuracy: {:.3f}".format(eval_accuracy))
        print("Classification Report:")
        print(classification_report)
        print("Global Sparsity:")
        print("{:.2f}".format(sparsity))

        # print(model.conv1._forward_pre_hooks)

        print("Fine-tuning...")

        train_model(model=model,
                    train_loader=train_loader,
                    test_loader=test_loader,
                    device=device,
                    l1_regularization_strength=l1_regularization_strength,
                    l2_regularization_strength=l2_regularization_strength,
                    learning_rate=learning_rate * (learning_rate_decay**i),
                    num_epochs=num_epochs_per_iteration)

        _, eval_accuracy = top1Accuracy(model=model,
                                          test_loader=test_loader,
                                          device=device,
                                          criterion=None)

        classification_report = create_classification_report(
            model=model, test_loader=test_loader, device=device)

        num_zeros, num_elements, sparsity = measure_global_sparsity(
            model,
            weight=True,
            bias=False,
            conv2d_use_mask=True,
            linear_use_mask=False)

        print("Test Accuracy: {:.3f}".format(eval_accuracy))
        print("Classification Report:")
        print(classification_report)
        print("Global Sparsity:")
        print("{:.2f}".format(sparsity))

        model_filename = "{}_{}.pt".format(model_filename_prefix, i + 1)
        model_filepath = os.path.join(model_dir, model_filename)
        # save_model(model=model,
                   # model_dir=model_dir,
                   # model_filename=model_filename)
        # model = load_model(model=model,
        #                    model_filepath=model_filepath,
        #                    device=device)

    return model

def train_model(model,
                train_loader,
                test_loader,
                device,
                l1_regularization_strength=0,
                l2_regularization_strength=1e-4,
                learning_rate=1e-1,
                num_epochs=200):

    # The training configurations were not carefully selected.

    criterion = nn.CrossEntropyLoss()

    model.to(device)

    # It seems that SGD optimizer is better than Adam optimizer for ResNet18 training on CIFAR10.
    optimizer = optim.SGD(model.parameters(),
                          lr=learning_rate,
                          momentum=0.9,
                          weight_decay=l2_regularization_strength)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=500)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                     milestones=[100, 150],
                                                     gamma=0.1,
                                                     last_epoch=-1)
    # optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)

    # Evaluation
    model.eval()
    eval_loss, eval_accuracy = top1Accuracy(model=model,
                                              test_loader=test_loader,
                                              device=device,
                                              criterion=criterion)
    print("Epoch: {:03d} Eval Loss: {:.3f} Eval Acc: {:.3f}".format(
        0, eval_loss, eval_accuracy))

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

            l1_reg = torch.tensor(0.).to(device)
            for module in model.modules():
                mask = None
                weight = None
                for name, buffer in module.named_buffers():
                    if name == "weight_mask":
                        mask = buffer
                for name, param in module.named_parameters():
                    if name == "weight_orig":
                        weight = param
                # We usually only want to introduce sparsity to weights and prune weights.
                # Do the same for bias if necessary.
                if mask is not None and weight is not None:
                    l1_reg += torch.norm(mask * weight, 1)

            loss += l1_regularization_strength * l1_reg

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