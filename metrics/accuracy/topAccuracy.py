import torch

def top1Accuracy(model, test_loader, device, criterion=None):

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


def top5Accuracy(model, test_loader, device):

    model.eval()
    model.to(device)

    running_corrects = 0

    for inputs, labels in test_loader:
        
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)

        for i, output in enumerate(outputs):
            result = torch.topk(output,5)
            top5preds = result.indices
            label = labels.data[i]
            if label in top5preds:
                running_corrects += 1

    eval_accuracy = running_corrects / len(test_loader.dataset)

    return eval_accuracy
