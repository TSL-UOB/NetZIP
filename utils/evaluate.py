from metrics. import PTQ
from metrics. import GUP

def evaluate_model(models, metrics, device, test_loader=""):
    if compression_technique == "PTQ":
        compressed_model = PTQ(model, train_loader)

    elif compression_technique == "QAT":
        compressed_model = QAT(model, train_loader, test_loader, device)

    elif compression_technique == "GUP_R":
        compressed_model = GUP(model, train_loader, test_loader, device, method = "Random", prune_amount=0.75, num_epochs_per_iteration=num_epochs)

    elif compression_technique == "GUP_L1":
        compressed_model = GUP(model, train_loader, test_loader, device, method = "L1", prune_amount=0.75, num_epochs_per_iteration=num_epochs)

    else:
        raise Exception("ERROR: An unknown compression method was selected.")
    
    return compressed_model

def metrics_switch():
    pass