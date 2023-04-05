from methods.PTQ import PTQ
from methods.QAT import QAT
from methods.GUP import GUP

def compress_model(model, compression_technique, device, train_loader, test_loader="",  learning_rate=1e-3 , num_epochs=10):
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