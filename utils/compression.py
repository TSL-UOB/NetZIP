from methods.PTQ import PTQ
from methods.QAT import QAT

def compress_model(model, compression_technique, device, train_loader, test_loader="",  learning_rate=1e-3 , num_epochs=10):
    if compression_technique == "PTQ":
        compressed_model = PTQ(model, train_loader)

    elif compression_technique == "QAT":
        compressed_model = QAT(model, train_loader, test_loader, device)

    else:
        print("ERROR: An unknown compression method was selected.")
    return compressed_model
