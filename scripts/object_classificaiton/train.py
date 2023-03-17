import os
import sys
# import importlib

sys.path.append('../../')

from utils.common import set_random_seeds, set_cuda
from utils.dataloader import pytorch_dataloader
from utils.model import model_selection, train_model, save_model
from metrics.accuracy.topAccuracy import top1Accuracy
from metrics.speed.latency import inference_latency


# from models.resnet.resnet import resnet18

SEED_NUMBER              = 0
USE_CUDA                 = True

DATASET_NAME             = "TinyImageNet" # Options: "CIFAR10" "CIFAR100" "TinyImageNet"  "ImageNet"
NUM_CLASSES              = 200 # Number of classes in dataset

MODEL_CHOICE             = "resnet" # Option: Explore files in models to find the different options.
MODEL_VARIANT            = "resnet18" # Option: Explore files in models to find the different options.

MODEL_DIR                = "../../models/resnet"
SAVED_MODEL_FILENAME     = MODEL_CHOICE +"_"+DATASET_NAME+".pt"
MODEL_SELECTION_FLAG     = 1 # create an untrained model = 0, start from a pytorch trained model = 1, start from a previously saved local model = 2
SAVED_MODEL_FILEPATH     = os.path.join(MODEL_DIR, SAVED_MODEL_FILENAME)

TRAINED_MODEL_FILENAME   = MODEL_CHOICE +"_"+DATASET_NAME+".pt"


NUM_EPOCHS               = 2
LEARNING_RATE            = 1e-2

# model_module_path = "../../"+MODEL_DIR+"/"+MODEL_CHOICE+".py"
# model_module      = importlib.util.spec_from_file_location("",model_module_path).loader.load_module()
# model_function    = getattr(model_module, MODEL_VARIANT)#importlib.import_module(model_module0, "resnet18")
# model             = model_function(num_classes=NUM_CLASSES, pretrained=False)

# print(model)
# input("Succesful dynami import")
def main():
    # Fix seeds to allow for repeatable results 
    set_random_seeds(SEED_NUMBER)

    # Setup device used for training either gpu or cpu
    device = set_cuda(USE_CUDA)

    # Setup dataset
    train_loader, test_loader = pytorch_dataloader(dataset_name=DATASET_NAME)
    print("Progress: Dataset Loaded.")
    
    # Setup model
    model = model_selection(model_selection_flag=MODEL_SELECTION_FLAG, model_dir=MODEL_DIR, model_choice=MODEL_CHOICE, model_variant=MODEL_VARIANT, saved_model_filepath=SAVED_MODEL_FILEPATH, num_classes=NUM_CLASSES, device=device)
    print("Progress: Model has been setup.")
    
    # Train model
    model = train_model(model=model, train_loader=train_loader, test_loader=test_loader, device=device, learning_rate=LEARNING_RATE, num_epochs=NUM_EPOCHS)
    print("Progress: Model training done.")

    # Save model.
    save_model(model=model, model_dir=MODEL_DIR, model_filename=TRAINED_MODEL_FILENAME)
    print("Progress: Model Saved.")

    # Evaluate model
    _,eval_accuracy     = top1Accuracy(model=model, test_loader=test_loader, device=device, criterion=None)
    eval_speed_latency = inference_latency(model=model, device=device, input_size=(1,3,32,32), num_samples=100)
    
    print("FP32 evaluation accuracy: {:.3f}".format(eval_accuracy))
    print("FP32 CPU Inference Latency: {:.2f} ms / sample".format(eval_speed_latency * 1000))
    
if __name__ == "__main__":

    main()

