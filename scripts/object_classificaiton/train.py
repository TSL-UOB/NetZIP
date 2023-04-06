import os
import sys

sys.path.append('../../')

from utils.common import set_random_seeds, set_cuda
from utils.dataloader import pytorch_dataloader
from utils.model import model_selection, train_model, save_model
from metrics.accuracy.topAccuracy import top1Accuracy
from metrics.speed.latency import inference_latency

import yaml
import argparse
parser = argparse.ArgumentParser()                                               

parser.add_argument("--config_file", "-cf", type=str, required=True)
args = parser.parse_args()

with open(args.config_file, "r") as ymlfile:
    cfg = yaml.load(ymlfile, yaml.Loader)

SEED_NUMBER                      = cfg["SEED_NUMBER"]

USE_CUDA                         = cfg["USE_CUDA"]

DATASET_NAME                     = cfg["DATASET_NAME"] # Options: "CIFAR10" "CIFAR100" "TinyImageNet"  "ImageNet"
NUM_CLASSES                      = cfg["NUM_CLASSES"] # Number of classes in dataset

MODEL_CHOICE                     = cfg["MODEL_CHOICE"] # Option:"resnet" "vgg"
MODEL_VARIANT                    = cfg["MODEL_VARIANT"] # Common Options: "resnet18" "vgg11" For more options explore files in models to find the different options.

MODEL_DIR                        = "../../models/" + MODEL_CHOICE
MODEL_SELECTION_FLAG             = cfg["MODEL_SELECTION_FLAG"] # create an untrained model = 0, start from a pytorch trained model = 1, start from a previously saved local model = 2

SAVED_MODEL_FILENAME             = MODEL_VARIANT +"_"+DATASET_NAME+str(NUM_CLASSES)+".pt"
SAVED_MODEL_FILEPATH             = os.path.join(MODEL_DIR, SAVED_MODEL_FILENAME)

TRAINED_MODEL_FILENAME           = MODEL_VARIANT +"_"+DATASET_NAME+str(NUM_CLASSES)+".pt"

NUM_EPOCHS                       = cfg["NUM_EPOCHS"]
LEARNING_RATE                    = cfg["LEARNING_RATE"] # for imagenet use 1e-5, otherwise 1e-2

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
    # eval_speed_latency = inference_latency(model=model, device=device, test_loader=test_loader)#input_size=(1,3,32,32), num_samples=100)

    print("FP32 evaluation accuracy: {:.3f}".format(eval_accuracy))
    # print("FP32 CPU Inference Latency: {:.2f} ms / sample".format(eval_speed_latency * 1000))

if __name__ == "__main__":

    main()

