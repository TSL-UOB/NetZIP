import torch 
import os
import sys

sys.path.append('../../')

from utils.common import set_random_seeds, set_cuda
from utils.dataloader import pytorch_dataloader
from utils.model import model_selection, save_model
from utils. compression import compress_model
from metrics.accuracy.topAccuracy import top1Accuracy
from metrics.speed.latency import inference_latency

import yaml
import argparse
parser = argparse.ArgumentParser()                                               

parser.add_argument("--config_file", "-cf", type=str, required=True)
args = parser.parse_args()

with open(args.config_file, "r") as ymlfile:
    cfg = yaml.load(ymlfile)
        
SEED_NUMBER                      = cfg["SEED_NUMBER"]

USE_CUDA                         = cfg["USE_CUDA"]

DATASET_NAME                     = cfg["DATASET_NAME"] # Options: "CIFAR10" "CIFAR100" "TinyImageNet"  "ImageNet"
NUM_CLASSES                      = cfg["NUM_CLASSES"] # Number of classes in dataset

MODEL_CHOICE                     = cfg["MODEL_CHOICE"] # Option:"resnet" "vgg"
MODEL_VARIANT                    = cfg["MODEL_VARIANT"] # Common Options: "resnet18" "vgg11" For more options explore files in models to find the different options.

COMPRESSION_TECHNIQUE            = cfg["COMPRESSION_TECHNIQUE"]      # Option: "PTQ" "QAT" "GUP_R" "GUP_L1"

MODEL_DIR                        = "../../models/" + MODEL_CHOICE

MODEL_TO_BE_COMPRESSED_FILENAME  = MODEL_VARIANT +"_"+DATASET_NAME+str(NUM_CLASSES)+".pt"
MODEL_TO_BE_COMPRESSED_FILEPATH  = os.path.join(MODEL_DIR, MODEL_TO_BE_COMPRESSED_FILENAME)

COMPRESSED_MODEL_FILENAME        = COMPRESSION_TECHNIQUE+"_"+MODEL_VARIANT +"_"+DATASET_NAME+str(NUM_CLASSES)+".pt"
COMPRESSED_MODEL_FILEPATH        = os.path.join(MODEL_DIR, COMPRESSED_MODEL_FILENAME)

NUM_EPOCHS                       = cfg["NUM_EPOCHS"]
LEARNING_RATE                    = cfg["LEARNING_RATE"] # for imagenet use 1e-5, otherwise 1e-2

MODEL_SELECTION_FLAG     = 2 # create an untrained model = 0, start from a pytorch trained model = 1, start from a previously saved local model = 2


def main():
    # Fix seeds to allow for repeatable results 
    set_random_seeds(SEED_NUMBER)

    # Setup device used for training either gpu or cpu
    device = set_cuda(USE_CUDA)

    # Setup dataset
    train_loader, test_loader = pytorch_dataloader(dataset_name=DATASET_NAME)
    print("Progress: Dataset Loaded.")
    
    # Setup model
    model = model_selection(model_selection_flag=MODEL_SELECTION_FLAG, model_dir=MODEL_DIR, model_choice=MODEL_CHOICE, model_variant=MODEL_VARIANT, saved_model_filepath=MODEL_TO_BE_COMPRESSED_FILEPATH, num_classes=NUM_CLASSES, device=device)
    print("Progress: Model has been setup.")

    # Compress model
    compressed_model = compress_model(model, COMPRESSION_TECHNIQUE, device, train_loader, test_loader, learning_rate=LEARNING_RATE , num_epochs=NUM_EPOCHS)
    
    # Save compressed model.
    save_model(model=compressed_model, model_dir=MODEL_DIR, model_filename=COMPRESSED_MODEL_FILENAME)
    print("Progress: Compressed Model Saved.")


    # Evaluate accuracy of original model and compressed model
    _,model_eval_accuracy                = top1Accuracy(model=model, test_loader=test_loader, device=device, criterion=None)
    print("Original model accuracy: {:.3f}".format(model_eval_accuracy))
    
    cpu_device = torch.device("cpu:0") # Use CPU device for assessing compressed model. Quantisation only works on CPU.
    _,compressed_model_eval_accuracy     = top1Accuracy(model=compressed_model, test_loader=test_loader, device=cpu_device, criterion=None)
    print("Compressed model accuracy: {:.3f}".format(compressed_model_eval_accuracy))

if __name__ == "__main__":

    main()

