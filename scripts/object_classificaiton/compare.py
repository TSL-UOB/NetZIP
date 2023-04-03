import os
import sys

sys.path.append('../../')

from utils.common import set_random_seeds, set_cuda
from utils.dataloader import pytorch_dataloader
from utils.model import model_selection, train_model, save_model
from metrics.accuracy.topAccuracy import top1Accuracy
from metrics.speed.latency import inference_latency

SEED_NUMBER              = 0
USE_CUDA                 = True

DATASET_NAME             = "TinyImageNet" # Options: "CIFAR10" "CIFAR100" "TinyImageNet"  "ImageNet"
NUM_CLASSES              = 1000 # Number of classes in dataset

MODEL_CHOICE             = "resnet" # Option:"resnet" "vgg"
MODEL_VARIANT            = "resnet18" # Common Options: "resnet18" "vgg11" For more options explore files in models to find the different options.

MODEL_DIR                = "../../models/" + MODEL_CHOICE
MODEL_SELECTION_FLAG     = 2 # create an untrained model = 0, start from a pytorch trained model = 1, start from a previously saved local model = 2

UNCOMPRESSED_MODEL_FILENAME     = MODEL_VARIANT +"_"+DATASET_NAME+str(NUM_CLASSES)+".pt"
UNCOMPRESSED_MODEL_FILEPATH     = os.path.join(MODEL_DIR, SAVED_MODEL_FILENAME)

COMPRESSION_TECHNIQUES_LIST  = ["PTQ", "QAT", "GUP_L1", "GUP_R"]      # Option: "PTQ" "QAT" "GUP_R" "GUP_L1"

EVALUATION_METRICS_LIST = [] 

# Option:
# == Accuracy: "TOP1accuracy" "TOP5accuracy" "mAP" "Precision" "Recall" "F1Score"
# == Size    : "MemorySize" "RAMutilisation"
# == Speed   : "Latency" "MAC" "FLOPS"
# == Energy  : "Energy" "Power"

def main():
    # Fix seeds to allow for repeatable results 
    set_random_seeds(SEED_NUMBER)

    # Setup device used for training either gpu or cpu
    device = set_cuda(USE_CUDA)

    # Setup dataset
    train_loader, test_loader = pytorch_dataloader(dataset_name=DATASET_NAME)
    print("Progress: Dataset Loaded.")

    # Setup original model
    uncompressed_model = model_selection(model_selection_flag=MODEL_SELECTION_FLAG, model_dir=MODEL_DIR, model_choice=MODEL_CHOICE, model_variant=MODEL_VARIANT, saved_model_filepath=UNCOMPRESSED_MODEL_FILEPATH, num_classes=NUM_CLASSES, device=device)
    print("Progress: Model has been setup.")

    
    for compression_technique in COMPRESSION_TECHNIQUES_LIST:
        compressed_model_filename        = compression_technique+"_"+MODEL_VARIANT +"_"+DATASET_NAME+str(NUM_CLASSES)+".pt"
        compressed_model_filepath        = os.path.join(MODEL_DIR, compressed_model_filename)
        compressed_model = model_selection(model_selection_flag=MODEL_SELECTION_FLAG, model_dir=MODEL_DIR, model_choice=MODEL_CHOICE, model_variant=MODEL_VARIANT, saved_model_filepath=compressed_model_filepath, num_classes=NUM_CLASSES, device=device)

    # Evaluate model
    _,eval_accuracy     = top1Accuracy(model=model, test_loader=test_loader, device=device, criterion=None)
    eval_speed_latency = inference_latency(model=model, device=device, input_size=(1,3,32,32), num_samples=100)

    print("FP32 evaluation accuracy: {:.3f}".format(eval_accuracy))
    print("FP32 CPU Inference Latency: {:.2f} ms / sample".format(eval_speed_latency * 1000))

if __name__ == "__main__":

    main()

