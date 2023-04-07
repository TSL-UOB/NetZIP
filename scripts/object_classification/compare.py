import os
import sys

sys.path.append('../../')

from utils.common import set_random_seeds, set_cuda
from utils.dataloader import pytorch_dataloader
from utils.model import model_selection, train_model, save_model
from utils.evaluation import evaluate_model
from utils.results_manager import log, plot_results
from metrics.accuracy.topAccuracy import top1Accuracy
from metrics.speed.latency import inference_latency

import yaml
import argparse
parser = argparse.ArgumentParser()                                               

parser.add_argument("--config_file", "-cf", type=str, required=True)
args = parser.parse_args()

with open(args.config_file, "r") as ymlfile:
    cfg = yaml.load(ymlfile)

SEED_NUMBER                     = cfg["SEED_NUMBER"]

USE_CUDA                        = cfg["USE_CUDA"]

DATASET_NAME                    = cfg["DATASET_NAME"] # Options: "CIFAR10" "CIFAR100" "TinyImageNet"  "ImageNet"
NUM_CLASSES                     = cfg["NUM_CLASSES"] # Number of classes in dataset

MODEL_CHOICE                    = cfg["MODEL_CHOICE"] # Option:"resnet" "vgg"
MODEL_VARIANT                   = cfg["MODEL_VARIANT"] # Common Options: "resnet18" "vgg11" For more options explore files in models to find the different options.

MODEL_DIR                       = "../../models/" + MODEL_CHOICE
MODEL_SELECTION_FLAG            = 2 # create an untrained model = 0, start from a pytorch trained model = 1, start from a previously saved local model = 2

UNCOMPRESSED_MODEL_FILENAME     = MODEL_VARIANT +"_"+DATASET_NAME+str(NUM_CLASSES)+".pt"
UNCOMPRESSED_MODEL_FILEPATH     = os.path.join(MODEL_DIR, UNCOMPRESSED_MODEL_FILENAME)

COMPRESSION_TECHNIQUES_LIST  = cfg["COMPRESSION_TECHNIQUES_LIST"]      # Option: "PTQ" "QAT" "GUP_R" "GUP_L1"

EVALUATION_METRICS_LIST = cfg["EVALUATION_METRICS_LIST"] 

output_plots              = cfg["OUTPUT_PLOTS"]

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

    results_log = log()
    for evaluation_metric in EVALUATION_METRICS_LIST:
        print("====== EVALUATION METRIC: ", evaluation_metric)
        # For Uncompressed model
        print("=== Compression Technique: None - Uncompressed model")
        value = evaluate_model(uncompressed_model, evaluation_metric, device, test_loader)
        results_log.append(MODEL_VARIANT, DATASET_NAME, evaluation_metric, "None", value)
        
        # For compressed model
        for compression_technique in COMPRESSION_TECHNIQUES_LIST:
            print("=== Compression Technique: ", compression_technique)
            compressed_model_filename        = compression_technique+"_"+MODEL_VARIANT +"_"+DATASET_NAME+str(NUM_CLASSES)+".pt"
            compressed_model_filepath        = os.path.join(MODEL_DIR, compressed_model_filename)
            compressed_model = model_selection(model_selection_flag=MODEL_SELECTION_FLAG, model_dir=MODEL_DIR, model_choice=MODEL_CHOICE, model_variant=MODEL_VARIANT, saved_model_filepath=compressed_model_filepath, num_classes=NUM_CLASSES, device=device)
            
            # Evaluate model
            value = evaluate_model(compressed_model, evaluation_metric, device, test_loader)
            
            # Log
            results_log.append(MODEL_VARIANT, DATASET_NAME, evaluation_metric, compression_technique, value)

    # Save output log results
    results_log.write_file()

    # Create Plots
    if output_plots:
        plot_results(results_log)


if __name__ == "__main__":

    main()

