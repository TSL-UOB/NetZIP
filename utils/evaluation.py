from metrics.accuracy.topAccuracy import top1Accuracy
from metrics.speed.latency import inference_latency
from metrics.speed.ops import macs, flops
from metrics.size.size import model_size, gpu_mem_usage, cpu_mem_usage, parameters_count 
from metrics.size.sparsity import get_global_sparsity
# from metrics.energy import energy
# import numpy as np


def evaluate_model(model, evaluation_metric, device, test_loader="", model_path =""):
    if evaluation_metric == "TOP1accuracy":
        _ , evaluation_output = top1Accuracy(model=model, test_loader=test_loader, device=device, criterion=None)
        evaluation_output=evaluation_output.item()#np.float(evaluation_output)
        # print("Top1-Accuracy = ", evaluation_output)

    elif evaluation_metric == "mAP":
        pass

    elif evaluation_metric == "Precision":
        pass

    elif evaluation_metric == "Recall":
        pass

    elif evaluation_metric == "F1Score":
        pass

    elif evaluation_metric == "Size":
        evaluation_output = model_size(model_path) 

    elif evaluation_metric == "GPU_usage":
        evaluation_output = gpu_mem_usage() 

    elif evaluation_metric == "CPU_usage":
        evaluation_output, _ = cpu_mem_usage() 

    elif evaluation_metric == "Parameters_count":
        evaluation_output = parameters_count(model)
        # print("Parameters count = ", evaluation_output)

    elif evaluation_metric == "Sparsity":
        _,_,evaluation_output = get_global_sparsity(model)

    elif evaluation_metric == "Latency":
        evaluation_output = inference_latency(model=model, device=device,test_loader=test_loader)
        # print("CPU Inference Latency: {:.2f} ms / sample".format(evaluation_output))

    elif evaluation_metric == "MAC":
        evaluation_output,_ = macs(model=model, device=device,test_loader=test_loader)

    elif evaluation_metric == "FLOPS":
        evaluation_output = flops(model=model, device=device,test_loader=test_loader)

    elif evaluation_metric == "Energy":
        pass 

    elif evaluation_metric == "Power":
        pass 

    else:
        raise Exception("ERROR: An unknown evaluation metric was selected.")
    
    return evaluation_output