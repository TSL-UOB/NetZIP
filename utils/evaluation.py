from metrics.accuracy.topAccuracy import top1Accuracy
from metrics.speed.latency import inference_latency
# from metrics.size import model_size 
# from metrics.energy import energy
import numpy as np
def evaluate_model(model, evaluation_metric, device, test_loader=""):
    if evaluation_metric == "TOP1accuracy":
        _ , evaluation_output = top1Accuracy(model=model, test_loader=test_loader, device=device, criterion=None)
        evaluation_output=evaluation_output.item()#np.float(evaluation_output)
        print("Top1-Accuracy = ", evaluation_output)

    elif evaluation_metric == "mAP":
        pass

    elif evaluation_metric == "Precision":
        pass

    elif evaluation_metric == "Recall":
        pass

    elif evaluation_metric == "F1Score":
        pass

    elif evaluation_metric == "MemorySize":
        pass 

    elif evaluation_metric == "RAMutilisation":
        pass

    elif evaluation_metric == "Latency":
        evaluation_output = inference_latency(model=model, device=device,test_loader=test_loader)
        print("CPU Inference Latency: {:.2f} ms / sample".format(evaluation_output * 1000))

    elif evaluation_metric == "MAC":
        pass

    elif evaluation_metric == "FLOPS":
        pass 

    elif evaluation_metric == "Energy":
        pass 

    elif evaluation_metric == "Power":
        pass 

    else:
        raise Exception("ERROR: An unknown evaluation metric was selected.")
    
    return evaluation_output