import pandas as pd
import matplotlib.pyplot as plt

# df = pd.read_csv('YOLOv5s_COCO_laptop.csv')
df = pd.read_csv('YOLOv5s_COCO_raspberrypi.csv')

print(df) 
print(df.columns)

metrics = ["F1 Score", "Latency", "Size", "Energy", "Power"]
compression_techniques = ["FP32 - PyTorch", "FP32 - TF", "FP16 - TFlite", "INT8 - TFlite"]
for metric in metrics:
    plt.clf()
    fig = plt.figure()
    plt.rcParams.update({'font.size': 16})
    ax = fig.add_axes([0.12,0.1,0.84,0.85])
    # idx = np.where(np.array(logs_class.evaluationMetric_array) == metric)[0]
    # compression_techniques = np.array(logs_class.compressionTechnique_array)[idx]
    if metric == "F1 Score":
        values = df["F1 Score"].values
        plt.ylabel("F1 Score")

    elif metric == "Latency":
        values = df["Inference Latency (ms)"].values
        plt.ylabel("Latency (ms)")

    elif metric == "Size":
        values = df["Model Size (MB)"].values
        plt.ylabel("Disk Size (MB)")

    elif metric == "Energy":
        values = df["Total Energy (kJ)"].values
        plt.ylabel("Energy (kJ)")

    elif metric == "Power":
        values = df["Average Power(W)"].values
        plt.ylabel("Power (W)")

    ax.bar(compression_techniques,values, color = 'k', width = 0.25)
    # if metric == "MAC" or metric == "FLOPS" or metric == "CPU_usage":
    #     ax.set_yscale('log')
    #     if max(values) == 0 or max(values) == float('Inf') or max(values) == float('NaN'):
    #         pass #ax.set_ylim([1, None])
    #     else:
    #         ax.set_ylim([1, max(values)*2])
    
    # else:

    ax.set_yscale('linear')
    if max(values) == 0 or max(values) == float('Inf') or max(values) == float('NaN'):
        pass #ax.set_ylim([0, None])
    else:
        ax.set_ylim([0, None])#max(values)*1.1])

    # save plot to logs folder
    # fig.tight_layout()
    plt.savefig(metric+".png")
