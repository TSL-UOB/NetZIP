import time
import os
import numpy as np
import matplotlib.pyplot as plt

class log():
    def __init__(self):
        self.modelName_array               = []
        self.datasetName_array             = []
        self.evaluationMetric_array        = []
        self.compressionTechnique_array    = []
        self.value_array                   = []

    def append(self,modelName,datasetName,evaluationMetric,compressionTechnique,value):

        self.modelName_array.append(modelName)
        self.datasetName_array.append(datasetName)
        self.evaluationMetric_array.append(evaluationMetric)
        self.compressionTechnique_array.append(compressionTechnique)
        self.value_array.append(value)


    def write_file(self, output_folder="output_results", results_folder="", file_name=""):
        results_folder = time.strftime("%Y%m%d-%H%M%S")
        file_name      = results_folder + ".txt"
        self.output_results_folder = output_folder+"/"+results_folder
        # Folder "results" if not already there
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        if not os.path.exists(self.output_results_folder):
            os.makedirs(self.output_results_folder)

        self.file_path = os.path.join(self.output_results_folder, file_name)
        with open(self.file_path, 'w') as log_file: 
            log_file.write('modelName ,datasetName, evaluationMetric, compressionTechnique, value_array\n')
            for i in range(len(self.modelName_array)):
                log_file.write('%s, %s, %s, %s, %3.3f \n' %\
                    (self.modelName_array[i],self.datasetName_array[i], self.evaluationMetric_array[i], self.compressionTechnique_array[i], self.value_array[i]))
        print('Log file SUCCESSFULLY generated and saved to '+ self.file_path)

def plot_results(logs_class):
    metrics = np.unique(logs_class.evaluationMetric_array)
    for metric in metrics:
        plt.clf()
        fig = plt.figure()
        ax = fig.add_axes([0.1,0.1,0.8,0.8])
        idx = np.where(np.array(logs_class.evaluationMetric_array) == metric)[0]
        compression_techniques = np.array(logs_class.compressionTechnique_array)[idx]
        values = np.array(logs_class.value_array)[idx]
        ax.bar(compression_techniques,values, color = 'k', width = 0.25)
        if metric == "MAC" or metric == "FLOPS" or metric == "CPU_usage":
            ax.set_yscale('log')
        else:
            ax.set_yscale('linear')
        # save plot to logs folder
        fig.tight_layout()
        plt.savefig(logs_class.output_results_folder+"/"+metric+".png")
