import time
import os

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


    def write_file(self, output_folder="output_results", file_name=time.strftime("%Y%m%d-%H%M%S")+".txt"):
        # Folder "results" if not already there
        # output_folder = "tests_logs"
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        file_path = os.path.join(output_folder, file_name)
        with open(file_path, 'w') as log_file: 
            log_file.write('modelName ,datasetName, evaluationMetric, compressionTechnique, value_array\n')
            for i in range(len(self.modelName_array)):
                log_file.write('%s, %s, %s, %s, %3.3f \n' %\
                    (self.modelName_array[i],self.datasetName_array[i], self.evaluationMetric_array[i], self.compressionTechnique_array[i], self.value_array[i]))
        print('Log file SUCCESSFULLY generated and saved to '+output_folder+"/"+file_name)
