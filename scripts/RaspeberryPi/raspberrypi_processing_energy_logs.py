import numpy
import pandas as pd

# ==========================
# === Inputs
# ==========================
# File name of log file
log_file_name_array = ["test_logs/raspberry_pi_energy_pytorch_fp32.txt",
                       "test_logs/raspberry_pi_energy_tensorflow_fp32.txt",
                       "test_logs/raspberry_pi_energy_tflite_fp16.txt",
                       "test_logs/raspberry_pi_energy_tflite_int8.txt"
                      ]    
# Duration in log file for energy calculations                       
time_start = ["06:07:37",
              "02:15:33",
              "16:48:49",
              "00:12:23"]       

time_end   = ["10:30:21",
              "04:03:59",
              "20:01:25",
              "02:10:34"] 

time_increment = 0.1 # Measurment time increments

# === Processing log files data

for i in range(len(log_file_name_array)):

    # Read in log file
    df = pd.read_csv(log_file_name_array[i])

    # Cut data out of duration defined
    experiment_start_index = df.index[df["datetime"].str[-8:].values == time_start[i]][0]
    experiment_end_index   = df.index[df["datetime"].str[-8:].values == time_end[i]][0]
    cropped_df = df.iloc[experiment_start_index:experiment_end_index]
   

    # === Calculate energy, average power, and max power
    I  =   cropped_df["current"].values # array of Currents
    V  =   cropped_df["voltage"].values # array of Volatages

    Power         = I * V 
    Average_power = sum(Power)/(len(Power))
    Max_power     = max(Power)
    Energy        = sum(Power)*time_increment
    Duration      = len(Power)*time_increment

    print("==========================")
    print(log_file_name_array[i])
    print("Average_power = ",Average_power)
    print("Max_power = ",Max_power)
    print("Energy = ",Energy)
    print("Duration = ",Duration)

