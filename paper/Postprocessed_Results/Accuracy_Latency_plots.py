import matplotlib.pyplot as plt
import pandas as pd

# plt.rcParams['text.usetex'] = True

c1 = [2.2, 3.1, 1.3, 3.9, 5.1]
c2 = [3.2, 1.1, 5.3, 2.9, 4.1]
c3 = [4.2, 2.1, 4.3, 1.9, 1.1]

x = [1, 2, 3, 4, 5]
s = [48, 134, 391, 203, 193]

max_ram = 64 # GB

None_df = pd.DataFrame({'Latency':    [14],
                   'Accuracy':        [70.3],
                   'Disk_size':       [44.7],
                   'RAM_usage':       [2.48/max_ram],
                   'cpu_utilisation': [0.496]})

PTQ_df = pd.DataFrame({'Latency':    [7],
                   'Accuracy':        [70.1],
                   'Disk_size':       [11.3],
                   'RAM_usage':       [2.52/max_ram],
                   'cpu_utilisation': [0.491]})

QAT_df = pd.DataFrame({'Latency':    [7],
                   'Accuracy':        [69.6],
                   'Disk_size':       [11.3],
                   'RAM_usage':       [2.51/max_ram],
                   'cpu_utilisation': [0.491]})


GUP_R_df = pd.DataFrame({'Latency':    [15],
                   'Accuracy':        [50.8],
                   'Disk_size':       [44.7],
                   'RAM_usage':       [2.51/max_ram],
                   'cpu_utilisation': [0.496]})

GUP_L1_df = pd.DataFrame({'Latency':    [14],
                   'Accuracy':        [66.6],
                   'Disk_size':       [44.7],
                   'RAM_usage':       [2.46/max_ram],
                   'cpu_utilisation': [0.496]})

# print(None_df.cpu_utilisation[0])
fig, ax = plt.subplots()
ax.scatter(None_df.Latency, None_df.Accuracy, s=None_df.Disk_size, color=(1-None_df.cpu_utilisation[0], 1-None_df.cpu_utilisation[0], 1-None_df.cpu_utilisation[0]), cmap='gray', marker='o', label= "None")
ax.scatter(PTQ_df.Latency, PTQ_df.Accuracy, s=PTQ_df.Disk_size, color=(1-PTQ_df.cpu_utilisation[0], 1-PTQ_df.cpu_utilisation[0], 1-PTQ_df.cpu_utilisation[0]), cmap='gray', marker='*', label= "PTQ")
ax.scatter(QAT_df.Latency, QAT_df.Accuracy, s=QAT_df.Disk_size, color=(1-QAT_df.cpu_utilisation[0], 1-QAT_df.cpu_utilisation[0], 1-QAT_df.cpu_utilisation[0]), cmap='gray', marker='s', label= "QAT")
ax.scatter(GUP_R_df.Latency, GUP_R_df.Accuracy, s=GUP_R_df.Disk_size, color=(1-GUP_R_df.cpu_utilisation[0], 1-GUP_R_df.cpu_utilisation[0], 1-GUP_R_df.cpu_utilisation[0]), cmap='gray', marker='^', label= "$GUP_R$")
ax.scatter(GUP_L1_df.Latency, GUP_L1_df.Accuracy, s=GUP_L1_df.Disk_size, color=(1-GUP_L1_df.cpu_utilisation[0], 1-GUP_L1_df.cpu_utilisation[0], 1-GUP_L1_df.cpu_utilisation[0]), cmap='gray', marker='v', label= "$GUP_{L1}$")

plt.ylabel("mAP (%)")
plt.xlabel("Latency (ms)")
ax.legend()
plt.ylim(0,100)
# plt.xlim(left=0)
plt.savefig("Accurac_Latency.pdf")