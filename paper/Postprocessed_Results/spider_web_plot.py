import numpy as np

import matplotlib.pyplot as plt
import pandas as pd
from math import pi


# PC - Resnet18 - CIFAR10
df = pd.DataFrame({
'group': ['PTQ','QAT','GUP_R', 'GUP_L1'],
'Overall Compression Success': [4.7,4.7,0,0.2],
'Performance Ratio':           [1,1.0,0.73,0.94],
'Speedup Ratio':               [4,4,1,1],
'Compression Ratio':           [1.0,1.0,1,1],
'Efficiency Ratio':            [2.75,2.75,1.2,1.3]
})

# # Raspebrry pi - YOLO - COCO
# df = pd.DataFrame({
# 'group': ['None TF (FP32)','PTQ TFlite (FP16)','PTQ TFlite (INT8)'],
# 'Overall Compression Success': [0.76,3.52,6.8],
# 'Performance Ratio':           [1.0,1.0,1.0],
# 'Speedup Ratio':               [1,2,4],
# 'Compression Ratio':           [0.8,3.0,3.0],
# 'Efficiency Ratio':            [1.9,1.5,2.5]
# })
 
# ------- PART 1: Create background
 
# number of variable
categories=list(df)[1:]
N = len(categories)
 
# What will be the angle of each axis in the plot? (we divide the plot / number of variable)
angles = [n / float(N) * 2 * pi for n in range(N)]
angles += angles[:1]
 
# Initialise the spider plot
ax = plt.subplot(111, polar=True)
 
# If you want the first axis to be on top:
ax.set_theta_offset(pi / 2)
ax.set_theta_direction(-1)
 
# Draw one axe per variable + add labels
plt.xticks(angles[:-1], categories)
 
# Draw ylabels
ax.set_rlabel_position(0)
plt.yticks([2,4,6], ["2","4","6"], color="grey", size=7)
plt.ylim(0,6)
 

# ------- PART 2: Add plots
 
# Plot each individual = each line of the data
# I don't make a loop, because plotting more than 3 groups makes the chart unreadable
 
# Ind1
values=df.loc[0].drop('group').values.flatten().tolist()
values += values[:1]
ax.plot(angles, values, linewidth=1, linestyle='solid', label=df.group[0])
ax.fill(angles, values, 'b', alpha=0.1)
 
# Ind2
values=df.loc[1].drop('group').values.flatten().tolist()
values += values[:1]
ax.plot(angles, values, linewidth=1, linestyle='solid', label=df.group[1])
ax.fill(angles, values, 'r', alpha=0.1)

# Ind3
values=df.loc[2].drop('group').values.flatten().tolist()
values += values[:1]
ax.plot(angles, values, linewidth=1, linestyle='solid', label=df.group[2])
ax.fill(angles, values, 'g', alpha=0.1)

# Ind4
values=df.loc[3].drop('group').values.flatten().tolist()
values += values[:1]
ax.plot(angles, values, linewidth=1, linestyle='solid', label=df.group[3])
ax.fill(angles, values, 'y', alpha=0.1)
 
# Add legend
plt.legend(loc=(-0.3,0.8))#'upper left')#, bbox_to_anchor=(0.1, 0.1))

# Show the graph
# plt.savefig("spider_raspebrrypi_yolo.pdf")
plt.savefig("spider_pc_resnet18.pdf")
