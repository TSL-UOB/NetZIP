import numpy as np

import matplotlib.pyplot as plt
import pandas as pd
from math import pi



max_ram = 64 # GB

None_df = pd.DataFrame({'Latency':    [25],
                   'Accuracy':        [90],
                   'Disk_size':       [100],
                   'RAM_usage':       [50/max_ram],
                   'cpu_utilisation': [0.7]})

PTQ_df = pd.DataFrame({'Latency':    [15],
                   'Accuracy':        [70],
                   'Disk_size':       [25],
                   'RAM_usage':       [50/max_ram],
                   'cpu_utilisation': [0.1]})

QAT_df = pd.DataFrame({'Latency':    [15],
                   'Accuracy':        [90],
                   'Disk_size':       [25],
                   'RAM_usage':       [50/max_ram],
                   'cpu_utilisation': [0.1]})


GUP_R_df = pd.DataFrame({'Latency':    [25],
                   'Accuracy':        [70],
                   'Disk_size':       [100],
                   'RAM_usage':       [50/max_ram],
                   'cpu_utilisation': [0.2]})

GUP_L1_df = pd.DataFrame({'Latency':    [25],
                   'Accuracy':        [85],
                   'Disk_size':       [100],
                   'RAM_usage':       [50/max_ram],
                   'cpu_utilisation': [0.5]})

# Set data
df = pd.DataFrame({
'group': ['None PyTorch (FP32)','None TF (FP32)','PTQ TFlite (FP16)','PTQ TFlite (INT8)'],
'Overall Compression Success': [38, 1.5, 30, 4],
'Performance Ratio': [29, 10, 9, 34],
'Speedup Ratio': [8, 39, 23, 24],
'Compression Ratio': [7, 31, 33, 14],
'Efficiency Ratio': [28, 15, 32, 14]
})
 
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
plt.yticks([10,20,30], ["10","20","30"], color="grey", size=7)
plt.ylim(0,40)
 

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
plt.savefig("temp.pdf")

# def radar_factory(num_vars, frame='circle'):
#     """
#     Create a radar chart with `num_vars` axes.

#     This function creates a RadarAxes projection and registers it.

#     Parameters
#     ----------
#     num_vars : int
#         Number of variables for radar chart.
#     frame : {'circle', 'polygon'}
#         Shape of frame surrounding axes.

#     """
#     # calculate evenly-spaced axis angles
#     theta = np.linspace(0, 2*np.pi, num_vars, endpoint=False)

#     class RadarTransform(PolarAxes.PolarTransform):

#         def transform_path_non_affine(self, path):
#             # Paths with non-unit interpolation steps correspond to gridlines,
#             # in which case we force interpolation (to defeat PolarTransform's
#             # autoconversion to circular arcs).
#             if path._interpolation_steps > 1:
#                 path = path.interpolated(num_vars)
#             return Path(self.transform(path.vertices), path.codes)

#     class RadarAxes(PolarAxes):

#         name = 'radar'
#         PolarTransform = RadarTransform

#         def __init__(self, *args, **kwargs):
#             super().__init__(*args, **kwargs)
#             # rotate plot such that the first axis is at the top
#             self.set_theta_zero_location('N')

#         def fill(self, *args, closed=True, **kwargs):
#             """Override fill so that line is closed by default"""
#             return super().fill(closed=closed, *args, **kwargs)

#         def plot(self, *args, **kwargs):
#             """Override plot so that line is closed by default"""
#             lines = super().plot(*args, **kwargs)
#             for line in lines:
#                 self._close_line(line)

#         def _close_line(self, line):
#             x, y = line.get_data()
#             # FIXME: markers at x[0], y[0] get doubled-up
#             if x[0] != x[-1]:
#                 x = np.append(x, x[0])
#                 y = np.append(y, y[0])
#                 line.set_data(x, y)

#         def set_varlabels(self, labels):
#             self.set_thetagrids(np.degrees(theta), labels)

#         def _gen_axes_patch(self):
#             # The Axes patch must be centered at (0.5, 0.5) and of radius 0.5
#             # in axes coordinates.
#             if frame == 'circle':
#                 return Circle((0.5, 0.5), 0.5)
#             elif frame == 'polygon':
#                 return RegularPolygon((0.5, 0.5), num_vars,
#                                       radius=.5, edgecolor="k")
#             else:
#                 raise ValueError("Unknown value for 'frame': %s" % frame)

#         def _gen_axes_spines(self):
#             if frame == 'circle':
#                 return super()._gen_axes_spines()
#             elif frame == 'polygon':
#                 # spine_type must be 'left'/'right'/'top'/'bottom'/'circle'.
#                 spine = Spine(axes=self,
#                               spine_type='circle',
#                               path=Path.unit_regular_polygon(num_vars))
#                 # unit_regular_polygon gives a polygon of radius 1 centered at
#                 # (0, 0) but we want a polygon of radius 0.5 centered at (0.5,
#                 # 0.5) in axes coordinates.
#                 spine.set_transform(Affine2D().scale(.5).translate(.5, .5)
#                                     + self.transAxes)
#                 return {'polar': spine}
#             else:
#                 raise ValueError("Unknown value for 'frame': %s" % frame)

#     register_projection(RadarAxes)
#     return theta


# def example_data():

#     data = [
#         ['Overall Compression Success', 'Nitrate', 'EC', 'OC1', 'OC2', 'OC3', 'OP', 'CO', 'O3'],
#         ('Basecase', [
#             [0.88, 0.01, 0.03, 0.03, 0.00, 0.06, 0.01, 0.00, 0.00],
#             [0.07, 0.95, 0.04, 0.05, 0.00, 0.02, 0.01, 0.00, 0.00],
#             [0.01, 0.02, 0.85, 0.19, 0.05, 0.10, 0.00, 0.00, 0.00],
#             [0.02, 0.01, 0.07, 0.01, 0.21, 0.12, 0.98, 0.00, 0.00],
#             [0.01, 0.01, 0.02, 0.71, 0.74, 0.70, 0.00, 0.00, 0.00]])
        
#     ]
#     return data


# if __name__ == '__main__':
#     N = 9
#     theta = radar_factory(N, frame='polygon')

#     data = example_data()
#     spoke_labels = data.pop(0)

#     fig, axs = plt.subplots(figsize=(9, 9), nrows=1, ncols=1,
#                             subplot_kw=dict(projection='radar'))
#     fig.subplots_adjust(wspace=0.25, hspace=0.20, top=0.85, bottom=0.05)

#     colors = ['b', 'r', 'g', 'm', 'y']
#     # Plot the four cases from the example data on separate axes
#     for ax, (title, case_data) in axs, data:
#         ax.set_rgrids([0.2, 0.4, 0.6, 0.8])
#         ax.set_title(title, weight='bold', size='medium', position=(0.5, 1.1),
#                      horizontalalignment='center', verticalalignment='center')
#         for d, color in zip(case_data, colors):
#             ax.plot(theta, d, color=color)
#             ax.fill(theta, d, facecolor=color, alpha=0.25, label='_nolegend_')
#         ax.set_varlabels(spoke_labels)

#     # add legend relative to top-left plot
#     labels = ('None PyTorch (FP32)', 'None TF (FP32)', 'PTQ TFlite (FP16)', 'PTQ TFlite (INT8)', 'Factor 5')
#     legend = axs[0, 0].legend(labels, loc=(0.9, .95),
#                               labelspacing=0.1, fontsize='small')

#     fig.text(0.5, 0.965, '5-Factor Solution Profiles Across Four Scenarios',
#              horizontalalignment='center', color='black', weight='bold',
#              size='large')

    # plt.savefig("temp.pdf")