# import sys
# sys.path.append("..") # Adds parent directory to python modules path.

from shrinkbench.experiment import PruningExperiment

import os
os.environ['DATAPATH'] = 'netzip_datasets'#'shrinkbench/datasets'#'netzip_datasets/data/'#'../netzip/datasets/data/'

from IPython.display import clear_output
clear_output()

for strategy in ['RandomPruning', 'GlobalMagWeight', 'LayerMagWeight']:
	for  c in [1,2,4,8,16,32,64]:
		exp = PruningExperiment(dataset='MNIST', 
								model='MnistNet',
								strategy=strategy,
								compression=c,
								train_kwargs={'epochs':10},
								pretrained=False)
		exp.run()
		clear_output()


from shrinkbench.plot import df_from_results, plot_df
df = df_from_results('results')

plot_df("temp1.png", df, 'compression', 'pre_acc5', markers='strategy', line='--', colors='strategy', suffix=' - pre')
plot_df("temp2.png", df, 'compression', 'post_acc5', markers='strategy', fig=False, colors='strategy')

plot_df("temp3.png", df, 'speedup', 'post_acc5', colors='strategy', markers='strategy')
# plt.yscale('log')
# plt.ylim(0.996,0.9995)
# plt.xticks(2**np.arange(7))
# plt.gca().set_xticklabels(map(str, 2**np.arange(7)))
# None

df['compression_err'] = (df['real_compression'] - df['compression'])/df['compression']

plot_df("temp3.png",df, 'compression', 'compression_err', colors='strategy', markers='strategy')