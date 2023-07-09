import os
import datetime
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

# plot checkpoints
df = pd.read_csv('results/checkpoints.csv')

fig = plt.figure()

for j in range(2):
	ax = plt.gca()
	axs = [ax, ]#plt.gca().twinx()]
	handles, labels = [], []
	if j > 0: 
		df = pd.read_csv('results/checkpoints_ppo_07.csv')
	best_chkpt = df.iloc[df.idxmax().PT1_mean]
	best_coor = (best_chkpt.checkpoint, best_chkpt.PT1_mean)
	for i, (id, ax) in enumerate(zip(['PT1'], axs)):
		ax.plot(df['checkpoint'], df[f'{id}_mean'], color=f'C{i}', label=id)
		ax.fill_between(
			df['checkpoint'],
			df[f'{id}_mean']-df[f'{id}_std'],
			df[f'{id}_mean']+df[f'{id}_std'],
			alpha=0.3,
			color=f'C{i}'
		)
		ax.annotate(f'global_max : {best_coor}', xy=best_coor, xytext= (best_coor[0]+500,best_coor[1]+.16e6), arrowprops=dict(arrowstyle="->", connectionstyle="angle3,angleA=0,angleB=90"))
		h, l = axs[i].get_legend_handles_labels()
		handles += h
		labels += l

	plt.grid()
	plt.xlim(df['checkpoint'].values[0], df['checkpoint'].values[-1])
	plt.xlabel('Training iteration')
	axs[0].set_ylabel('reward')
	# axs[1].set_ylabel('MM reward')
	plt.tight_layout()
	plt.legend(handles, labels, loc='upper left')
	#plt.savefig(f"images/train_{j}_{datetime.date.today().isoformat()}.png")
	plt.show()
	axs[0].clear()

