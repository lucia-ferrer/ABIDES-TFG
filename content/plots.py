import os
import pandas as pd
from matplotlib import pyplot as plt

# plot checkpoints
df = pd.read_csv('results/checkpoints.csv')

fig = plt.figure()
ax = plt.gca()
axs = [ax, ]#plt.gca().twinx()]
handles, labels = [], []
for i, (id, ax) in enumerate(zip(['PT1'], axs)):
	ax.plot(df['checkpoint'], df[f'{id}_mean'], color=f'C{i}', label=id)
	ax.fill_between(
		df['checkpoint'],
		df[f'{id}_mean']-df[f'{id}_std'],
		df[f'{id}_mean']+df[f'{id}_std'],
		alpha=0.3,
		color=f'C{i}'
	)
	h, l = axs[i].get_legend_handles_labels()
	handles += h
	labels += l

""" plt.grid()
plt.xlim(df['checkpoint'].values[0], df['checkpoint'].values[-1])
plt.xlabel('Training iteration')
axs[0].set_ylabel('reward')
# axs[1].set_ylabel('MM reward')
plt.tight_layout()
# plt.legend(handles, labels, loc='upper left')
#plt.show()
plt.savefig("images/train.png") """

# plot attacks
x_iterator = {
 	'UniformAttack': 'freq',
 	'STAttack': 'beta',
 	'VFAttack': 'beta'
 }
for i, attack in enumerate(x_iterator):
	path = f'results/attacks/{attack}_PT1.csv'
	if os.path.exists(path):
		df_attack = pd.read_csv(path)
		fig = plt.figure()
	for e, df in iter(df_attack.groupby('epsilon')):
		plt.plot(df[x_iterator[attack]], df['reward_mean'], label=e)
		plt.fill_between(
				df[x_iterator[attack]],
			df['reward_mean']-df['reward_std'],
				df['reward_mean']+df['reward_std'],
				alpha=0.3,
				color=f"C{i}"
			)
		plt.grid()
		plt.xlim(df[x_iterator[attack]].values[0], df[x_iterator[attack]].values[-1])
		ax.set_xticks(df_attack[x_iterator[attack]])
		plt.ylabel('reward')
		plt.xlabel(x_iterator[attack])
	# plt.tight_layout()
		plt.legend()
		plt.title(attack)
		plt.show()
