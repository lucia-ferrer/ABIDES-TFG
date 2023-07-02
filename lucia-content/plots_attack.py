import os
import pandas as pd
from matplotlib import pyplot as plt


fig = plt.figure()
ax = plt.gca()

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
