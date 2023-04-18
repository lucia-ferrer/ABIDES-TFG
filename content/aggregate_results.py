import pandas as pd
from collections import defaultdict
import os
import argparse


def parse_args():
	parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
	parser.add_argument("--folder", default='results')
	return parser.parse_args()


if __name__ == '__main__':
	args = parse_args()

	all = []
	normal = [[], []]
	attacked = defaultdict(lambda: [[], []])
	path = f"{args.folder}/recovery/recovery_tests_z-score.csv"
	if not os.path.exists(path): print(f"{path} not found")

	res = pd.read_csv(path)
	# idx = res[(res['detector'] == 'KMeansProb') & (res['detector params'].map(lambda x: 'True' in str(x)))].index
	# res.loc[idx, 'detector'] = 'KMeansProbLocal'
	# idx = res[(res['detector'] == 'KMeansProb') & (res['detector params'].map(lambda x: 'False' in str(x)))].index
	# res.loc[idx, 'detector'] = 'KMeansProbGlobal'
	normal[0].append(res['normal_reward'].mean())
	normal[1].append(res['normal_reward_std'].mean())

	for a, v in res.groupby('attack')['attacked_reward'].mean().iteritems():
		attacked[a][0].append(v)
	for a, v in res.groupby('attack')['attacked_reward_std'].mean().iteritems():
		attacked[a][1].append(v)

	idxmax = res.groupby(['detector', 'attack', 'norm'])['recovered_attack_reward'].idxmax().values
	res.iloc[idxmax].to_csv(f"{args.folder}/best_recovery_tests.csv", index=False)
	res = res.iloc[idxmax][['detector', 'attack', 'norm', 'recovered_attack_reward', 'recovered_attack_reward_std', 'balanced_accuracy']]
	all.append(res.set_index(['detector', 'attack', 'norm']))
	if len(all) > 0:
		df = pd.concat(all, axis=1)
		columns = []
	columns.append(f"(mean)")
	columns.append(f"(std)")
	columns.append(f"(acc)")
	df.columns = columns
	df_r = df[[c for c in columns if 'mean' in c]+[c for c in columns if 'std' in c]+[c for c in columns if 'accuracy' in c]]
	df_r.loc[('normal_reward', '', ''), :] = normal[0] + normal[1]
	for attack, v in attacked.items():
		df_r.loc[('attacked_reward', attack, ''), :] = v[0]+v[1]
	df_r.to_csv(f"{args.folder}/recovery_tests.csv")
