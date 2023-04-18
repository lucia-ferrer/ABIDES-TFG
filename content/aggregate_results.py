import pandas as pd
from collections import defaultdict
from config.defaults import ENVS
import os
import argparse


def parse_args():
	parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
	parser.add_argument("--folder", default='results')
	return parser.parse_args()


if __name__ == '__main__':
	args = parse_args()
	all = []
	found_envs = []
	for env in ENVS:
		path = f"{args.folder}/{env}/detector_tests.csv"
		if not os.path.exists(path):
			print(f"{path} not found")
			continue
		else:
			found_envs.append(env)
		res = pd.read_csv(path)
		# idx = res[(res['detector'] == 'KMeansProb') & (res['detector params'].map(lambda x: 'True' in str(x)))].index
		# res.loc[idx, 'detector'] = 'KMeansProbLocal'
		# idx = res[(res['detector'] == 'KMeansProb') & (res['detector params'].map(lambda x: 'False' in str(x)))].index
		# res.loc[idx, 'detector'] = 'KMeansProbGlobal'

		res_g = res.groupby(['detector', 'attack', 'norm'])['balanced_accuracy']

		res.iloc[res_g.idxmax().values].to_csv(f"{args.folder}/{env}/best_detector_tests.csv", index=False)
		all.append(res_g.max())
	if len(all) > 0:
		df = pd.concat(all, axis=1)
		df.columns = found_envs
		df['mean'] = df.mean(axis=1)
		df.to_csv(f"{args.folder}/detector_tests.csv")

	all = []
	normal = [[], []]
	attacked = defaultdict(lambda: [[], []])
	found_envs = []
	for env in ENVS:
		path = f"{args.folder}/{env}/recovery_tests.csv"
		if not os.path.exists(path):
			print(f"{path} not found")
			continue
		else:
			found_envs.append(env)
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
		res.iloc[idxmax].to_csv(f"{args.folder}/{env}/best_recovery_tests.csv", index=False)
		res = res.iloc[idxmax][['detector', 'attack', 'norm', 'recovered_attack_reward', 'recovered_attack_reward_std', 'balanced_accuracy']]
		all.append(res.set_index(['detector', 'attack', 'norm']))

	if len(all) > 0:
		df = pd.concat(all, axis=1)
		columns = []
		for env in found_envs:
			columns.append(f"{env} (mean)")
			columns.append(f"{env} (std)")
			columns.append(f"{env} (acc)")
		df.columns = columns
		df_r = df[[c for c in columns if 'mean' in c]+[c for c in columns if 'std' in c]+[c for c in columns if 'accuracy' in c]]

		df_r.loc[('normal_reward', '', ''), :] = normal[0] + normal[1]
		for attack, v in attacked.items():
			df_r.loc[('attacked_reward', attack, ''), :] = v[0]+v[1]
		df_r.to_csv(f"{args.folder}/recovery_tests.csv")

