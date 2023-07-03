import argparse
import datetime
import pandas as pd
import matplotlib.pyplot as plt
#import tensorflow as tf
#tf.compat.v1.enable_eager_execution()
from ray.rllib.agents.ppo.ppo import PPOTrainer

from config.defaults import NUM_TRIALS
from config.attacks import ATTACK_CLASS, get_agent_attack_config
from config.utils import grid_generator, Logger
from adversarial.utils import test
from reinforcement.models import get_env, load_weights

from config.my_experiments import *
from adversarial.lucia.my_agent import AdversarialWrapper
from adversarial.lucia.my_defense import available_norms, Defense
""" 
from config.experiments import *
from adversarial.agents import AdversarialWrapper
from adversarial.defense import available_norms, Defense
"""

def parse_args():
	parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
	parser.add_argument("--agent", default=-1,
						help="The agent to use. -1 for all (Default)"
	)
	parser.add_argument("--norm", choices=available_norms, default=available_norms[0],
						help=f"The normalization to use (default {available_norms[0]}"
	)
	parser.add_argument("--val_episodes", type=int, default=NUM_TRIALS,
						help=f"Number of episodes to validate the agent with (default {NUM_TRIALS})"
	)
	parser.add_argument("--detector", default=-1,
						help="The name of the detector to run. -1 for all (Default)"
	)
	parser.add_argument("--detector_parameter", type=int, default=-1,
						help="Parameter ids to try for the detector. -1 for all (Default)"
	)
	parser.add_argument("--recovery", default=-1,
						help="The name of the recovery to run. -1 for all (Default)"
						)
	parser.add_argument("--recovery_parameter", type=int, default=-1,
						help="Parameter ids to try for the recovery. -1 for all (Default)"
						)
	parser.add_argument("--attack", default=-1,
						help="Which attack to use. -1 for all (Default)"
	)
	parser.add_argument("--attack_parameter", type=int, default=-1,
						help="Parameter ids to try for the attacks. -1 for all (Default)"
	)
	parser.add_argument("--plotting", type=bool, default=False, 
		     			help="Parameter to indicate whether we keep track of each trial in a plot, \
							instead of storing the tests in a csv"
	)
	return parser.parse_args()


def get_agent():
	agent = AdversarialWrapper(PPOTrainer)(env='marl-v0', config=config)
	load_weights(agent)
	agent.attacker = {policy_id: ATTACK_CLASS[attack_name](**attack_params)}
	agent.defender = {policy_id: defenses[policy_id]}
	agent._plot = True
	return agent


if __name__ == '__main__':
	args = parse_args()
	env, config = get_env()
	ids = config['env_config']['learning_agent_ids'] if args.agent == -1 else [args.agent]
	transitions = {id: pd.read_csv(f'data/transitions_{id}.csv', header=None).values for id in ids}

	# list of detectors to do
	detectors_list = []
	iterator = detector_experiments.items() if args.detector == -1\
		else ((args.detector, detector_experiments[args.detector],),)
	for detector_name, exp_config in iterator:
		for params in grid_generator(exp_config, args.detector_parameter):
			detectors_list.append((detector_name, params))

	# list of recoveries to do
	recovery_list = []
	iterator = recovery_experiments.items() if args.recovery == -1\
		else (([args.recovery, 'None'], [recovery_experiments[args.recovery],recovery_experiments['None']],),)
	for recovery_name, exp_config in iterator:
		for params in grid_generator(exp_config, args.recovery_parameter):
			recovery_list.append((recovery_name, params))

	# list of attacks to do
	attacks_list = []
	for id in ids[1:]:
		for a_name in ATTACK_CLASS if args.attack == -1 else [args.attack, 'Empty']:
			for params in grid_generator(get_agent_attack_config(id)[a_name], args.attack_parameter):
				attacks_list.append((id, a_name, params))


	# file name
	file_name = f"img_recovery_test_"
	if args.agent != -1:				# agent
		file_name += f"_{args.agent}"
	if args.detector != -1:				# detector
		file_name += f"_{args.detector}"
	if args.detector_parameter != -1:	   # detector params
		file_name += f"_detparam{args.detector_parameter}"
	if args.recovery != -1:				 # recovery
		file_name += f"_{args.recovery}"
	if args.recovery_parameter != -1:	   # recovery params
		file_name += f"_recparam{args.recovery_parameter}"
	if args.attack != -1:				  # attack
		file_name += f"_{args.attack}"
	if args.attack_parameter != -1:		 # attack params
		file_name += f"_atckparam{args.attack_parameter}"
	
	logger = Logger(file_name, len(detectors_list)*len(recovery_list)*len(attacks_list))

	# execute experiments
	results = pd.DataFrame()
	fig, ax = plt.subplots()
	
	done_attack_empty = False

	for detector_name, detector_params in detectors_list:
		defenses = {policy_id: Defense(norm=args.norm, detector=DETECTOR_CLASS[detector_name](**detector_params))
					for policy_id in ids[1:]}
				
		[defense.fit(transitions[policy_id]) for policy_id, defense in defenses.items()]

		for recovery_name, recovery_params in recovery_list:
			# print(f"Recovery : {recovery_name} \t Params : {recovery_params}")
			
			for policy_id, defense in defenses.items():
				state_dims = np.prod(env.observation_space[policy_id].shape)
				defense.recovery = RECOVERY_CLASS[recovery_name](**recovery_params, state_dims=state_dims) if recovery_name != 'None' else None
				defense.fit_recovery()

			for policy_id, attack_name, attack_params in attacks_list:
					if attack_name == 'Empty' : 
						if done_attack_empty: continue 
						else: 
							done_attack_empty = True
							empty_attack_col = f"{policy_id}:{attack_name}:{detector_name}:{recovery_name}:{params_to_str(recovery_params)}"
					if recovery_name == 'None': 
						empty_recovery_col = f"{policy_id}:{attack_name}:{detector_name}:{recovery_name}:{params_to_str(recovery_params)}"

					agent = get_agent()
					row = {
						'norm': args.norm,
						'detector': detector_name,
						'detector_params': params_to_str(detector_params),
						'recovery': recovery_name,
						'recovery_params': params_to_str(recovery_params),
						'attack': attack_name,
						'attack_params': params_to_str(attack_params),
						'attacked_policy': policy_id,
						**test(env, agent, config, args.val_episodes, policy_id, test=False)
					}
					logger()
					#results.append(row)
					results[f"{policy_id}:{attack_name}:{detector_name}:{recovery_name}:{params_to_str(recovery_params)}"] = agent.last_rewards
					
	results.to_csv(f"images/recovery/epsidoes_{file_name}_{datetime.date.today().isoformat()}.csv", index=False)
	for unitest in results.columns(): 
		if unitest in (empty_attack_col, empty_recovery_col): continue
		id, attack, detector, recover, r_params = unitest.split(':', 5)
		results[unitest].plot(ax=ax, label=f"recovered")
		results[empty_attack_col].plot(ax=ax, label=f"original")
		results[empty_recovery_col].plot(ax=ax, label="attacked")
		ax.legend()
		ax.set_title(unitest)	
		ax.set_ylabel('rewards')	
		ax.set_xlabel('steps')
		plt.save_fig(f"images/recovery/plot_{unitest}.png")
		plt.show()



