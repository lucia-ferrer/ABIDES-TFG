import os
import argparse

import numpy as np
import pandas as pd
#import tensorflow as tf
#tf.compat.v1.enable_eager_execution()
from ray.rllib.agents.ppo.ppo import PPOTrainer

from adversarial.agents import AdversarialWrapper
from config.defaults import NUM_TRIALS
from config.attacks import ATTACK_CLASS, get_agent_attack_config
from config.utils import grid_generator
from reinforcement.episodes import evaluate
from reinforcement.models import get_env, load_weights



def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--agent", default=-1,
                        help="The agent to use. -1 for all (Default)"
    )
    parser.add_argument("--val_episodes", default=NUM_TRIALS,
                        help=f"Number of episodes to validate the agent with (default {NUM_TRIALS})")
    parser.add_argument("--attack", default=-1,
                        help="Which attack to use. -1 for all (Default)"
    )
    return parser.parse_args()


def get_agent():
    agent = AdversarialWrapper(PPOTrainer)(env='marl-v0', config=config)
    load_weights(agent)
    agent.attacker = attack
    return agent


def test_attack(path, attack_cls, attacked_policy, trials):
    results = pd.read_csv(path) if os.path.exists(path) else pd.DataFrame()
    i = 0
    #print(attacked_policy, attack_cls.__name__)
    for params in grid_generator(get_agent_attack_config(attacked_policy)[attack_cls.__name__]):
        print(f"Attacking {attacked_policy} with {attack_cls.__name__} and params {params}")
        i += 1
        if i > len(results):

            global attack
            attack = {attacked_policy: attack_cls(**params)}
            r = evaluate(env, get_agent, config, trials, verbose=True)[0][attacked_policy]
            results = results.append({
                'agent': attacked_policy,
                **params,
                'reward_mean': np.mean(r),
                'reward_std': np.std(r)
            }, ignore_index=True)
            results.to_csv(path, index=False)
        print(results)


if __name__ == '__main__':
    args = parse_args()
    env, config = get_env()
    ids = config['env_config']['learning_agent_ids'] if args.agent == -1 else [args.agent]
    attack = None

    for policy_id in ids:
        for name, cls in ATTACK_CLASS.items() if args.attack == -1 else ((args.attack, ATTACK_CLASS[args.attack],),):
            #TODO: verify if the empty attack is neccesary
            print(cls)
            test_attack(
                path=f'results/attacks/{name}_{policy_id}.csv',
                attack_cls=cls,
                attacked_policy=policy_id,
                trials=args.val_episodes
            )
