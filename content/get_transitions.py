import argparse

import pandas as pd
#import tensorflow as tf
#tf.compat.v1.enable_eager_execution()
from ray.rllib.agents.ppo.ppo import PPOTrainer

from adversarial.agents import AdversarialWrapper
from reinforcement.episodes import evaluate
from reinforcement.models import get_env, load_weights


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--epsilon", type=float, default=.1)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--agent", default=-1)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    # load environment and agent
    env, config = get_env()
    agent = AdversarialWrapper(PPOTrainer)(env='marl-v0', record=True, config=config)
    load_weights(agent)
    ids = config['env_config']['learning_agent_ids'] if args.agent == -1 else [args.agent]

    # extract normal transitions
    for id in ids: 
        print(f"Extracting transitions for {id}...")
        print()
        agent.epsilon_greedy = {id: args.epsilon}
        evaluate(env, agent, config, num_trials=args.episodes, verbose=True)
        pd.DataFrame(agent.transitions[id]).to_csv(f'data/transitions_{id}.csv', index=False)