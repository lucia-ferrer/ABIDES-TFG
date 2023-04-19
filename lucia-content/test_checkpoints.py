import os

import numpy as np
import pandas as pd
from ray.rllib.agents.ppo.ppo import PPOTrainer

from config.defaults import NUM_TRIALS
from reinforcement.episodes import evaluate
from reinforcement.models import get_env, load_weights


def eval_checkpoint(env, agent, ids, config, loaded_checkpoint, results):
    print(f"Testing checkponit {loaded_checkpoint}")
    r = np.asarray(list(evaluate(env, agent, config, num_trials=NUM_TRIALS, verbose=True).values()))
    means = r.mean(axis=1)
    stds = r.std(axis=1)
    row = {'checkpoint': loaded_checkpoint}
    for id, mean, std in zip(ids, means, stds):
        row.update({f'{id}_mean': mean, f'{id}_std': std})
    return results.append(row, ignore_index=True)

if __name__ == '__main__':
    env, config = get_env()
    agent = PPOTrainer(env='marl-v0', config=config)
    ids = config['env_config']['learning_agent_ids']

    path = 'results/checkpoints.csv'
    if os.path.exists(path):
        results = pd.read_csv(path)
    else:
        results = pd.DataFrame()
        results = eval_checkpoint(env, agent, ids, config, 0, results)

    loaded_checkpoint = True
    i = len(results)-1
    while loaded_checkpoint:
        loaded_checkpoint = load_weights(agent, i)
        i += 1
        if loaded_checkpoint:
            results = eval_checkpoint(env, agent, ids, config, loaded_checkpoint, results)
            results.to_csv(path, index=False)
