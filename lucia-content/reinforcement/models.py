import os
import re

import pandas as pd
import gym
from gym.envs.registration import register
from abides_gym.envs.marl_environment_v0 import SubGymMultiAgentRLEnv_v0
import ray.cloudpickle as cloudpickle

from config.defaults import root


def get_env():
    register(
        id="marl-v0",
        entry_point=SubGymMultiAgentRLEnv_v0,
    )
    with open(os.path.join(root, 'params.pkl'), "rb") as f:
        config = cloudpickle.load(f)
    config["create_env_on_driver"] = True
    config['env_config']['background_config'] = "rmsc04"
    config['env_config']['log_flag'] = False
    # config['framework'] = 'tf2'
    # Initialize the environment
    env = gym.make('marl-v0', **config['env_config'])
    return env, config


def load_weights(agent, checkpoint_idx='max', max_agent='PT1'):
    """ Input : agente, idx -> (o max o posicion del checkpoint dentro de la listaa ordenada'
        Output : Number del checkpoint seleccionado ya sea por index o por el max (resultado de mean max)
        THe function moreover loades in the agent the weights of that checkpoint
    """
    checkpoints = sorted([f for f in os.listdir(root) if 'checkpoint' in f])
    checkpoints_results_path = 'results/checkpoints_06.csv'
    print(checkpoint_idx, checkpoints_results_path)    
    if checkpoint_idx == 'max' and os.path.exists(checkpoints_results_path):
        df = pd.read_csv(checkpoints_results_path)
        checkpoint_idx = df.sort_values(f'{max_agent}_mean').index[-1]-1
        if checkpoint_idx == -1:
            return None
    elif checkpoint_idx < 0:
        checkpoint_idx = len(checkpoints) + checkpoint_idx

    p = re.compile('checkpoint-\d+')
    if len(checkpoints) > 0 and checkpoint_idx in range(len(checkpoints)):
        path = os.path.join(root, checkpoints[checkpoint_idx])
        checkpoint_file = [f for f in os.listdir(path) if p.match(f) and 'metadata' not in f][0]
        # c = deepcopy(agent.config)
        # c.pop('framework')
        # agent_tf = PPOTrainer(env='marl-v0', config=c)
        # agent_tf.restore(os.path.join(path, checkpoint_file))
        # weights = agent_tf.get_weights()
        # agent.set_weights({agent: list(w.values()) for agent, w in weights.items()})
        agent.restore(os.path.join(path, checkpoint_file))
        return int(checkpoint_file.split('-')[1])
