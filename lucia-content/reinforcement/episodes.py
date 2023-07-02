from collections import defaultdict
from tqdm import tqdm
import numpy as np
#import tensorflow as tf
#tf.compat.v1.enable_eager_execution()


def evaluate(env, agent, config, num_trials, norm=False, verbose=False, test=False):
    # if 'at_test_start' in agent.__dir__():
    #     agent.at_test_start()
    rewards = defaultdict(list)
    iterator = range(int(num_trials))
    matrix = defaultdict(lambda: np.zeros((2, 2)))
    for _ in iterator if not verbose else tqdm(iterator):
        results, m = episode(env, agent, config, norm, test)
        if m: 
            for id, v in m.items():
            	matrix[id] += v
        for id, reward in results.items():
            rewards[id].append(reward)
    return rewards, matrix


def episode(env, agent, config, norm=False, test=False):
    # Basically done as a fix to get_transitions and test_check, test_attacks ... both working
    if test: agent = agent()

    # restart adversarial attributes
    #if 'at_episode_start' in agent.__dir__():
    #    agent.at_episode_start() 

    ids = config['env_config']['learning_agent_ids']
    norm_state = env.reset()
    episode_total_rewards = {i: 0 for i in ids}
    for t in range(config['horizon']):
        # compute actions
        action = {}
        for id in ids:
            action[id] = agent.compute_single_action(norm_state[id], policy_id=id)
        # execute actions
        norm_state_, norm_reward, done, info = env.step(action)
        # update rewards
        rewards = (env.reward if not norm else norm_reward)
        #if 'last_rewards' in agent.__dict__:
           # agent.last_rewards = rewards
        
        for i, r in rewards.items():
            episode_total_rewards[i] += r # single total rewards per agent.
            if 'last_rewards' in agent.__dict__:
            	agent.last_rewards[i] = np.array([r]) if i not in agent.last_rewards.keys() else \
            		np.append(agent.last_rewards[i], r) 
        
        norm_state = norm_state_

    # do one last forward pass so the agent sees the final states
    for i in range(len(ids)):
        agent.compute_single_action(norm_state[ids[i]], policy_id=ids[i])
    try: return episode_total_rewards, agent.matrix, 
    except AttributeError: return episode_total_rewards, None
