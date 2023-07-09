from reinforcement.models import get_env, load_weights
from ray.rllib.agents.ppo.ppo import PPOTrainer
import numpy as np
import pandas as pd

from adversarial.lucia.my_agent import AdversarialWrapper
from adversarial.lucia.my_defense import Defense

from reinforcement.episodes import evaluate

if __name__ == '__main__':
    env,config = get_env()
    ids = config['env_config']['learning_agent_ids'] #MM, PT1
    transitions = {id: pd.read_csv(f'data/transitions_{id}.csv', header=None).values for id in ids}

    agent = PPOTrainer(env='marl-v0', config=config)
    agentADV = AdversarialWrapper(PPOTrainer)(env='marl-v0', config=config)

    load_chkpoint = load_weights(agent)
    load_chkpointADV = load_weights(agentADV)

    agentADV.attacker = {'PT1': {}}
    agentADV.defender = {'PT1': Defense()}
    agentADV.defender['PT1'].fit(transitions['PT1'])

    #evaluate returns 
    #   dict with lists:total_rewards (per trial), 
    #   dict:matrix
    resultsADV = pd.DataFrame(evaluate(env, agentADV,  config, num_trials=50, verbose=True, test=False)[0])
    results = pd.DataFrame(evaluate(env, agent, config, num_trials=50, verbose=True)[0])

    print('Normal results mean', results.mean(axis=0), '\n std dev',  results.std(axis=0))
    print('Normal results with empty attack and no recovery mean', resultsADV.mean(axis=0), '\n std dev', results.std(axis=0))
    
    resultsADV.to_csv('./resultsADV.csv')
    results.to_csv('./results.csv')



