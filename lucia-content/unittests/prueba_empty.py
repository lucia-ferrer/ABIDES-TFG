from reinforcement.models import get_env, load_weights
from ray.rllib.agents.ppo.ppo import PPOTrainer
import pandas as pd

from adversarial.lucia.my_agent import AdversarialWrapper
from adversarial.lucia.my_defense import Defense

from reinforcement.episodes import evaluate

if __name__ == '__main__':
    env,config = get_env()
    ids = config['env_config']['learning_agent_ids'] #MM, PT1

    agent = PPOTrainer(env='marl-v0', config=config)
    agentADV = AdversarialWrapper(PPOTrainer)(env='marl-v0', config=config)()

    load_chkpoint = load_weights(agent)
    load_chkpointADV = load_weights(agentADV)

    agentADV.attacker = {'PT1': {}}
    agentADV.defender = {'PT1': Defense()}
    agentADV.defender['PT1'].fit()

    #evaluate returns 
    #   dict with lists:total_rewards (per trial), 
    #   dict:matrix
    results = evaluate(env, agent, config, num_trials=10, verbose=True)
    resultsADV = evaluate(env, agentADV,  num_trials=10, verbose=True)

    print('Normal results: ', results)
    print('Normal results with empty attack and no recovery: ', resultsADV)



