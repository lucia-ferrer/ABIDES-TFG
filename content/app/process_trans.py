""" The transitions has been  """
from content_alb.reinforcement.models import get_env, load_weights, evaluate
from content_alb.adversarial.agents import AdversarialWrapper
from ray.rllib.agents.ppo.ppo import PPOTrainer
import pandas as pd

def read_transition(saved=True):
    # extract normal transitions from csv [Written in get_transitions]
    if not saved: return agent.transitions[ids[0]]
    transitions = {}
    for id in ids: 
        transitions[id] = pd.read_csv(f'../content_alb/data/transitions_{id}.csv')
    return transitions


def transform_transitions(transitions, window=3):
    """ 
    Input: Transitions is a DataFrame with structure (prev_state, action, state, reward)
    Return: transitions with dimension state*window size --> data: increment_1, increment_2, ... increment_n-1
            
    """
    #Obtain and store difference with the next state. /Has 1 less row./
    diff_transitions = transitions.diff()[1:]
    

if __name__ == "__main__":
    # Get the agents 
    env, config = get_env()
    agent = AdversarialWrapper(PPOTrainer)(env='marl-v0', record=True, config=config)
    load_weights(agent)

    # options: _optimal_policy OR  _epsilon_greedy 
    ids = config['env_config']['learning_agent_ids'] #if args.agent == -1 else [args.agent] (NON-DEFAULT AGENTS)

    # Read transitions
    transitions = read_transition(saved=True)
    transitions = transform_transitions(transitions)


    # list of detectors to do -> only 1 
    detectors_list = []
    iterator = detector_experiments.items() if args.detector == -1\
        else ((args.detector, detector_experiments[args.detector],),)
    for detector_name, exp_config in iterator:
        for params in grid_generator(exp_config, args.detector_parameter):
            detectors_list.append((detector_name, params))

    # list of recoveries to do
    recovery_list = []
    iterator = recovery_experiments.items() if args.recovery == -1\
        else ((args.recovery, recovery_experiments[args.recovery],),)
    for recovery_name, exp_config in iterator:
        for params in grid_generator(exp_config, args.recovery_parameter):
            recovery_list.append((recovery_name, params))

    # list of attacks to do
    attacks_list = []
    for id in ids:
        for a_name in ATTACK_CLASS if args.attack == -1 else [args.attack]:
            for params in grid_generator(get_agent_attack_config(id)[a_name], args.attack_parameter):
                attacks_list.append((id, a_name, params))

    # Execute
    results = pd.DataFrame()
    for detector_name, detector_params in detectors_list:
        defenses = {policy_id: Defense(norm=args.norm, detector=DETECTOR_CLASS[detector_name](**detector_params))
                    for policy_id in ids}
        [defense.fit(transitions[policy_id]) for policy_id, defense in defenses.items()]
   
    


