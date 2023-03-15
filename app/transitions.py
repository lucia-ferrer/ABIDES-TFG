""" Once the politic has been trained this programs will store The Path : State by State in RAW Form. """
from content_alb.reinforcement.models import get_env, load_weights
from content_alb.adversarial.agents import AdversarialWrapper
from ray.rllib.agents.ppo.ppo import PPOTrainer

def get_transition():
    # extract normal transitions
    for id in ids: 
        print(f"Extracting transitions for {id}...")
        print()
        agent.epsilon_greedy = {id: 0.1}
        evaluate(env, agent, config, num_trials=args.episodes, verbose=True)
        pd.DataFrame(agent.transitions[id]).to_csv(f'data/transitions_{id}.csv', index=False)


if __name__ == "__main__":
    # Get the agents 
    env, config = get_env()
    agent = AdversarialWrapper(PPOTrainer)(env='marl-v0', record=True, config=config)
    load_weights(agent)
    ids = config['env_config']['learning_agent_ids'] #if args.agent == -1 else [args.agent] (NON-DEFAULT AGENTS)
    
    # Start the simulation
    # Save the state transactions
    
    get_transition()
    
    # Preprocess
