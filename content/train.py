import os
import numpy as np
import gym
from gym.envs.registration import register
import ray
from ray import tune
from ray.tune.registry import register_env
from abides_gym.envs.marl_environment_v0 import SubGymMultiAgentRLEnv_v0
from abides_core.utils import str_to_ns
from scripts.marl_utils import multi_agent_init, multi_agent_policies

mm_add_volume = 0
pt_add_volume = 0
pt_add_momentum = 1
num_pts = 1
L = 2
M = 2
d = 0
timestep_duration = "60S"

register_env(
    "marl-v0",
    lambda config: SubGymMultiAgentRLEnv_v0(**config),
)
ray.shutdown()
ray.init()
name_xp = f"ppo_marl_vol{mm_add_volume}{pt_add_volume}_L{L}_d{d}_M{M}_pts{num_pts}"
base_log_dir = os.getcwd() + '/results/' + name_xp

env_config = multi_agent_init(num_pts, mm_add_volume, pt_add_volume, L, d, M,
                base_log_dir,pt_add_momentum=pt_add_momentum,
                timestep_duration=timestep_duration)
# env_config["linear_oracle"] = True
multiagent_policies = multi_agent_policies(env_config['learning_agent_ids'],
                        env_config['observation_space'],env_config['action_space'])



config = {
        "env": "marl-v0",
        "env_config": env_config,
        "seed": 0,
        "num_gpus": 0,
        "num_workers": 0,
        "multiagent": multiagent_policies,
        "framework": "tf2",
        "horizon": int((str_to_ns(env_config["mkt_close"]) - \
                    str_to_ns(env_config["mkt_open"]))/str_to_ns(env_config["timestep_duration"]))
    }

stop = {
    "training_iteration": 5000
}

tune.run(
    "PPO",
    name=name_xp,
    resume=True,
    stop=stop,
    local_dir='results',
    checkpoint_at_end=True,
    checkpoint_freq=50,
    config=config,
    verbose=1,
)
