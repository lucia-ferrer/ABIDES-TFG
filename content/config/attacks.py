from adversarial.attacks import UniformAttack, STAttack, VFAttack
import numpy as np

_epsilon = [0.1]
_reward_limits = {'MM': [0, 0], 'PT1': [6, 10]}
_multiplier = 5


def get_agent_attack_config(agent):
    return {
        'UniformAttack': {
            'epsilon': _epsilon,
            'freq': 0.25  # np.linspace(0, 1, _multiplier),
        },
        'STAttack': {
            'epsilon': _epsilon,
            'beta': 0.9,  # np.linspace(1, 0.7, _multiplier),
            'total': 100,
            'temperature': 3
        },
        'VFAttack': {
            'epsilon': _epsilon,
            'beta': 9,  # np.linspace(*_reward_limits[agent], _multiplier),
        },
        'Empty': {}
    }


ATTACK_CLASS = {
    'UniformAttack': UniformAttack,
    'STAttack': STAttack,
    'VFAttack': VFAttack,
    'Empty': lambda *args: {}
}
