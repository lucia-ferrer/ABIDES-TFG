import numpy as np
from copy import deepcopy
from collections import defaultdict

#import tensorflow as tf
#tf.compat.v1.enable_eager_execution()


def AdversarialWrapper(cls):
    class Adversarial(cls):
        def __init__(self, attacker=None, defender=None, record=False, epsilon_greedy=None, *args, **kwargs):
            self.attacker = attacker if attacker is not None else {}
            self.defender = defender if defender is not None else {}
            self.epsilon_greedy = epsilon_greedy if epsilon_greedy is not None else {}
            self.policy = defaultdict(lambda: self._optimal_policy)
            for id in self.epsilon_greedy:
                self.policy[id] = self._epsilon_greedy_policy
            self.record = record

            self.at_test_start()
            self.at_episode_start()
            super().__init__(*args, **kwargs)

        def _optimal_policy(self, observation, policy_id, *args, **kwargs):
            return super().compute_single_action(observation, policy_id=policy_id, *args, **kwargs)

        def _epsilon_greedy_policy(self, observation, policy_id, *args, **kwargs):
            if np.random.random() > self.epsilon_greedy[policy_id]:
                return self.get_policy(policy_id).action_space.sample()
            return self._optimal_policy(observation, policy_id, *args, **kwargs)

        def at_test_start(self):
            self.transitions = defaultdict(lambda: [])
            self.matrix = defaultdict(lambda: np.zeros((2, 2)))

        def at_episode_start(self):
            for _, a in self.attacker.items():
                if 'at_episode_start' in a.__dir__():
                    a.at_episode_start()

            self.last_states = {}
            self.last_actions = {}
            self.last_rewards = {}

        def get_transition(self, observation, policy_id):
            state = self.last_states[policy_id].flatten()
            next_state = observation.flatten()
            # one-hot encode actions
            action_space = self.get_policy(policy_id).action_space.nvec
            action = np.zeros(action_space.sum())
            action[[self.last_actions[policy_id][0], self.last_actions[policy_id][1]+action_space[0]]] = 1
            # generate transitions
            return np.concatenate((state, action, next_state, [self.last_rewards[policy_id]]))

        def compute_single_action(self, observation, policy_id, *args, **kwargs):

            # local_worker = self.workers.local_worker()
            #
            # # Check the preprocessor and preprocess, if necessary.
            # pp = local_worker.preprocessors[policy_id]
            # if pp and type(pp).__name__ != "NoPreprocessor":
            #     observation = pp.transform(observation)
            # filtered_observation = local_worker.filters[policy_id](
            #     observation, update=False)
            og_observation = observation[:]
            transition = None

            # attack
            if policy_id in self.attacker and self.attacker[policy_id] != {}:
                observation = self.attacker[policy_id].attack(observation, self.get_policy(policy_id))

            # defense
            if policy_id in self.defender and policy_id in self.last_states:
                transition = self.get_transition(observation, policy_id)
                # detection
                is_attack = np.linalg.norm((observation - og_observation).flatten()) > 0
                is_adversarial = self.defender[policy_id].is_adversarial(transition)
                self.matrix[policy_id][int(is_attack), int(is_adversarial)] += 1

                if is_adversarial:
                    if self.defender[policy_id].recovery == 'cheat':
                        observation = og_observation
                    else:
                        # recovery
                        observation = self.defender[policy_id].recover(transition).reshape(observation.shape)

            # record transitions
            if self.record and policy_id in self.last_states:
                if transition is None:
                    transition = self.get_transition(observation, policy_id)
                self.transitions[policy_id].append(transition)

            self.last_states[policy_id] = observation
            self.last_actions[policy_id] = self.policy[policy_id](observation, policy_id, *args, **kwargs)
            return self.last_actions[policy_id]
    return Adversarial
