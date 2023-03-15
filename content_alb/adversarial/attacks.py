#import tensorflow as tf
#tf.compat.v1.enable_eager_execution()
import tensorflow as tf
import numpy as np
from cleverhans.tf2.attacks.fast_gradient_method import fast_gradient_method as fgsm
from cleverhans.tf2.attacks.carlini_wagner_l2 import CarliniWagnerL2 as cwl2


class Call_wrapper:
    def __init__(self, model):
        self.model = model
    def __call__(self, state):
        return self.model(state)[0]


class Attack:
    def __init__(self, attack='noise', epsilon=None, **kwargs):
        self.epsilon = epsilon
        if attack == 'noise':
            if self.epsilon is None:
                print("Epsilon needs a value")
                raise AttributeError
            self.perturbation = self._noise_perturbation
        elif attack == 'fgsm':
            if self.epsilon is None:
                print("Epsilon needs a value")
                raise AttributeError
            self.perturbation = self._fgsm_perturbation
        elif attack == 'cw':
            self.perturbation = self._cw_perturbation
        else:
            print(f"Attack {attack} not found, only supported 'noise', 'fgsm' and 'cw'")
        self.pert_params = kwargs

    def _noise_perturbation(self, state, policy):
        noise = self.epsilon / np.sqrt(len(state.flatten()))
        return state + noise*(policy.observation_space.high.flatten()-policy.observation_space.low.flatten())

    def _fgsm_perturbation(self, state, policy):
        with policy.model.graph.as_default():
            with tf.GradientTape() as tape:
                s = tf.constant(state)
                tape.watch(s)
                prediction = policy.model.base_model(state)[0]
                label = tf.one_hot(tf.argmax(prediction, 1), prediction.shape[1])
                loss = tf.compat.v1.losses.mean_squared_error(label, prediction)
            gradient = tape.gradient(loss, s)

        return fgsm(Call_wrapper(policy.model.base_model), state, self.epsilon, 2, **self.pert_params).numpy()[0]

    def _cw_perturbation(self, state, policy):
        y = policy.model.base_model(state)[0].numpy()
        l = np.zeros_like(y)
        l[..., y.argmin()] = 1
        pert_state = cwl2(
            Call_wrapper(policy.model.base_model), batch_size=1, y=l,
            clip_min=policy.observation_space.low.min(),
            clip_max=policy.observation_space.high.max(), **self.pert_params
        ).attack(state.astype('float32'))
        return pert_state

    def attack(self, state, policy):
        og_shape = state.shape
        state = state.reshape((1,) + policy.model.base_model.layers[0].input_shape[0][1:])
        if self._should_attack(state, policy):
            state = self.perturbation(state, policy)
            self._after_attack()
        return state.reshape(og_shape)

    def _should_attack(self, state, policy):
        return True

    def _after_attack(self):
        pass

    def at_episode_start(self):
        pass


class UniformAttack(Attack):
    def __init__(self, freq=1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.freq = freq

    def _should_attack(self, state, policy):
        return np.random.random()<self.freq


class STAttack(Attack):
    def __init__(self, beta=None, total=None, temperature=1, *args, **kwargs):
        super().__init__(attack='noise', *args, **kwargs)
        self.beta = beta
        self.max = total
        self.total = total
        self.temperature = temperature

    def _after_attack(self):
        self.total -= 1

    def _should_attack(self, state, policy):
        if self.total <= 0:
            return False
        with policy.model.graph.as_default():
            s = tf.constant(state)
            prediction = policy.model.base_model(s)[0].eval(session=policy._sess)[0]
        eQ = np.exp(prediction/self.temperature)
        for n in policy.action_space.nvec:
            action_preference = eQ[:n] / eQ[:n].sum()
            eQ = eQ[n:]
            c = action_preference.max() - action_preference.min()
            if c >= self.beta:
                return True
        return False

    def at_episode_start(self):
        self.total = self.max


class VFAttack(Attack):
    def __init__(self, beta=None, *args, **kwargs):
        super().__init__(attack='noise', *args, **kwargs)
        self.beta = beta

    def _should_attack(self, state, policy):
        with policy.model.graph.as_default():
            s = tf.constant(state)
            prediction = policy.model.base_model(s)[0].eval(session=policy._sess)[0]
        for n in policy.action_space.nvec:
            if prediction[:n].max() > self.beta:
                return True
            prediction = prediction[n:]
        return False
