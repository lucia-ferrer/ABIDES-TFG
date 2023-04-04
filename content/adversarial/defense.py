import numpy as np
from sklearn.neighbors import BallTree

available_norms = ['z-score', 'min-max']


class Defense:
    def __init__(self, norm='z-score', detector=None, recovery=None):
        if norm not in available_norms:
            raise Exception(f"Normalization type not found, must be one of {available_norms}")
        self.norm = norm
        self.detector = detector
        self.recovery = recovery

    def process_transitions(self, transitions):
        if isinstance(transitions, np.ndarray): print('Transition Dim->',transition.ndim,'\tShape->', transition.shape)
        else: print('Transition len ->', len(transitions))
        return np.true_divide(transitions - self.norm_translation, self.norm_scaling,
                              out=np.ones_like(transitions), where=self.norm_scaling != 0)

    def fit(self, X):
        # set normalization parameters
        self.train = X.copy()
        if self.norm == 'z-score':
            self.norm_translation = self.train.mean(axis=0)
            self.norm_scaling = self.train.std(axis=0)
        elif self.norm == 'min-max':
            self.norm_translation = self.train.min(axis=0)
            self.norm_scaling = self.train.max(axis=0) - self.train.min(axis=0)

        # normalize transitions and set auxiliar structures
        self.normalized = self.process_transitions(self.train)
        self.tree = BallTree(self.normalized)
 
        # fit submodules
        self.fit_detector()
        self.fit_recovery()

    def fit_detector(self):
        if self.detector is not None:
            self.detector.defense = self
            self.detector.fit(self.normalized)

    def fit_recovery(self):
        if self.recovery is not None and self.recovery != 'cheat':
            self.recovery.defense = self
            self.recovery.fit(self.normalized)

    def is_adversarial(self, t):
        return self.detector.predict([t])[0] 

    def recover(self, t):
        distances, parents = self.recovery.find_parents(t)
        return self.recovery.new_state_from_parents(distances, parents)

