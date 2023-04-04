""" Recover with window size transitions : based on more than 1 state"""


import numpy as np
from sklearn.neighbors import BallTree

class KNNRecovery:
    def __init__(self, k=1, consider_next_state=False, state_dims=None, window=1, difference=False):
        self.k = k
        self.consider_next_state = consider_next_state
        self.state_dims = state_dims
        print('State_dim->',self.state_dims)
        self.defense = None
        self.diff = True
        self.window = window
    
    def fit(self, X):
        self.data = self.defense.train
        
        #We can have problems with norm_parameters in detector. 
        if self.diff: 
            print(f'Type of X {type(X)}')
        self.X = X

        self.tree = self.defense.tree if self.consider_next_state else BallTree(self.skip_next_state(X))

    def skip_next_state(self, transitions):
        dims_indexes = list(range(0, len(transitions[0]))) #-> tuple tamaño 4 [(prev_state, action, obser, rewards)]
        for _ in range(self.state_dims): dims_indexes.pop(len(dims_indexes) - 2) #indexes to remove: new dim_indexes [0,1,2,4] : no next_state
        print('From skip_next_state:  Len(transitions[0])->',len(transitions[0]))
        print('Dim_indexes->',dims_indexes)
        if transitions.ndim>1: 
            print('Transitions from skip_state with ndim>1->', transitions[:, dims_indexes][0])
        else:
            print('Transition from skip_state with 1dim->', np.take(transitions, dims_indexes))
        return transitions[:, dims_indexes] if transitions.ndim > 1 else np.take(transitions, dims_indexes)

    def find_parents(self, transition):
        print('From find_parents -> Calling process with  [transition]')
        transitions = self.defense.process_transitions([transition]) #transition normalized list with [[last_states,(current_transition)]]
        if not self.consider_next_state: transitions = self.skip_next_state(transitions)
        closest_distances, closest_idxs = self.tree.query(transitions, k=self.k)
        #TODO: Verify why different ks are used as an input, if only referenced k=0 (nearest)
        print('Self_data->', self.data[0])
        print('Self_data Dimension->', len(self.data[0]))
        return closest_distances[0], self.data[closest_idxs][:, :, -self.state_dims-1:-1][0]

    def new_state_from_parents(self, distances, parents):
        print('len(distances)->', len(distances), ', parents.ndim->', parents.ndim)
        if distances.min() == 0:
            return parents[distances.argmin()]
        distances = distances[:, None]
        new_state = np.sum(parents * (distances/distances.sum()), axis=0)
        print('parents->',parents)
        print('new_state ->', new_state)
        return new_state

class TimeSeries:
    def __init__(self) -> None:
        pass       
