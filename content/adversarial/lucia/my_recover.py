""" Recovert classes with window size transitions : based on more than 1 state
        - KNNRecovery -> Based on density distribution function. 
        - TimeSeries -> To be implemented. 
"""

import numpy as np
from sklearn.neighbors import BallTree

class KNNRecovery:
    def __init__(self, k=1, state_dims=None, 
                 consider_next_state=False,
                 consider_transition=True,
                 window=2, 
                 diff_state=False):
        self.k = k
        self.consider_next_state = consider_next_state
        self.state_dims = state_dims
        self.transition_dmin = None
        self.defense = None
        self.diff_state = diff_state if window > 2 else False
        self.consider_transition = consider_transition  if window > 2 else True
        self.window = window if window > 2 else 2
    
    def fit(self, X):
        #We can have problems with norm_parameters in detector. 
        """
            Input: X -> not normalized
            Stored structures: 
            - self.data -> Normal transitions, if window we only store from n-1 Transition. 
            - self.X -> Processed transitions, with wnd size transitions, increments, and normalized. 
            - self.tree -> Structure for BinarySearch. 
        """
        self.data = self.defense.train[self.window-1:] if self.window > 2 else self.defense.train
        self.X = X if not self.diff_state and self.consider_transition and self.window<=2 else self.wnd_transform_transition(X)
        #print(f"Parameter for norm_param \t self.X type->{type(self.X)}, self.X shape->{self.X.shape}")
        
        # Normalize the data and store the parameters with correct dimension
        self.norm_values = self.defense.norm_parameters(self.X)  # self.norm_translation, self.norm_scaling
        #print(f"Parameter for defense.process \t  self.norm_values[0]-type->{type(self.norm_values)}, self.norm_values[0]-shape->{self.norm_values[0].shape}")
        #print("Are the defense and recovery same norm_values? -> ", self.defense.norm_translation == self.norm_values[0])
        self.X = self.defense.process_transitions(self.X, self.norm_values)

        # To build the tree if specified skip 'attacked info' -> Next_State, Reward, Action.
        tree =  self.skip_next_transition(self.X) if not self.consider_transition else self.skip_next_state(self.X) if not self.consider_next_state else self.defense.tree
        #print(f"Parameter for BallTree \t tree type->{type(tree)}, tree shape->{tree.shape}")
        self.tree = BallTree(tree)

    def wnd_transform_transition(self, X):
        """
        This method transform the current transitions to the desired format: 
            Input: np.array with shape (m,n) -> n = (state, action, next_state, reward)
            Output: window size states per transition
        Transitions can be made up with:
            - States Raw info | Increment with previous States
            - Only States | State+Action+Reward
        """
        # Reward/Action, or not. -> [Sn, An, Rn]    -> (S0,A0,R0), (S1,A1,R1), (S2,A2,R2) ...
        x = X[:,:self.state_dims] if not self.consider_transition else self.skip_next_state(X) if not self.consider_next_state else X
        print('X_shape[1]->', x.shape[1],'X_shape->', x.shape) if len(X)>1 else print()
        if self.transition_dmin is None: self.transition_dmin = x.shape[1]
        print('Transition_shape->', self.transition_dmin)

        # Increment difference or not.  -> [Sn+1 - Sn] -> ΔS1-0, ΔS2-1, ΔS3-2, ...
        if self.diff_state: x = np.diff(x, axis=0) 

        # Window size transitions. -> (S0,S1 ..., Swnd), (Swnd+1, Swnd+2 ..., Swnd+wnd), ...
        y = x.copy()
        print(f'X_shape->{x.shape}, Y_shape->{y.shape}')
        for indx in range(self.window-1):
            y = np.column_stack((y[:-1,:], x[indx+1:, :]))
            print(f'X_shape->{x.shape}, Y_shape->{y.shape}')
    
        #y = y.flatten() if y.ndim<3 else y
        print(f"Return of transform_transition  y-type->{type(y)}, y-shape->{y.shape}, y-ndim->{y.ndim}")
        #Self.X contains the window sized transitions, to be input to the Tree and Recuperation of Indexes. 
        return y
    
    def skip_next_transition(self, transitions):
        """
        This method eliminates the next transition of a transformed transition
                Input: np.array with shape (m,n) -> [ Wnd · (State+Action+Reward), ...  ] 
                Output: np.array with shape (m,n-transition) -> [ Wnd-1 · (State+Action+Reward), ... ] 
        """
        dims_indexes = list(range(len(transitions[0]))) if transitions.ndim > 1 else list(range(len(transitions)))
        for _ in range(self.transition_dmin): dims_indexes.pop(len(dims_indexes) - 1) 
        return transitions[:, dims_indexes] if transitions.ndim > 1 else np.take(transitions, dims_indexes)

    def skip_next_state(self, transitions):
        """
        This method eliminates the next state of a transition.
            Input: np.array with shape (m,n) -> n : (State, Action, Next_State, Reward)  
            Output: np.array with shape (m,n-state) -> n : (State, Action, Reward)    
        """
        dims_indexes = list(range(len(transitions[0]))) if transitions.ndim > 1 else list(range(len(transitions))) # (prev_state, action, obser, rewards)
        for _ in range(self.state_dims): dims_indexes.pop(len(dims_indexes) - 2) 
        return transitions[:, dims_indexes] if transitions.ndim > 1 else np.take(transitions, dims_indexes)

    def find_parents(self, transition):
        """
        This method will recover from the Fast Index Tree, the k nearest neighbours of the state specified. 
            Input: Last Wnd Raw transitions
            Output: List with K simple transitions (Sn, A, An+1, R)
        """

        #Transform to obtain wnd size transitions and process the transition
        transitions = self.wnd_transform_transition(transition) if self.window > 2 else [transition]
        transitions = self.defense.process_transitions(transitions, self.norm_values) 

        #Skip attacked info: State/Reward/Action' is specified
        transitions = self.skip_next_transition(transitions) if not self.consider_transition else self.skip_next_state(transitions) if not self.consider_next_state else transitions
        
        #Search for K neighbours
        closest_distances, closest_idxs = self.tree.query(transitions, k=self.k)

        #In self.data we have the simple transitions starting in n-1 state. (state, action, next_state, reward)
        return closest_distances[0], self.data[closest_idxs][:, :, -self.state_dims-1:-1][0]

    def new_state_from_parents(self, distances, parents):
        if distances.min() == 0:
            return parents[distances.argmin()]
        distances = distances[:, None]
        new_state = np.sum(parents * (distances/distances.sum()), axis=0)
        return new_state

class TimeSeries:
    def __init__(self) -> None:
        pass       
