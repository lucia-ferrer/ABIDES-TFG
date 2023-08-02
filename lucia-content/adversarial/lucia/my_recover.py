""" Recovery classes with window size transitions : based on more than 1 state
        - KNNRecovery -> Based on density distribution function. 
        - GAN Generator with NN -> To be tested. 
"""

import numpy as np
from sklearn.neighbors import BallTree


class Recover():
    def __init__(self, state_dims=None, 
                 # consider_next_state=False,
                 trans=True,
                 window=2, 
                 diff_state=False):
        self.consider_next_state = False # consider_next_state
        self.state_dims = state_dims
        self.transition_dmin = None
        self.diff_state = diff_state if window > 2 else False
        self.trans = trans  if window > 2 else True
        self.window = window if window > 2 else 2

        self.defense = None

    def fit():
        pass

    def wnd_transform_transition(self, X, nan = False):
        """
        This method transform the current transitions to the desired format: 
            Input: np.array with shape (m,n) -> n = (state, action, next_state, reward)
            Output: window size states per transition
        Transitions can be made up with:
            - States Raw info | Increment with previous States
            - Only States | State+Action+Reward
        """
        # Reward/Action, or not. -> [Sn, An, Rn]    -> (S0,A0,R0), (S1,A1,R1), (S2,A2,R2) ...
        x = X[:,:self.state_dims] if not self.trans else self.skip_next_state(X) if not self.consider_next_state else X
        
        if self.transition_dmin is None: self.transition_dmin = x.shape[1]
        
        # Increment difference or not.  -> [Sn+1 - Sn] -> ΔS1-0, ΔS2-1, ΔS3-2, ...
        if self.diff_state: 
            intial_state = x[0,:]
            x = np.row_stack((intial_state, np.diff(x, axis=0)))

        # Window size transitions. -> (S0,S1 ..., Swnd), (Swnd+1, Swnd+2 ..., Swnd+wnd), ...
        y = x.copy()
        
        for indx in range(self.window-1):
            y = np.column_stack((y[:-1,:], x[indx+1:, :]))      

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


class KNNRecovery(Recover):
    def __init__(self, k=1, state_dims=None, 
                 # consider_next_state=False,
                 trans=True,
                 window=2, 
                 diff_state=False):
        self.k = k
        # self.consider_next_state = consider_next_state
        super().__init__(state_dims, trans, window, diff_state)
            
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
        self.X = X if not self.diff_state and self.trans and self.window<=2 else self.wnd_transform_transition(X)
        
        
        # Normalize the data and store the parameters with correct dimension
        self.norm_values = self.defense.norm_parameters(self.X)  # self.norm_translation, self.norm_scaling
        
        
        self.X = self.defense.process_transitions(self.X, self.norm_values)

        # To build the tree if specified skip 'attacked info' -> Next_State, Reward, Action.
        tree =  self.skip_next_transition(self.X) if not self.trans else self.skip_next_state(self.X) if not self.consider_next_state else self.defense.tree
        
        self.tree = BallTree(tree)
    
    def find_parents(self, transition):
        """
        This method will recover from the Fast Index Tree, the k nearest neighbours of the state specified. 
            Input: Last Wnd Raw transitions
            Output: (List of distances , List with K NextStates An+1 obtained from simple transitions (Sn, A, An+1, R) )
        """

        #Transform to obtain wnd size transitions and process the transition
        transitions = self.wnd_transform_transition(transition) if self.window > 2 else [transition]
        transitions = self.defense.process_transitions(transitions, self.norm_values) 

        #Skip attacked info: State/Reward/Action' is specified
        transitions = self.skip_next_transition(transitions) if not self.trans else self.skip_next_state(transitions) if not self.consider_next_state else transitions
        
        #Search for K neighbours
        closest_distances, closest_idxs = self.tree.query(transitions, k=self.k)
        
        #In self.data we have the simple transitions starting in n-1 state. (state, action, next_state, reward)
        return closest_distances.reshape(self.k,), self.data[closest_idxs.reshape(self.k,)][:, -self.state_dims-1:-1] 

    def new_state_from_parents(self, distances, parents):

        if distances.min() == 0:
            return parents[distances.argmin()]
        distances = distances[:, None]
        new_state = np.sum(parents * (distances/distances.sum()), axis=0)
        return new_state

class GainRecovery():
    def __init__(self, state_dims=None, 
                 trans=True,
                 window=2, 
                 diff_state=False,
                 ):
        
        super.__init__(state_dims, trans, window, diff_state)

    def fill_with_nan(self, x=None):
        """
        This method inserts nan elements within the transitions of a transformed transition
                Input: np.array with shape (m, n-transformed*) -> [ Wnd · (State+Action+Reward), ...  ] 
                Output: np.array with shape (m,n-filled) -> [ Wnd-1 · (State+Action+ NaN + Reward), ... ] 
        """
        if x is None : x = self.X 

        no, dim = X.shape
        nan_next = np.full(shape=(self.state_dim, no), fill_value=np.nan)
        # if we do not take into account rewards, it should be in last place else in previous to last
        idx = dim if not self.trans else dim-1 
        filled = x.insert(nan_next, idx, axis=1)
    
    def params():
        #Generator variables
        # Data + Mask as inputs (Random noise is in missing components)
        G_W1 = tf.Variable(xavier_init([dim*2, h_dim]))  
        G_b1 = tf.Variable(tf.zeros(shape = [h_dim]))
        
        G_W2 = tf.Variable(xavier_init([h_dim, h_dim]))
        G_b2 = tf.Variable(tf.zeros(shape = [h_dim]))
        
        G_W3 = tf.Variable(xavier_init([h_dim, dim]))
        G_b3 = tf.Variable(tf.zeros(shape = [dim]))
        
        self.theta_G = [G_W1, G_W2, G_W3, G_b1, G_b2, G_b3]
    
    def generator(x,m):
        ''' Args: x -> transtions, m-> mask to indicate which values to impute
            Returns: G_prob <- probability of being correct (Sigmoid layer) 
        '''
        G_W1, G_W2, G_W3, G_b1, G_b2, G_b3 = self.theta
         # concatenate Mask and DAta
        inputs = tf.concat(values = [x,m], axis = 1)
        G_h1 = tf.nn.relu(tf.matmul(inputs, G_W1) + G_b1)
        G_h2 = tf.nn.relu(tf.matmul(G_h1, G_W2) + G_b2)
        #MinMax normalized output
        G_prob = tf.nn.sigmoid(tf.matmul(G_h2, G_W3) + G_b3)
        return G_prob







