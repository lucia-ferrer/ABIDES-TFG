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

    def process_transitions(self, transitions, norm_values=None):
        # Use norm parameters obtained to fit the defense
        if norm_values is None:
            return np.true_divide(transitions - self.norm_translation, self.norm_scaling,
                              out=np.ones_like(transitions), where=self.norm_scaling != 0)
        # Use special norm parameters -> Recovery Class may use different values. 
        return np.true_divide(transitions - norm_values[0], norm_values[1],
                              out=np.ones_like(transitions), where=norm_values[1] != 0)

    def fit(self, X):
        # set normalization parameters
        self.train = X.copy()
        self.norm_translation, self.norm_scaling = self.norm_parameters()

        # normalize transitions and set auxiliar structures
        self.normalized = self.process_transitions(self.train)
        self.tree = BallTree(self.normalized)
 
        # fit submodules
        self.fit_detector()
        self.fit_recovery()

    def norm_parameters(self, X=None):
        train = self.train if X is None else X
        if self.norm == 'z-score':
            norm_translation = train.mean(axis=0)
            norm_scaling = train.std(axis=0)
        elif self.norm == 'min-max':
            norm_translation = train.min(axis=0)
            norm_scaling = train.max(axis=0) - train.min(axis=0)
        return norm_translation, norm_scaling

    def fit_detector(self):
        if self.detector is not None:
            self.detector.defense = self
            self.detector.fit(self.normalized)

    def fit_recovery(self):
        if self.recovery is not None and self.recovery != 'cheat':
            self.recovery.defense = self
            self.recovery.fit(self.train)
 
    def is_adversarial(self, t):
        return self.detector.predict([t])[0] 

    def recover(self, t):
        distances, parents = self.recovery.find_parents(t)
        return self.recovery.new_state_from_parents(distances, parents)

class GainDefense(Defense):
    def __init__(self, batch, hint, alpha, iterations,
                    norm = 'min-max', detector=None, recovery=None): 
        if norm not in available_norms:
            raise Exception(f"Normalization type not found, must be one of {available_norms}")
        #agent systems
        self.norm = norm
        self.detector = detector
        self.recovery = recovery

         # System params
        self.batch_size = batch
        self.hint_rate = hint
        self.alpha = alpha
        self.iterations = iterations      # NUM_trials ? 
    
    def fit(self, X):
        ''' Impute values in data X : 
                Args: -  X -> original data with missing next state
                Returns: - imputed data -> transition with next state. 
        '''
        # set normalization parameters
        self.train = X.copy()
        self.norm_translation, self.norm_scaling = self.norm_parameters()

        # normalize transitions and set auxiliar structures
        self.normalized = self.process_transitions(self.train)
        #nok = self.recover.skip_next_state(self.normalized)

        self.detector.defense = self
        self.recovery.defense = self

        X = self.recovery.fill_with_nan(self.data_incomplete) 
        data_mask = 1 - np.isnan(X)
        no, dim = X.shape
        h_dim = int(dim)

        #-------------GAIN Architecture----------------------------

        #input placeholders : data vector, mask vector, hint vector
        X = tf.placeholder(tf.float32, shape = [None, dim])
        M = tf.placeholder(tf.float32, shape = [None, dim])
        H = tf.placeholder(tf.float32, shape = [None, dim])
        
        self.detector.params()
        self.recovery.params()

        self.train()
    
        #Generator
        G_sample = self.recovery.generator(self.X, self.M)
        #Combine with observed data
        Hat_X = X * M * G_sample(1-M)

        # TODO: Update so as to accept KNN detector, and the loss function other.
        #Discriminator
        D_prob = self.detector.discriminator(Hat_X, H)

        ## GAIN Losss
        D_loss_temp = - tf.reduce_mean(M * tf.log(D_prob + 1e-8) + (1-M) * tf.log(1. - D_prob + 1e-8))
        G_loss_temp = - tf.reduce_mean((1-M) * tf.log(D_prob + 1e-8))

        MSE_loss = tf.reduce_mean((M * X - M * G_sample)**2) / tf.reduce_mean(M)

        D_loss = D_loss_temp
        G_loss = G_loss_temp
        
        ## ---------------- GAIN solver -----------
        D_solver = tf.train.AdamOptimizer().minimize(D_loss, var_list=theta_D)
        G_solver = tf.train.AdamOptimizer().minimize(G_loss, var_list=theta_G)

        sess = tf.Session()
        sess.run(tf.global_variables_initalizer())

        #mb > masked ?? and batch size sample

        #Start iterations
        for it in tqdm(range(iterations)):
            # sample bacth 
            batch_idx = sample_batch_index(no, batch_size)
            X_mb = norm_values[batch_idx, :]
            M_mb = data_m[batch_idx, :]

            #Sample random vectors
            Z_mb = uniform_sampler(0., 0.01, batch_size, dim)
            #Sample hint vectors
            H_mb_temp = binary_sampler(hint_rante,batch_size, dim)

            H_mb = M_mb * H_mb_temp

            # Combine random vectors with observed vectors
            X_mb = M_mb * X_mb + (1-M_mb) * Z_mb

            _, D_loss_curr = sess.run([D_solver, D_loss_temp], feed_dict = {M: M_mb, X:X_mb, H: H_mb})
            _, G_loss_curr, MSE_loss_curr = \
                sess.run([G_solver, G_loss_temp, MSE_loss], feed_dict = {M: M_mb, X:X_mb, H: H_mb})
        # does not return the imputed data, but we could. 
        return MSE_loss_curr

    @staticmethod
    def xavier_init(size):
        ''' Xavier Weights initialization
        - Args: size : vector size
        - Returns: initialized random vector'''
        in_dim = size[0]
        xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
        return tf.random_normal(shape=size, stddev = xavier_stddev)





