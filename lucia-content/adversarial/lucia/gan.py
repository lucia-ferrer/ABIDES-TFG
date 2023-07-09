''' Code and reference from : 
        - https://github.com/jsyoon0823/GAIN/tree/master
    GAIN neural network 

    TODO: 
    (1) look if its possible to insert our detector as the discriminator, thus Loss: distance ~ MSE Error
            -> In this case is the hint necessary ? ,
                since we already know the mask and the part that if so has been modified and should be restored
    (2) the batch size should have a logical input
    (3) is correct to assume the iterations the same as the num_trials
    (4) Insert the structure within the agent
        
'''

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np 
from tqdm import tqdm

# normalization, rounding, binary_sampler, uniform_samples, sample_batch_index

class GAIN:
    """Generative Adversarial Nets for Data Imputation """
    def __init__(self, batch_size, hint_rate, alpha, iteratios) -> None:
        # System params
        self.batch_size = batch_size
        self.hint_rate = hint_rate
        self.alpha = alpha
        self.iterations = iterations      # NUM_trials ? 

        #Defense for data
        self.defense = None
        
    def fit(X):
        ''' Impute values in data X : 
        Args: -  X -> original data with missing next state
        Returns: - imputed data -> transition with next state. 
        '''
        data_mask = 1 - np.isnan(X) # x[-state_dim-1:-1]
        no, dim = X.shape
        h_dim = int(dim)
        
        # Normalize the data and store the parameters with correct dimension
        self.norm_values = self.defense.norm_parameters(X)  # self.norm_translation, self.norm_scaling
                
        X = self.defense.process_transitions(X, self.norm_values)

        #-------------GAIN Architecture----------------------------

        #input placeholders : data vector, mask vector, hint vector
        X = tf.placeholder(tf.float32, shape = [None, dim])
        M = tf.placeholder(tf.float32, shape = [None, dim])
        H = tf.placeholder(tf.float32, shape = [None, dim])

        # Discriminator variables
        D_W1 = tf.Variable(xavier_init([dim*2, h_dim])) # Data + Hint as inputs
        D_b1 = tf.Variable(tf.zeros(shape = [h_dim]))
        
        D_W2 = tf.Variable(xavier_init([h_dim, h_dim]))
        D_b2 = tf.Variable(tf.zeros(shape = [h_dim]))
        
        D_W3 = tf.Variable(xavier_init([h_dim, dim]))
        D_b3 = tf.Variable(tf.zeros(shape = [dim]))  # Multi-variate outputs

        theta_D = [D_W1, D_W2, D_W3, D_b1, D_b2, D_b3]

        #Generator variables
        # Data + Mask as inputs (Random noise is in missing components)
        G_W1 = tf.Variable(xavier_init([dim*2, h_dim]))  
        G_b1 = tf.Variable(tf.zeros(shape = [h_dim]))
        
        G_W2 = tf.Variable(xavier_init([h_dim, h_dim]))
        G_b2 = tf.Variable(tf.zeros(shape = [h_dim]))
        
        G_W3 = tf.Variable(xavier_init([h_dim, dim]))
        G_b3 = tf.Variable(tf.zeros(shape = [dim]))
        
        theta_G = [G_W1, G_W2, G_W3, G_b1, G_b2, G_b3]

        def generator(x,m):
            # concatenate Mask and DAta
            inputs = tf.concat(values = [x,m], axis = 1)
            G_h1 = tf.nn.relu(tf.matmul(inputs, G_W1) + G_b1)
            G_h2 = tf.nn.relu(tf.matmul(G_h1, G_W2) + G_b2)
            #MinMax normalized output
            G_prob = tf.nn.sigmoid(tf.matmul(G_h2, G_W3) + G_b3)
            return G_prob

        def discriminator(x,h):
            ''' Args: x -> transtions, h-> hint
                Returns: D_prob <- probability of being attacked (Sigmoid layer) '''
            #concatenate Data and Hint
            input = tf.concat(values = [x,h], axis = 1)
            D_h1 = tf.nn.relu(tf.matmul(inputs, D_W1) + D_b1)
            D_h2 = tf.nn.relu(tf.matmul(D_h1, D_W2) + D_b2)
            D_logit = tf.matmul(D_h2, D_W3) + D_b3
            return  tf.nn.sigmoid(D_logit) # D_prob <- probability of attack
        
        ## --------- GAIN structure ----------------
        #TODO: this section can be moved to a subclass of defense as a whole ? 
        #Generator
        G_sample = generator(X, M)
        #Combine with observed data
        Hat_X = X * M * G_sample(1-M)
        #Discriminator
        D_prob = discriminator(Hat_X, H)

        ## GAIN Losss
        D_loss_temp = - tf.reduce_mean(M * tf.log(D_prob + 1e-8)\ 
                                        + (1-M) * tf.log(1. - D_prob + 1e-8))
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
            Z_mb = uniform_sampler(0. 0.01, batch_size, dim)
            #Sample hint vectors
            H_mb_temp = binary_sampler(hint_rante,batch_size, dim)

            H_mb = M_mb * H_mb_temp

            # Combine random vectors with observed vectors
            X_mb = M_mb * X_mb + (1-M_mb) * Z_mb

            _, D_loss_curr = sess.run([D_solver, D_loss_temp], feed_dict = {M: M_mb, X:X_mb, H: H_mb})
            _, G_loss_curr, MSE_loss_curr = \
                sess.run([G_solver, G_loss_temp, MSE_loss], feed_dict = {M: M_mb, X:X_mb, H: H_mb})

        ## --------- Return imputed data 
        Z_mb = uniform_sampler(0, 0.01, no, dim) 
        M_mb = data_m
        X_mb = norm_data_x          
        X_mb = M_mb * X_mb + (1-M_mb) * Z_mb 
            
        imputed_data = sess.run([G_sample], feed_dict = {X: X_mb, M: M_mb})[0]
            
        imputed_data = data_m * norm_data_x + (1-data_m) * imputed_data
            
        # Renormalization
        imputed_data = renormalization(imputed_data, norm_parameters)  
            
        # Rounding
        imputed_data = rounding(imputed_data, data_x)  
                    
        return imputed_data
    
    @static method
    def xavier_init(size):
        ''' Xavier Weights initialization
        - Args: size : vector size
        - Returns: initialized random vector'''
        in_dim = size[0]
        xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
        return tf.random_normal(shape=size, stddev = xavier_stddev)

    @static method
    def rounding():
        pass 

    @static method 
    def renormalization():
        pass 
    
    @static method
    def uniform_sampler():
        pass 
    
    @static method
    def binary_sampler():
        pass 
    









        