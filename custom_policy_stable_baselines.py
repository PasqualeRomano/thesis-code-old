



import tensorflow as tf
import time,math
from tensorflow.python import training
from tensorflow.contrib import layers
from tensorflow import keras
import training_config
from gym.spaces import Box



from stable_baselines.common.policies import BasePolicy, nature_cnn, register_policy
from  stable_baselines.ddpg.policies import DDPGPolicy


from stable_baselines.ddpg.policies import FeedForwardPolicy
from stable_baselines import DDPG



# class CustomPolicy(DDPGPolicy):
    
#     """
#     Policy object that implements a DDPG-like actor critic, using a feed forward neural network.

#     :param sess: (TensorFlow session) The current TensorFlow session
#     :param ob_space: (Gym Space) The observation space of the environment
#     :param ac_space: (Gym Space) The action space of the environment
#     :param n_env: (int) The number of environments to run
#     :param n_steps: (int) The number of steps to run for each environment
#     :param n_batch: (int) The number of batch to run (n_envs * n_steps)
#     :param reuse: (bool) If the policy is reusable or not
#     :param layers: ([int]) The size of the Neural network for the policy (if None, default to [64, 64])
#     :param cnn_extractor: (function (TensorFlow Tensor, ``**kwargs``): (TensorFlow Tensor)) the CNN feature extraction
#     :param feature_extraction: (str) The feature extraction type ("cnn" or "mlp")
#     :param layer_norm: (bool) enable layer normalisation
#     :param act_fun: (tf.func) the activation function to use in the neural network.
#     :param kwargs: (dict) Extra keyword arguments for the nature CNN feature extraction
#     """

#     def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=False, layers=None,
#                  cnn_extractor=nature_cnn, feature_extraction="mlp",
#                  layer_norm=False, act_fun=tf.nn.relu, **kwargs):
#         super(CustomPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=reuse,
#                                                 scale=(feature_extraction == "mlp"))
        
        
        
       
        
        
        
#         self._kwargs_check(feature_extraction, kwargs)
#         self.layer_norm = layer_norm
#         self.feature_extraction = feature_extraction
#         self.cnn_kwargs = kwargs
#         self.cnn_extractor = cnn_extractor
#         self.reuse = reuse
#         self._qvalue = None
#         if layers is None:
#             layers = [training_config.NH1, training_config.NH2]
#         self.layers = layers

#         assert len(layers) >= 1, "Error: must have at least one hidden layer for the policy."

#         self.activ = act_fun

#     def make_actor(self, obs=None, reuse=False, scope="pi"):
        
        
#         RANDOM_SEED = int((time.time()%10)*1000)
#         n_init = tf.keras.initializers.TruncatedNormal(seed=RANDOM_SEED)
#         u_init = tf.keras.initializers.RandomUniform(minval=-3e-3, maxval=3e-3, seed=RANDOM_SEED)
        
#         if obs is None:
#             obs = self.processed_obs

#         with tf.variable_scope(scope, reuse=reuse):
#             if self.feature_extraction == "cnn":
#                 pi_h = self.cnn_extractor(obs, **self.cnn_kwargs)
#             else:
#                 pi_h = tf.layers.flatten(obs)
#             for i, layer_size in enumerate(self.layers):
#                 pi_h = tf.layers.dense(pi_h, layer_size, name='fc' + str(i),kernel_initializer=n_init)
#                 if self.layer_norm:
#                     pi_h = tf.contrib.layers.layer_norm(pi_h, center=True, scale=True)
#                 pi_h = self.activ(pi_h)
#             self.policy = tf.nn.tanh(tf.layers.dense(pi_h, self.ac_space.shape[0], name=scope,
#                                                      kernel_initializer=tf.random_uniform_initializer(minval=-3e-3,
#                                                                                                       maxval=3e-3)))
#         return self.policy

#     def make_critic(self, obs=None, action=None, reuse=False, scope="qf"):
        
        
#         RANDOM_SEED = int((time.time()%10)*1000)
#         n_init = tf.keras.initializers.TruncatedNormal(seed=RANDOM_SEED)
#         u_init = tf.keras.initializers.RandomUniform(minval=-3e-3, maxval=3e-3, seed=RANDOM_SEED)
        
        
        
        
#         if obs is None:
#             obs = self.processed_obs
#         if action is None:
#             action = self.action_ph

#         with tf.variable_scope(scope, reuse=reuse):
            
           
            
#             qf_h = tf.layers.flatten(obs)
            
                  
#             layer_size_1 =  training_config.NH1
#             layer_size_2 =  training_config.NH2
                                        
#             netx1 = tf.layers.dense(qf_h, layer_size_1, name='netx1',activation= 'relu',kernel_initializer=n_init)
#             netu1 = tf.layers.dense(action, layer_size_2, name='netu1',kernel_initializer=n_init)
#             netx2 = tf.layers.dense(netx1,layer_size_1,name='netx2',activation = 'relu',kernel_initializer=n_init)
#             netu2 = tf.layers.dense(netu1,layer_size_2,name='netu2',activation = 'relu',kernel_initializer=n_init)
#             net_act = tf.keras.activations.relu(netx2+netu2) 
#             qvalue_fn = tf.layers.dense(net_act,1, activation="linear", kernel_initializer=tf.random_uniform_initializer(minval=-3e-3,maxval=3e-3), name="qf_output") 

#             self.qvalue_fn = qvalue_fn
#             self._qvalue = qvalue_fn[:, 0]
#         return self.qvalue_fn

#     def step(self, obs, state=None, mask=None):
#         return self.sess.run(self.policy, {self.obs_ph: obs})

#     def proba_step(self, obs, state=None, mask=None):
#         return self.sess.run(self.policy, {self.obs_ph: obs})

#     def value(self, obs, action, state=None, mask=None):
#         return self.sess.run(self._qvalue, {self.obs_ph: obs, self.action_ph: action})


# Custom MLP policy of two layers of size 16 each
class CustomPolicy_2(FeedForwardPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomPolicy_2, self).__init__(*args, **kwargs,
                                           layers=[training_config.NH1, training_config.NH2],
                                           layer_norm=False,
                                           feature_extraction="mlp")


class FeedForwardTanh(DDPGPolicy):
    """
    Policy object that implements a DDPG-like actor critic, using a feed forward neural network.

    :param sess: (TensorFlow session) The current TensorFlow session
    :param ob_space: (Gym Space) The observation space of the environment
    :param ac_space: (Gym Space) The action space of the environment
    :param n_env: (int) The number of environments to run
    :param n_steps: (int) The number of steps to run for each environment
    :param n_batch: (int) The number of batch to run (n_envs * n_steps)
    :param reuse: (bool) If the policy is reusable or not
    :param layers: ([int]) The size of the Neural network for the policy (if None, default to [64, 64])
    :param cnn_extractor: (function (TensorFlow Tensor, ``**kwargs``): (TensorFlow Tensor)) the CNN feature extraction
    :param feature_extraction: (str) The feature extraction type ("cnn" or "mlp")
    :param layer_norm: (bool) enable layer normalisation
    :param act_fun: (tf.func) the activation function to use in the neural network.
    :param kwargs: (dict) Extra keyword arguments for the nature CNN feature extraction
    """

    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=False, layers=None,
                 cnn_extractor=nature_cnn, feature_extraction="cnn",
                 layer_norm=False, act_fun=tf.nn.tanh, **kwargs):
        super(FeedForwardTanh, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=reuse,
                                                scale=(feature_extraction == "cnn"))

        self._kwargs_check(feature_extraction, kwargs)
        self.layer_norm = layer_norm
        self.feature_extraction = feature_extraction
        self.cnn_kwargs = kwargs
        self.cnn_extractor = cnn_extractor
        self.reuse = reuse
        self._qvalue = None
        if layers is None:
            layers = [64, 64]
        self.layers = layers

        assert len(layers) >= 1, "Error: must have at least one hidden layer for the policy."

        self.activ = act_fun

    def make_actor(self, obs=None, reuse=False, scope="pi"):
        if obs is None:
            obs = self.processed_obs

        with tf.variable_scope(scope, reuse=reuse):
            if self.feature_extraction == "cnn":
                pi_h = self.cnn_extractor(obs, **self.cnn_kwargs)
            else:
                pi_h = tf.layers.flatten(obs)
            for i, layer_size in enumerate(self.layers):
                pi_h = tf.layers.dense(pi_h, layer_size, name='fc' + str(i))
                if self.layer_norm:
                    pi_h = tf.contrib.layers.layer_norm(pi_h, center=True, scale=True)
                pi_h = self.activ(pi_h)
            self.policy = tf.nn.tanh(tf.layers.dense(pi_h, self.ac_space.shape[0], name=scope,
                                                     kernel_initializer=tf.random_uniform_initializer(minval=-3e-3,
                                                                                                      maxval=3e-3)))
        return self.policy

    def make_critic(self, obs=None, action=None, reuse=False, scope="qf"):
        if obs is None:
            obs = self.processed_obs
        if action is None:
            action = self.action_ph

        with tf.variable_scope(scope, reuse=reuse):
            if self.feature_extraction == "cnn":
                qf_h = self.cnn_extractor(obs, **self.cnn_kwargs)
            else:
                qf_h = tf.layers.flatten(obs)
            for i, layer_size in enumerate(self.layers):
                qf_h = tf.layers.dense(qf_h, layer_size, name='fc' + str(i))
                if self.layer_norm:
                    qf_h = tf.contrib.layers.layer_norm(qf_h, center=True, scale=True)
                qf_h = self.activ(qf_h)
                if i == 0:
                    qf_h = tf.concat([qf_h, action], axis=-1)

            # the name attribute is used in pop-art normalization
            qvalue_fn = tf.layers.dense(qf_h, 1, name='qf_output',
                                        kernel_initializer=tf.random_uniform_initializer(minval=-3e-3,
                                                                                         maxval=3e-3))

            self.qvalue_fn = qvalue_fn
            self._qvalue = qvalue_fn[:, 0]
        return self.qvalue_fn

    def step(self, obs, state=None, mask=None):
        return self.sess.run(self.policy, {self.obs_ph: obs})

    def proba_step(self, obs, state=None, mask=None):
        return self.sess.run(self.policy, {self.obs_ph: obs})

    def value(self, obs, action, state=None, mask=None):
        return self.sess.run(self._qvalue, {self.obs_ph: obs, self.action_ph: action})

class CustomPolicy_3(FeedForwardTanh):
    def __init__(self, *args, **kwargs):
        super(CustomPolicy_3, self).__init__(*args, **kwargs,
                                           layers=[training_config.NH1, training_config.NH2],
                                           layer_norm=False,
                                           feature_extraction="mlp")
        
        
class FeedForwardInit(DDPGPolicy):
    """
    Policy object that implements a DDPG-like actor critic, using a feed forward neural network.

    :param sess: (TensorFlow session) The current TensorFlow session
    :param ob_space: (Gym Space) The observation space of the environment
    :param ac_space: (Gym Space) The action space of the environment
    :param n_env: (int) The number of environments to run
    :param n_steps: (int) The number of steps to run for each environment
    :param n_batch: (int) The number of batch to run (n_envs * n_steps)
    :param reuse: (bool) If the policy is reusable or not
    :param layers: ([int]) The size of the Neural network for the policy (if None, default to [64, 64])
    :param cnn_extractor: (function (TensorFlow Tensor, ``**kwargs``): (TensorFlow Tensor)) the CNN feature extraction
    :param feature_extraction: (str) The feature extraction type ("cnn" or "mlp")
    :param layer_norm: (bool) enable layer normalisation
    :param act_fun: (tf.func) the activation function to use in the neural network.
    :param kwargs: (dict) Extra keyword arguments for the nature CNN feature extraction
    """

    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=False, layers=None,
                 cnn_extractor=nature_cnn, feature_extraction="cnn",
                 layer_norm=False, act_fun=tf.nn.relu, **kwargs):
        super(FeedForwardInit, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=reuse,
                                                scale=(feature_extraction == "cnn"))

        self._kwargs_check(feature_extraction, kwargs)
        self.layer_norm = layer_norm
        self.feature_extraction = feature_extraction
        self.cnn_kwargs = kwargs
        self.cnn_extractor = cnn_extractor
        self.reuse = reuse
        self._qvalue = None
        if layers is None:
            layers = [64, 64]
        self.layers = layers

        assert len(layers) >= 1, "Error: must have at least one hidden layer for the policy."

        self.activ = act_fun
        
        self.fan_in = [ training_config.NH1 ,training_config.NH2 ]


    def make_actor(self, obs=None, reuse=False, scope="pi"):
        if obs is None:
            obs = self.processed_obs

        with tf.variable_scope(scope, reuse=reuse):
            if self.feature_extraction == "cnn":
                pi_h = self.cnn_extractor(obs, **self.cnn_kwargs)
            else:
                pi_h = tf.layers.flatten(obs)
            for i, layer_size in enumerate(self.layers):
                pi_h = tf.layers.dense(pi_h, layer_size, name='fc' + str(i))
                if self.layer_norm:
                    pi_h = tf.contrib.layers.layer_norm(pi_h, center=True, scale=True)
                pi_h = self.activ(pi_h)
            self.policy = tf.nn.tanh(tf.layers.dense(pi_h, self.ac_space.shape[0], name=scope,
                                                     kernel_initializer=tf.random_uniform_initializer(minval=-3e-3,
                                                                                                      maxval=3e-3)))
        return self.policy

    def make_critic(self, obs=None, action=None, reuse=False, scope="qf"):
        
        if obs is None:
            obs = self.processed_obs
        if action is None:
            action = self.action_ph
            
        
        
        with tf.variable_scope(scope, reuse=reuse):
            if self.feature_extraction == "cnn":
                qf_h = self.cnn_extractor(obs, **self.cnn_kwargs)
            else:
                qf_h = tf.layers.flatten(obs)
            for i, layer_size in enumerate(self.layers):
                qf_h = tf.layers.dense(qf_h, layer_size,kernel_initializer=tf.random_uniform_initializer(minval=-1/math.sqrt(self.fan_in[i]),maxval=1/math.sqrt(self.fan_in[i])), name='fc' + str(i))
                if self.layer_norm:
                    qf_h = tf.contrib.layers.layer_norm(qf_h, center=True, scale=True)
                qf_h = self.activ(qf_h)
                if i == 0:
                    qf_h = tf.concat([qf_h, action], axis=-1)

            # the name attribute is used in pop-art normalization
            qvalue_fn = tf.layers.dense(qf_h, 1, name='qf_output',
                                        kernel_initializer=tf.random_uniform_initializer(minval=-3e-3,
                                                                                         maxval=3e-3))

            self.qvalue_fn = qvalue_fn
            self._qvalue = qvalue_fn[:, 0]
        return self.qvalue_fn

    def step(self, obs, state=None, mask=None):
        return self.sess.run(self.policy, {self.obs_ph: obs})

    def proba_step(self, obs, state=None, mask=None):
        return self.sess.run(self.policy, {self.obs_ph: obs})

    def value(self, obs, action, state=None, mask=None):
        return self.sess.run(self._qvalue, {self.obs_ph: obs, self.action_ph: action})

class CustomPolicy_4(FeedForwardInit):
    def __init__(self, *args, **kwargs):
        super(CustomPolicy_4, self).__init__(*args, **kwargs,
                                           layers=[training_config.NH1, training_config.NH2],
                                           layer_norm=False,
                                           feature_extraction="mlp")