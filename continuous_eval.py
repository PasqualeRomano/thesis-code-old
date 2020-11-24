import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers 
import random
from collections import deque
import time
import signal
import matplotlib.pyplot as plt
from robot_bullet import Robot
import training_config as tc


tf.compat.v1.disable_eager_execution()
RANDOM_SEED = int((time.time()%10)*1000)
print("Seed = %d" %  RANDOM_SEED)
np .random.seed     (RANDOM_SEED)
tf.compat.v1.set_random_seed(RANDOM_SEED)
random.seed         (RANDOM_SEED)
n_init = tf.keras.initializers.TruncatedNormal(seed=RANDOM_SEED)
u_init = tf.keras.initializers.RandomUniform(minval=-2., maxval=2., seed=RANDOM_SEED)


NEPISODES               = tc.NEPISODES                  # Max training steps
NSTEPS                  = tc.NSTEPS                     # Max episode length
QVALUE_LEARNING_RATE    = tc.QVALUE_LEARNING_RATE       # Base learning rate for the Q-value Network
POLICY_LEARNING_RATE    = tc.POLICY_LEARNING_RATE       # Base learning rate for the policy network
DECAY_RATE              = tc.DECAY_RATE                 # Discount factor 
UPDATE_RATE             = tc.UPDATE_RATE                # Homotopy rate to update the networks
REPLAY_SIZE             = tc.REPLAY_SIZE                # Size of replay buffer
BATCH_SIZE              = tc.BATCH_SIZE                 # Number of points to be fed in stochastic gradient
NH1 = NH2               = tc.NH1                        # Hidden layer size
range_esp               = tc.range_esp

reward_weights  = [1.,0.0,0.00]


sim_number = 10
RANDSET =0
env                 = Robot("single_pendulum.urdf")       
env_rend            = Robot("single_pendulum.urdf",sim_number=sim_number) #for rendering

env.RANDSET = 0

step_expl = 0.
epi_expl = 0.

NX                  = 3          # ... training converges with q,qdot with 2x more neurones.
NU                  = 1            # Control is dim-1: joint torque

def angle_normalize(x):
    return min(x%(2*np.pi),abs(x%(2*np.pi)-2*np.pi))

class QValueNetwork:
    def __init__(self):
        nvars           = len(tf.compat.v1.trainable_variables())

        x       =  keras.Input(shape=(NX,),name="State")
        u       =  keras.Input(shape=(NU,),name="Control")

        netx1 = layers.Dense(NH1, activation="relu", kernel_initializer=n_init, name="netx1")(x)
        netx2 = layers.Dense(NH2, activation="linear", kernel_initializer=n_init, name="netx2")(netx1)
        netu1 = layers.Dense(NH1, activation="linear", kernel_initializer=n_init, name="netu1")(u)
        netu2 = layers.Dense(NH2, activation="linear", kernel_initializer=n_init, name="netu2")(netu1)
        net_act = tf.keras.activations.relu(netx2+netu2)
        qvalue = layers.Dense(1, activation="linear", kernel_initializer=u_init, name="qvalue")(net_act)
        
        qvalue_model = keras.Model(
            inputs=[x,u],
            outputs=[qvalue],
        )

        self.x          = x                                # Network state   <x> input in Q(x,u)
        self.u          = u                                # Network control <u> input in Q(x,u)
        self.qvalue     = qvalue                           # Network output  <Q>
        self.variables  = tf.compat.v1.trainable_variables()[nvars:] # Variables to be trained
        self.hidens = [ netx1, netx2, netu1, netu2 ]                  # Hidden layers for debug
        self.model = qvalue_model

    def setupOptim(self):
        qref = tf.compat.v1.placeholder(tf.float32, [None, 1])
        loss = tf.keras.losses.MeanSquaredError()
        optim = tf.compat.v1.train.AdamOptimizer(QVALUE_LEARNING_RATE).minimize(loss(qref,self.qvalue))
        gradient = tf.gradients(self.qvalue, self.u)[0] / float(BATCH_SIZE)
        
        self.qref       = qref          # Reference Q-values
        self.optim      = optim         # Optimizer
        self.gradient   = gradient      # Gradient of Q wrt the control  dQ/du (for policy training)
        return self

    def setupTargetAssign(self,nominalNet,tau=UPDATE_RATE):
        self.update_variables = \
            [ target.assign( tau*ref + (1-tau)*target )  \
                  for target,ref in zip(self.variables,nominalNet.variables) ]
        return self

class PolicyNetwork:
    def __init__(self):
        nvars = len(tf.compat.v1.trainable_variables())
        
        x = keras.Input(shape=(NX,),name="State")
        # Define Sequential model with 3 layers
        net = layers.Dense(NH1, activation="relu", kernel_initializer=n_init, name="net")(x)
        net = layers.Dense(NH2, activation="relu", kernel_initializer=n_init, name="net2")(net)
        policy = layers.Dense(NU, activation="tanh", kernel_initializer=u_init, name="netu1")(net)*2. #1nm max torque

        policy_model = keras.Model(
            inputs=[x],
            outputs=[policy],
        )
        
        self.x          = x                                     # Network input <x> in Pi(x)
        self.policy     = policy                                # Network output <Pi>
        self.variables  = tf.compat.v1.trainable_variables()[nvars:]      # Variables to be trained
        self.model      = policy_model

    def setupOptim(self):

        qgradient = tf.compat.v1.placeholder(tf.float32, [None, NU])
        grad = tf.gradients(self.policy, self.variables, -qgradient)
        optim = tf.compat.v1.train.AdamOptimizer(POLICY_LEARNING_RATE).\
            apply_gradients(zip(grad, self.variables))

        # Q-value gradient wrt control (input value)
        self.qgradient = qgradient
        self.optim = optim         # Optimizer
        return self


    def setupTargetAssign(self,nominalNet,tau=UPDATE_RATE):
        self.update_variables = \
            [ target.assign( tau*ref + (1-tau)*target )  \
                  for target,ref in zip(self.variables,nominalNet.variables) ]
        return self
 

class ReplayItem:
    def __init__(self,x,u,r,d,x2):
        self.x          = x
        self.u          = u
        self.reward     = r
        self.done       = d
        self.x2         = x2

replayDeque = deque()
 
 
 
policy          = PolicyNetwork(). setupOptim()
policyTarget    = PolicyNetwork(). setupTargetAssign(policy)
 
qvalue          = QValueNetwork(). setupOptim()
qvalueTarget    = QValueNetwork(). setupTargetAssign(qvalue)
    

model_save = 'DDPG_continuous_saved.chkpt'


sess            = tf.compat.v1.InteractiveSession()
tf.compat.v1.global_variables_initializer().run()
tf.compat.v1.train.Saver().restore(sess,model_save)



robot = Robot("single_pendulum.urdf")
robot.sim_number=1
robot.RANDSET =0
robot.GUI_ENABLED = 0
robot.SINCOS=1
path_log= "/home/pasquale/Desktop/thesis/thesis-code/1D_pendulum/continuous/"
robot.setupSim()

#Evaluate policy 
#env.robot.stopSim()
#env = PendulumPyB()

#Check convergence
up_reach = False
h_sum_last =0
for i in range(NSTEPS):
    
    x = np.array([[robot.states_sincos[1][0],robot.states_sincos[1][1],robot.states_dot[1][3]]])    
    action =sess.run(policy.policy, feed_dict={ policy.x: x }) 
    action=action.tolist()
    robot.simulateDyn(action[0])
    
         
    if angle_normalize(robot.states[1][3]) < 1*np.pi/180 and up_reach == False:
         first_step_up = i
         up_reach = True
         
    if i >= NSTEPS -1:
        h_sum_last+= -angle_normalize(robot.states[1][3])**2

h_mean_last = h_sum_last/NSTEPS
print("mean return 20 last steps: "+str(h_mean_last)+", first reached top at "+str(first_step_up))

robot.stopSim()  

#mean return 20 last steps: -2.2419180192174818e-07, first reached top at 30

#valuta policy con (10x)random reset 
#salvare a che step arriva in posizione verticale (anche con random reset)
robot = Robot("single_pendulum.urdf")
robot.sim_number=1
robot.RANDSET =1
robot.GUI_ENABLED = 1
robot.SINCOS=1
path_log= "/home/pasquale/Desktop/thesis/thesis-code/1D_pendulum/stable_baselines/"
robot.setupSim()
h_sum_last =0
h_mean_last_list = []
first_step_up_list = []
for j in range (20):
    up_reach = False
    robot.resetRobot()
    for i in range(NSTEPS):
        
        x = np.array([[robot.states_sincos[1][0],robot.states_sincos[1][1],robot.states_dot[1][3]]])    
        action =sess.run(policy.policy, feed_dict={ policy.x: x }) 
        action=action.tolist()
        robot.simulateDyn(action[0])
        
            
        if angle_normalize(robot.states[1][3]) < 1*np.pi/180 and up_reach == False:
            first_step_up = i
            up_reach = True
            
        if i >= NSTEPS -1:
            h_sum_last+= -angle_normalize(robot.states[1][3])**2

    h_mean_last_list.append(h_sum_last/NSTEPS)
    first_step_up_list.append(first_step_up)
    first_step_up = 10000
    print(j)
    
print(" (RANDSET) mean return 20 last episodes: "+str(sum(h_mean_last_list)/len(h_mean_last_list))+", first reached top at "+str(sum(first_step_up_list)/len(first_step_up_list)))

robot.stopSim()

#(RANDSET) mean return 20 last episodes: -4.1383060313359755e-05, first reached top at 17.2