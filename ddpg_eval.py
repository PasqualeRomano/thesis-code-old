import tensorflow as tf
import numpy as np, pandas as pd
import csv
from tensorflow import keras
from tensorflow.keras import layers 
import random
from collections import deque
import time
import signal
import matplotlib.pyplot as plt
from robot_bullet import Robot
import training_config as tc
import statistics as st



tf.compat.v1.disable_eager_execution()
RANDOM_SEED = int((time.time()%10)*1000)
print("Seed = %d" %  RANDOM_SEED)
np .random.seed     (RANDOM_SEED)
tf.compat.v1.set_random_seed(RANDOM_SEED)
random.seed         (RANDOM_SEED)
n_init = tf.keras.initializers.TruncatedNormal(seed=RANDOM_SEED)
u_init = tf.keras.initializers.RandomUniform(minval=-3e-3, maxval=3e-3, seed=RANDOM_SEED) #check weights #######


NEPISODES               = tc.NEPISODES                  # Max training steps
NSTEPS                  = tc.NSTEPS                     # Max episode length 
QVALUE_LEARNING_RATE    = tc.QVALUE_LEARNING_RATE       # Base learning rate for the Q-value Network
POLICY_LEARNING_RATE    = tc.POLICY_LEARNING_RATE       # Base learning rate for the policy network
DECAY_RATE              = tc.DECAY_RATE                 # Discount factor 
UPDATE_RATE             = tc.UPDATE_RATE                # Homotopy rate to update the networks
REPLAY_SIZE             = tc.REPLAY_SIZE                # Size of replay buffer
BATCH_SIZE              = tc.BATCH_SIZE                 # Number of points to be fed in stochastic gradient
NH1                     = tc.NH1
NH2                     = tc.NH2                        # Hidden layer size
range_esp               = tc.range_esp
time_step               = tc.time_step

reward_weights  = [1.,0.0,0.00]


sim_number = 2
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
        net_u = tf.keras.layers.Concatenate(axis=-1)([netx1, u])
        netx2 = layers.Dense(NH2, activation="relu", kernel_initializer=n_init, name="netx2")(net_u)
        
        #netu1 = layers.Dense(NH1, activation="linear", kernel_initializer=n_init, name="netu1")(u)
        #netu2 = layers.Dense(NH2, activation="linear", kernel_initializer=n_init, name="netu2")(netu1)
        #net_act = tf.keras.activations.relu(netx2+netu2)
        
        qvalue = layers.Dense(1, activation="linear", kernel_initializer=u_init, name="qvalue")(netx2)
        
        qvalue_model = keras.Model(
            inputs=[x,u],
            outputs=[qvalue],
        )

        self.x          = x                                # Network state   <x> input in Q(x,u)
        self.u          = u                                # Network control <u> input in Q(x,u)
        self.qvalue     = qvalue                           # Network output  <Q>
        self.variables  = tf.compat.v1.trainable_variables()[nvars:] # Variables to be trained
        self.hidens = [ netx1, netx2 ]                  # Hidden layers for debug
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
        policy = layers.Dense(NU, activation="tanh", kernel_initializer=u_init, name="netu1")(net)*2. #2nm max torque

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
    
SIM_NUMBER = 1.1    
model_save = "DDPG_saved_"+str(SIM_NUMBER)+".chkpt"


sess            = tf.compat.v1.InteractiveSession()
tf.compat.v1.global_variables_initializer().run()
tf.compat.v1.train.Saver().restore(sess,model_save)



robot = Robot("single_pendulum.urdf")
robot.sim_number=SIM_NUMBER
robot.RANDSET =0
robot.GUI_ENABLED = 0
robot.SINCOS=1
path_log= "/home/pasquale/Desktop/thesis/thesis-code/1D_pendulum/continuous/"
robot.time_step =time_step
robot.setupSim()

#Evaluate policy 
#env.robot.stopSim()
#env = PendulumPyB()

#Check convergence

c = 100000000


h_sum_last =0



angles_list=[]
angles_list.append(angle_normalize(robot.states[1][3])*180/np.pi)
vel_ang_list = []
vel_ang_list.append(robot.states_dot[1][3])
action_list = []


up_reach = False
j=0
max_after_up = 0

for i in range(NSTEPS):
    
    x = np.array([[robot.states_sincos[1][0],robot.states_sincos[1][1],robot.states_dot[1][3]]])    
    action =sess.run(policy.policy, feed_dict={ policy.x: x }) 
    action=action.tolist()
    robot.simulateDyn(action[0])
    action_list.append(action)
    
    
    if angle_normalize(robot.states[1][3]) < 1*np.pi/180 and up_reach == False:
            first_step_up_1 = i
            up_reach = True
         
    if angle_normalize(robot.states[1][3])**2<c:  
        c = angle_normalize(robot.states[1][3])**2 
        step_max = i
         
    if up_reach == True:
        j+=1
        h_sum_last+= -angle_normalize(robot.states[1][3])**2
       
            
    if i >= NSTEPS-50:
        if  angle_normalize(robot.states[1][3])*180/np.pi > max_after_up:
            max_after_up = angle_normalize(robot.states[1][3])*180/np.pi
        
        
    angles_list.append(angle_normalize(robot.states[1][3])*180/np.pi)
    vel_ang_list.append(robot.states_dot[1][3])
    
h_mean_last = h_sum_last/j
print("up position reached at step"+str(first_step_up_1)+",mean reward last steps after up reached: "+str(h_mean_last)+", angle is lower than "+str(max_after_up)+"in the last 50 steps")

robot.stopSim()  

#mean return 20 last steps: -2.2419180192174818e-07, first reached top at 30

#valuta policy con (10x)random reset 
#salvare a che step arriva in posizione verticale (anche con random reset)



robot = Robot("single_pendulum.urdf")
robot.sim_number=1
robot.RANDSET =1
robot.GUI_ENABLED = 1
robot.SINCOS=1
path_eval= "/home/pasquale/Desktop/thesis/thesis-code/1D_pendulum/ddpg/eval/"
robot.time_step=time_step
robot.setupSim()


up_reach= False


h_mean_last_list = []
first_step_up_list = []
max_after_up_list = []




for k in range (100):
    
    h_sum_last =0
    firs_step_up = None
    max_after_up = 0
    j=0
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
            
        if i >= NSTEPS -50:
            if  angle_normalize(robot.states[1][3])*180/np.pi > max_after_up:
                max_after_up = angle_normalize(robot.states[1][3])*180/np.pi
           
        if up_reach == True:   
            
            j+=1
            h_sum_last+= -angle_normalize(robot.states[1][3])**2

    h_mean_last_list.append(h_sum_last/j)
    first_step_up_list.append(first_step_up)
    max_after_up_list.append(max_after_up)
    
    
    
mean_h_mean_last = st.mean(h_mean_last_list)
std_mean_h_mean_last= st.stdev(h_mean_last_list)

mean_first_step_up_list =  st.mean(first_step_up_list)
std_first_step_up_list = st.stdev(first_step_up_list)

mean_max_after_up_list = st.mean(max_after_up_list)
std_max_after_up_list= st.stdev(max_after_up_list)
    
print(" (RANDSET) mean 100 last episodes: up step "+str(mean_first_step_up_list)+"+-" +str(std_first_step_up_list)+",return afer "+str(mean_h_mean_last)+"+-" +str(std_mean_h_mean_last)+" ,max angle"+str(mean_max_after_up_list)+"+-" +str(std_max_after_up_list))

robot.stopSim()




f=open(path_eval + 'results_baselines.txt', 'w')
f.write("up position reached at step"+str(first_step_up_1)+",mean reward last steps after up reached: "+str(h_mean_last)+", angle is lower than "+str(max_after_up)+"in the last 50 steps"
        +"\n (RANDSET) mean 100 last episodes: up step "+str(mean_first_step_up_list)+"+-" +str(std_first_step_up_list)+",return afer "+str(mean_h_mean_last)+"+-" +str(std_mean_h_mean_last)+" ,max angle"+str(mean_max_after_up_list)+"+-" +str(std_max_after_up_list))
#            
f.close() 

action_sav = np.array(action_list).T[0][0].tolist()
pd.DataFrame([angles_list,vel_ang_list,action_sav]).T.to_csv(path_eval + 'ang_vel_act_seq.csv')