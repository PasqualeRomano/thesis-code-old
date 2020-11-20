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
import dill
import training_config as tc

dill.load(open("DDPG_continuous.pickle"))
    
sess            = tf.compat.v1.InteractiveSession()
tf.compat.v1.global_variables_initializer().run()    
    
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

robot = Robot("single_pendulum.urdf")
robot.sim_number=1
RANDSET =0
robot.GUI_ENABLED = 0
robot.SINCOS=1
path_log= "/home/pasquale/Desktop/thesis/thesis-code/1D_pendulum/stable_baselines/"
robot.setupSim()

#Evaluate policy 
#env.robot.stopSim()
#env = PendulumPyB()

#Check convergence
up_reach = False
h_sum_last =0
for i in range(NSTEPS):
    
    np.array([[env_rend.states_sincos[1][0],env_rend.states_sincos[1][1],env_rend.states_dot[1][3]]])    
    obs = np.array([robot.states_sincos[1][0],robot.states_sincos[1][1],robot.states_dot[1][3]])
    action, _states = model.predict(obs)
    action=action.tolist()
    robot.simulateDyn(action)
    time.sleep(0.1)
         
    if angle_normalize(robot.states[1][3]) < 1*np.pi/180 and up_reach == False:
         first_step_up = i
         up_reach = True
         
    if i >= NSTEPS -1:
        h_sum_last+= -angle_normalize(robot.states[1][3])**2

h_mean_last = h_sum_last/NSTEPS
print("mean return 20 last episodes: "+str(h_mean_last)+", first reached top at "+str(first_step_up))

robot.stopSim()  