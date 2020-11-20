import gym
import time
import numpy as np
from robot_bullet import Robot
from stable_baselines.ddpg.policies import MlpPolicy #multilayerperceptor type of neural netwrok (fully connected)
from stable_baselines.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise, AdaptiveParamNoiseSpec
from stable_baselines.common.evaluation import evaluate_policy
from stable_baselines import DDPG
from custom_env_stable_baseline import PendulumPyB
import training_config as tc

env = PendulumPyB()

def angle_normalize(x):
    return min(x%(2*np.pi),abs(x%(2*np.pi)-2*np.pi))

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

model = DDPG.load("ddpg_pendulum_stb_baselines")

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






# f=open(path_log + 'baselines_config{}.txt'.format(robot.sim_number), 'w')
# f.write("NEPISODES = "+str(NEPISODES)+", NSTEPS = "+str(NSTEPS)+", QVALUE_LEARNING_RATE = "+str(QVALUE_LEARNING_RATE)+", POLICY_LEARNING_RATE = "+str(POLICY_LEARNING_RATE)+", DECAY_RATE = "+str(DECAY_RATE)+", UPDATE_RATE = "+str(UPDATE_RATE)+", REPLAY_SIZE"+str(REPLAY_SIZE)+", BATCH_SIZE"+str(BATCH_SIZE)+", NH1 = "+str(NH1)+", NH2 = "+str(NH2) + ",reward weights = "+str(reward_weights)
#            +"RANDOM RESET = "+str(RANDSET)+"step_expl = "+ str(0)+"epi_expl = "+ str(0)+"range_esp = "+ str(range_esp))
# f.close() 
#confronta 
#convergenza
#tempo di training
#average reward
#valuta policy con (10x)random reset 
#salvare a che step arriva in posizione verticale (anche con random reset)