import gym
import time
import numpy as np
from robot_bullet import Robot
#from stable_baselines.ddpg.policies import MlpPolicy #multilayerperceptor type of neural netwrok (fully connected
from custom_policy_stable_baselines import CustomPolicy_2,CustomPolicy_4
from stable_baselines.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise, AdaptiveParamNoiseSpec
from stable_baselines.common.evaluation import evaluate_policy
from stable_baselines import DDPG
from custom_env_stable_baselines import PendulumPyB
import training_config as tc
import os



env = PendulumPyB()

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

SIM_NUMBER = 1.2
##Training policies
#CustomPolicy_3 
#CustomPolicy_2  Standard mlp stable baselines policy with modified layer-size
#CustomPolicy_4  Modified initialization of layers and layer-size
policy = CustomPolicy_4




env.episode_duration = NSTEPS
## the noise objects for DDPG
n_actions = env.action_space.shape[-1]
param_noise = None
action_noise =  NormalActionNoise(0,range_esp)
#
model = DDPG(policy, env, verbose=1,nb_train_steps=NSTEPS, nb_rollout_steps=NSTEPS,nb_eval_steps=NSTEPS,gamma=DECAY_RATE, param_noise=None, action_noise=action_noise,batch_size=BATCH_SIZE,actor_lr=POLICY_LEARNING_RATE,
               critic_lr =  QVALUE_LEARNING_RATE,buffer_size=REPLAY_SIZE,tau= UPDATE_RATE)


# mean_reward, std_reward = evaluate_policy(model, env_eval, n_eval_episodes=10)
# print(f'Mean reward: {mean_reward} +/- {std_reward:.2f}')

start_time = time.time()
model.learn(total_timesteps=NSTEPS*NEPISODES)
end_time=time.time()
elapsed_time = end_time-start_time
model.save("ddpg_pendulum_stb_baselines_"+str(SIM_NUMBER))
print('elapsed '+str(elapsed_time)+'s')
mean_reward,std_reward = evaluate_policy(model,env,n_eval_episodes = 21)
env.robot.stopSim()


#mean_reward, std_reward = evaluate_policy(model, env_eval, n_eval_episodes=10)
#print(f'Mean reward: {mean_reward} +/- {std_reward:.2f}')

  
### save video
# model = DDPG.load("ddpg_pendulum_stb_baselines")

robot = Robot("single_pendulum.urdf")
robot.sim_number=SIM_NUMBER
RANDSET =0
robot.LOGDATA = 1
robot.SINCOS=1
robot.video_path = "/home/pasquale/Desktop/thesis/thesis-code/1D_pendulum/stable_baselines/Video"
path_log= "/home/pasquale/Desktop/thesis/thesis-code/1D_pendulum/stable_baselines/"
robot.time_step = time_step
robot.setupSim()
for i in range(NSTEPS):
        
         obs = np.array([robot.states_sincos[1][0],robot.states_sincos[1][1],robot.states_dot[1][3]])
         action, _states = model.predict(obs)
         action=action.tolist()
         robot.simulateDyn(action)
         time.sleep(0.05)
robot.stopSim()  

#Evaluate policy 
#env.robot.stopSim()
#env = PendulumPyB()



f=open(path_log + 'baselines_config{}.txt'.format(robot.sim_number), 'w')
f.write("NEPISODES = "+str(NEPISODES)+"\nNSTEPS = "+str(NSTEPS)+"\nQVALUE_LEARNING_RATE = "+str(QVALUE_LEARNING_RATE)+"\nPOLICY_LEARNING_RATE = "+str(POLICY_LEARNING_RATE)+"\nDECAY_RATE = "+str(DECAY_RATE)+"\nUPDATE_RATE = "+str(UPDATE_RATE)+"\nREPLAY_SIZE"+str(REPLAY_SIZE)+"\nBATCH_SIZE"+str(BATCH_SIZE)+"\nNH1 = "+str(NH1)+"\nNH2 = "+str(NH2) + "\nreward weights = "+str(0)
           +"\nRANDOM RESET = "+str(RANDSET)+"\nstep_expl = "+ str(0)+"\nepi_expl = "+ str(0)+"\nrange_esp = "+ str(range_esp)+"\nElapsed time = "+str(elapsed_time)+"\nMean reward (20 eps) = "+str(mean_reward)+"\nStd reward = "+str(std_reward)+"\nPolicy = "+str(policy))
f.close() 

#confronta 
#convergenza
#tempo di training
#average reward
#valuta policy con (10x)random reset 
#salvare a che step arriva in posizione verticale (anche con random reset)


os.system('spd-say "your program has finished you motherfucker"')