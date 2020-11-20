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


env.episode_duration = NSTEPS
## the noise objects for DDPG
n_actions = env.action_space.shape[-1]
param_noise = None
action_noise =  NormalActionNoise(0,range_esp)
#
model = DDPG(MlpPolicy, env, verbose=1,nb_train_steps=NSTEPS, nb_rollout_steps=NSTEPS,nb_eval_steps=NSTEPS,gamma=DECAY_RATE, param_noise=None, action_noise=action_noise,batch_size=BATCH_SIZE,actor_lr=POLICY_LEARNING_RATE,
               critic_lr =  QVALUE_LEARNING_RATE,buffer_size=REPLAY_SIZE)


# mean_reward, std_reward = evaluate_policy(model, env_eval, n_eval_episodes=10)
# print(f'Mean reward: {mean_reward} +/- {std_reward:.2f}')

start_time = time.time()
model.learn(total_timesteps=NSTEPS*NEPISODES)
end_time=time.time()
elapsed_time = end_time-start_time
model.save("ddpg_pendulum_stb_baselines")
print('elapsed '+str(elapsed_time)+'s')
mean_reward,std_reward = evaluate_policy(model,env,n_eval_episodes = 20)
env.robot.stopSim()


#mean_reward, std_reward = evaluate_policy(model, env_eval, n_eval_episodes=10)
#print(f'Mean reward: {mean_reward} +/- {std_reward:.2f}')

  
### save video
# model = DDPG.load("ddpg_pendulum_stb_baselines")

robot = Robot("single_pendulum.urdf")
robot.sim_number=1
RANDSET =0
robot.LOGDATA = 1
robot.SINCOS=1
robot.video_path = "/home/pasquale/Desktop/thesis/thesis-code/1D_pendulum/stable_baselines/Video"
path_log= "/home/pasquale/Desktop/thesis/thesis-code/1D_pendulum/stable_baselines/"
robot.setupSim()
for i in range(NSTEPS):
        
         obs = np.array([robot.states_sincos[1][0],robot.states_sincos[1][1],robot.states_dot[1][3]])
         action, _states = model.predict(obs)
         action=action.tolist()
         robot.simulateDyn(action)
         time.sleep(0.1)
robot.stopSim()  

#Evaluate policy 
#env.robot.stopSim()
#env = PendulumPyB()



f=open(path_log + 'baselines_config{}.txt'.format(robot.sim_number), 'w')
f.write("NEPISODES = "+str(NEPISODES)+", NSTEPS = "+str(NSTEPS)+", QVALUE_LEARNING_RATE = "+str(QVALUE_LEARNING_RATE)+", POLICY_LEARNING_RATE = "+str(POLICY_LEARNING_RATE)+", DECAY_RATE = "+str(DECAY_RATE)+", UPDATE_RATE = "+str(UPDATE_RATE)+", REPLAY_SIZE"+str(REPLAY_SIZE)+", BATCH_SIZE"+str(BATCH_SIZE)+", NH1 = "+str(NH1)+", NH2 = "+str(NH2) + ",reward weights = "+str(0)
           +"RANDOM RESET = "+str(RANDSET)+"step_expl = "+ str(0)+"epi_expl = "+ str(0)+"range_esp = "+ str(range_esp))
# f.close() 
f=open(path_log + 'baselines_results{}.txt'.format(1), 'w')
f.write("Elapsed time = "+str(elapsed_time)+", Mean reward (20 eps) = "+str(mean_reward)+",Std reward = "+str(std_reward) )
f.close()
#confronta 
#convergenza
#tempo di training
#average reward
#valuta policy con (10x)random reset 
#salvare a che step arriva in posizione verticale (anche con random reset)