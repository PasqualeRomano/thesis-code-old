import gym
import numpy as np
from stable_baselines.ddpg.policies import MlpPolicy #multilayerperceptor type of neural netwrok (fully connected)
from stable_baselines.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise, AdaptiveParamNoiseSpec
from stable_baselines.common.evaluation import evaluate_policy
from stable_baselines import DDPG

env = gym.make('Pendulum-v0')
env_eval= gym.make('Pendulum-v0')


NEPISODES               = 500           # Max training steps
NSTEPS                  = 100           # Max episode length
QVALUE_LEARNING_RATE    = 0.001         # Base learning rate for the Q-value Network
POLICY_LEARNING_RATE    = 0.0001        # Base learning rate for the policy network
DECAY_RATE              = 0.99          # Discount factor 
UPDATE_RATE             = 0.01          # Homotopy rate to update the networks
REPLAY_SIZE             = 10000         # Size of replay buffer
BATCH_SIZE              = 64            # Number of points to be fed in stochastic gradient
NH1 = NH2               = 250           # Hidden layer size

# the noise objects for DDPG
n_actions = env.action_space.shape[-1]
param_noise = None
action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=float(0.5) * np.ones(n_actions))

model = DDPG(MlpPolicy, env, verbose=1,nb_train_steps=NEPISODES, nb_rollout_steps=NSTEPS,nb_eval_steps=NSTEPS,gamma=DECAY_RATE, param_noise=None, action_noise=None,batch_size=BATCH_SIZE,actor_lr=POLICY_LEARNING_RATE,
             critic_lr =  QVALUE_LEARNING_RATE,buffer_size=REPLAY_SIZE)


mean_reward, std_reward = evaluate_policy(model, env_eval, n_eval_episodes=10)
print(f'Mean reward: {mean_reward} +/- {std_reward:.2f}')


model.learn(total_timesteps=NEPISODES*NSTEPS)
model.save("ddpg_pendulum")

mean_reward, std_reward = evaluate_policy(model, env_eval, n_eval_episodes=10)
print(f'Mean reward: {mean_reward} +/- {std_reward:.2f}')

""" del model # remove to demonstrate saving and loading

model = DDPG.load("ddpg_pendulum") """

obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()

