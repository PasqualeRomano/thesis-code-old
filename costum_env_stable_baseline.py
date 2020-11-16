import gym
import numpy as np
from gym import spaces
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import DDPG
from robot_bullet import Robot

N_DISCRETE_ACTIONS = 1

class CustomEnv(gym.Env):
  """Custom Environment that follows gym interface"""
  metadata = {'render.modes': ['human']}
  
  

  def __init__(self, arg1, arg2, ...):
    super(CustomEnv, self).__init__()
    
    
    
    
    self.pendulum = Robot("single_pendulum",[1],[1],0)    # Define action and observation space
    self.pendulum.SINCOS = 1
    self.pendulum.setupSim()
    # They must be gym.spaces objects
    # Example when using discrete actions:
    self.action_space = spaces.Box(low=-2.0, high=2.0,shape=None)
    # Example for using image as input:
    self.observation_space = spaces.Box(shape=(3,), dtype=np.float32)

  def step(self, action):
    
    reward_wheights  = [0.3,0.01]
    self.pendulum.simulateDyn([action])
    self.pendulum.observeState()
    observation = [self.pendulum.states_sincos[0][0],self.pendulum.states_sincos[0][1],self.pendulum.states_dot[3]]
    reward = (-np.square(self.pendulum.states[0][3]))*reward_wheights[0]-np.square(s.states_dot[0][3]*reward_wheights[1])
    
    return observation, reward
  def reset(self):
    self.pendulum.resetRobot()
    observation = [self.pendulum.states_sincos[0][0],self.pendulum.states_sincos[0][1],self.pendulum.states_dot[3]]
    return observation  # reward, done, info can't be included
  def render(self, mode='human'):
    ...
  def close (self):
    self.pendulum.stopSim()