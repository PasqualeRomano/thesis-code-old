import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from os import path
from robot_bullet import Robot


class PendulumPyB(gym.Env):

    metadata = {'render.modes': ['human']}

    def __init__(self):
        self.max_speed = 8
        self.max_torque = 2.
        #self.dt = .05
        #self.viewer = None
        
        self.robot = Robot("single_pendulum.urdf")
        self.robot.GUI_ENABLED = 0
        self.robot.setupSim()
        self.robot.SINCOS = 1
        self.reward_weights  = [1.,0.0,0.0]
        
        
        
        

        high = np.array([1., 1., self.max_speed], dtype=np.float32)
        self.action_space = spaces.Box(
            low=-self.max_torque,
            high=self.max_torque, shape=(1,),
            dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=-high,
            high=high,
            dtype=np.float32
        )



    def step(self, u):
       

        
        #dt = self.dt

        #u = np.clip(u, -self.max_torque, self.max_torque)[0]
        self.robot.simulateDyn([u])
        reward = -angle_normalize(self.robot.states[1][3])**2*self.reward_weights[0]-self.robot.states_dot[1][3]**2*self.reward_weights[1] - self.reward_weights[2] * (u ** 2)


        return self._get_obs(), reward, False, {}

    def reset(self):
        self.robot.resetRobot()
        return self._get_obs()

    def _get_obs(self):
        self.robot.observeState()
        return np.array([self.robot.states_sincos[1][0], self.robot.states_sincos[1][1],self.robot.states_dot[1][3]])
    

    """ def render(self, mode='human'):
        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(500, 500)
            self.viewer.set_bounds(-2.2, 2.2, -2.2, 2.2)
            rod = rendering.make_capsule(1, .2)
            rod.set_color(.8, .3, .3)
            self.pole_transform = rendering.Transform()
            rod.add_attr(self.pole_transform)
            self.viewer.add_geom(rod)
            axle = rendering.make_circle(.05)
            axle.set_color(0, 0, 0)
            self.viewer.add_geom(axle)
            fname = path.join(path.dirname(__file__), "assets/clockwise.png")
            self.img = rendering.Image(fname, 1., 1.)
            self.imgtrans = rendering.Transform()
            self.img.add_attr(self.imgtrans)

        self.viewer.add_onetime(self.img)
        self.pole_transform.set_rotation(self.state[0] + np.pi / 2)
        if self.last_u:
            self.imgtrans.scale = (-self.last_u / 2, np.abs(self.last_u) / 2)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array') """

    """ def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None """


def angle_normalize(x):
    return (((x+np.pi) % (2*np.pi)) - np.pi)