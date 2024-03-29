
import gym
import time
import numpy as np,pandas as pd,statistics as st
from robot_bullet import Robot
#from stable_baselines.ddpg.policies import MlpPolicy #multilayerperceptor type of neural netwrok (fully connected
from custom_policy_stable_baselines import CustomPolicy_2,CustomPolicy_4
from stable_baselines.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise, AdaptiveParamNoiseSpec
from stable_baselines.common.evaluation import evaluate_policy
from stable_baselines import DDPG
from custom_env_stable_baselines import PendulumPyB
import matplotlib.pyplot as plt,matplotlib.ticker 
import training_config as tc
import os


SAVE =0

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
time_step               = tc.time_step


SIM_NUMBER = 1.2

model = DDPG.load("ddpg_pendulum_stb_baselines_"+str(SIM_NUMBER))

robot = Robot("single_pendulum.urdf")
robot.sim_number=1
robot.RANDSET =0
robot.GUI_ENABLED = 1
robot.SINCOS=1
path_log= "/home/pasquale/Desktop/thesis/thesis-code/1D_pendulum/stable_baselines/"
robot.time_step = time_step
robot.setupSim()

#Evaluate policy 
#env.robot.stopSim()
#env = PendulumPyB()

#Check convergence
#confronta 
#convergenza
#tempo di training
#average reward

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
        
        
    
    obs = np.array([robot.states_sincos[1][0],robot.states_sincos[1][1],robot.states_dot[1][3]])
    action, _states = model.predict(obs)
    action=action.tolist()
    robot.simulateDyn(action)
    action_list.append(action)
    #time.sleep(0.01)
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
#mean return 20 last steps: -9.476149966369109e-05, first reached top at 64


#valuta policy con (10x)random reset 
#salvare a che step arriva in posizione verticale (anche con random reset)
robot = Robot("single_pendulum.urdf")
robot.sim_number=1
robot.RANDSET =1
robot.GUI_ENABLED = 1
robot.SINCOS=1
path_eval= "/home/pasquale/Desktop/thesis/thesis-code/1D_pendulum/stable_baselines/eval/"
robot.time_step = time_step
robot.setupSim()


up_reach= False


h_mean_last_list = []
first_step_up_list = []
max_after_up_list = []





for k in range (100):
    up_reach = False
    robot.resetRobot()
    for i in range(NSTEPS):
            
        obs = np.array([robot.states_sincos[1][0],robot.states_sincos[1][1],robot.states_dot[1][3]])
        action, _states = model.predict(obs)
        action=action.tolist()
        robot.simulateDyn(action)
        #time.sleep(0.001)
             
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
    
    
print(" (RANDSET) mean return 100 last episodes: "+str(sum(h_mean_last_list)/len(h_mean_last_list))+", first reached top at "+str(sum(first_step_up_list)/len(first_step_up_list)))

robot.stopSim()  
#(RANDSET) mean return 20 last episodes: -0.0001268793446133105, first reached top at 21.05
if SAVE:

    f=open(path_eval + 'results_baselines.txt', 'w')
    f.write("up position reached at step"+str(first_step_up_1)+",mean reward last steps after up reached: "+str(h_mean_last)+", angle is lower than "+str(max_after_up)+"in the last 50 steps"
        +"\n (RANDSET) mean 100 last episodes: up step "+str(mean_first_step_up_list)+"+-" +str(std_first_step_up_list)+",return afer "+str(mean_h_mean_last)+"+-" +str(std_mean_h_mean_last)+" ,max angle"+str(mean_max_after_up_list)+"+-" +str(std_max_after_up_list))
    #            
    f.close() 

    action_sav = np.array(action_list).T[0].tolist()
    pd.DataFrame([angles_list,vel_ang_list,action_sav]).T.to_csv(path_eval + 'ang_vel_act_seq.csv')

#confronta 
#convergenza
#tempo di training
#average reward
#valuta policy con (10x)random reset 
#salvare a che step arriva in posizione verticale (anche con random reset)

torque_map = list(np.arange(0,6,0.01))

ik = 0
for i in list(np.arange(0,6,0.01)):
    
    row = []
    for j in np.arange(-0.7*np.pi,0.7*np.pi,0.01*np.pi):
        
        
        obs = np.array([np.sin(i*np.pi/180),np.cos(i*np.pi/180),j])
        action, _states = model.predict(obs)
        row+= action.tolist()
        
    torque_map[ik] = row
    ik+=1
    
    

Z = np.array(torque_map)  #np.random.rand(6, 10)
x = np.arange(-0.7*np.pi,0.7*np.pi,0.01*np.pi)  
y = np.arange(0,6,0.01)

# fig, ax = plt.subplots()
# ax.pcolormesh(x, y, Z)
# plt.show()



levels = matplotlib.ticker.MaxNLocator(nbins=15).tick_values(Z.min(), Z.max())


# pick the desired colormap, sensible levels, and define a normalization
# instance which takes data values and translates those into levels.
cmap = plt.get_cmap('PiYG')
norm = matplotlib.colors.BoundaryNorm(levels, ncolors=cmap.N, clip=True)

fig, ax0 = plt.subplots()
ax0.set_ylabel('angle [deg]')
ax0.set_xlabel('angular velocity [rad/s]')
im = ax0.pcolormesh(x, y, Z, cmap=cmap, norm=norm)
fig.colorbar(im, ax=ax0,label = "Torque [Nm]")
ax0.set_title('Agent Implementation 2')


# contours are *point* based plots, so convert our bound into point
# centers
# cf = ax1.contourf(x,
#                   y, Z, levels=levels,
#                   cmap=cmap)
# fig.colorbar(cf, ax=ax1)
# ax1.set_title('contourf with levels')

# adjust spacing between subplots so `ax1` title and `ax0` tick labels
# don't overlap
#fig.tight_layout()
plt.savefig('/home/pasquale/Desktop/thesis/thesis-code/plots/plot_stbs_ddpg_policy_1Dp.eps', format='eps')
plt.show()