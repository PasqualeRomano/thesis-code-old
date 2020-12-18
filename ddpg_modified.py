'''
Deep actor-critic network, 
From "Continuous control with deep reinforcement learning", by Lillicrap et al, arXiv:1509.02971
'''


import tensorflow as tf
import numpy as np,math,unittest
from tensorflow import keras
from tensorflow.keras import layers
import random
from collections import deque
import time
import signal
import matplotlib.pyplot as plt
from robot_bullet import Robot
import json
import training_config as tc
import os


tf.compat.v1.disable_eager_execution()
RANDOM_SEED = int((time.time()%10)*1000)
print("Seed = %d" %  RANDOM_SEED)
np .random.seed     (RANDOM_SEED)
tf.compat.v1.set_random_seed(RANDOM_SEED)
random.seed         (RANDOM_SEED)
#n_init = tf.keras.initializers.RandomUniform(seed=RANDOM_SEED)
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
range_esp               = tc.range_esp                    # Hidden layer size
time_step               = tc.time_step


SIM_NUMBER = 0

reward_weights  = [1.,0.0,0.00]


### Definining environment ###
#
# Acrobot
#   fixed joint in [0.,0.,0.]
#       first revolute [0.0060872, 0., 0.035]
#           second revolute [0.023,0.,0.1]
#               center of gravity of link 2 [-0.0050107, 1.9371E-10, 0.10088]
#
# since the rotation is on x and all joints RF have the same orientation and also link2 CoG, consider only y and z for the length (actually only z since y is always 0)
#
# tip of Acrobot is considered to be in the center of gravity of link2
# -> goal position 

length = .035 + .01 + 0.1088
goal = np.array([0,0,length])





env                 = Robot("double_pendulum.urdf")       
env_rend            = Robot("double_pendulum.urdf",sim_number=SIM_NUMBER) #for rendering


env.GUI_ENABLED=0 
env.RANDSET = 0
env.SINCOS = 1
env.time_step= time_step
env.actuated_index =[2]
env.max_torque=5.0

env_rend.SINCOS = 1
env_rend.GUI_ENABLED = 1
env_rend.actuated_index = [2]






step_expl = 0.
epi_expl = 0.

NX                  = 6         
NU                  = 1          

def angle_normalize(x):
    return min(x%(2*np.pi),abs(x%(2*np.pi)-2*np.pi))


## Build Neural Networks for Actor and Critic ##


class QValueNetwork:
    def __init__(self):
        nvars           = len(tf.compat.v1.trainable_variables())

        x       =  keras.Input(shape=(NX,),name="State")
        u       =  keras.Input(shape=(NU,),name="Control")

        netx1 = layers.Dense(NH1, activation="relu", kernel_initializer=tf.keras.initializers.RandomUniform(seed=RANDOM_SEED,minval = -1/math.sqrt(NH1),maxval=1/math.sqrt(NH1)), name="netx1")(x)
        net_u = tf.keras.layers.Concatenate(axis=-1)([netx1, u])
        netx2 = layers.Dense(NH2, activation="relu", kernel_initializer=tf.keras.initializers.RandomUniform(seed=RANDOM_SEED,minval = -1/math.sqrt(NH2),maxval=1/math.sqrt(NH2)), name="netx2")(net_u)
        
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
        net = layers.Dense(NH1, activation="relu", kernel_initializer=tf.keras.initializers.RandomUniform(seed=RANDOM_SEED,minval = -1/math.sqrt(NH1),maxval=1/math.sqrt(NH1)), name="net")(x)
        net = layers.Dense(NH2, activation="relu", kernel_initializer=tf.keras.initializers.RandomUniform(seed=RANDOM_SEED,minval = -1/math.sqrt(NH2),maxval=1/math.sqrt(NH2)), name="net2")(net)
        policy = layers.Dense(NU, activation="tanh", kernel_initializer=u_init, name="netu1")(net)*env.max_torque #2nm max torque

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
if __name__ == "__main__": 
    replayDeque = deque()
 
 
 
    policy          = PolicyNetwork(). setupOptim()
    policyTarget    = PolicyNetwork(). setupTargetAssign(policy)
 
    qvalue          = QValueNetwork(). setupOptim()
    qvalueTarget    = QValueNetwork(). setupTargetAssign(qvalue)
    
    
    model_save = "/home/pasquale/Desktop/thesis/thesis-code/2D_Acrobot/ddpg/trined_agents/DDPG_saved_"+str(SIM_NUMBER)+".chkpt"
    
    sess            = tf.compat.v1.InteractiveSession()
    tf.compat.v1.global_variables_initializer().run()
    tf_saver = tf.compat.v1.train.Saver(tf.compat.v1.global_variables())

    # env_rend.SINCOS = 1
    # env_rend.GUI_ENABLED = 1
    # env_rend.setupSim() 

    ##  Function for rendering trial    ##
    
    def rendertrial(env,maxiter=NSTEPS,verbose=True):
        
        
        
        env.resetRobot()
        x = np.array([[env.states_sincos[1][0],env.states_sincos[1][1],env.states_dot[1][3],
                       env.states_sincos[2][0],env.states_sincos[2][1],env.states_dot[2][3]]])
        rsum = 0.
        for i in range(maxiter):
            u = sess.run(policy.policy, feed_dict={ policy.x: x }) 
            env_rend.simulateDyn([u[0][0]])
            x = np.array([[env.states_sincos[1][0],env.states_sincos[1][1],env.states_dot[1][3],
                           env.states_sincos[2][0],env.states_sincos[2][1],env.states_dot[2][3]]])
            reward =  -np.linalg.norm(goal-np.array([env.states[2][0],env.states[2][1],env.states[2][2]])) 
            #print(reward)
            time.sleep(1e-1)
            rsum += reward
            #print(rsum)
        print(x)
        if verbose: print('Lasted ',i,' timestep -- total reward:',rsum)
    
    signal.signal(signal.SIGTSTP, lambda x,y:rendertrial()) # Roll-out when CTRL-Z is pressed
 

    h_rwd = []
    h_qva = []
    h_ste = []    
 
 
    ##  Envirnonment Configuration ##
    
    env.setupSim() 
    
    
    
    ##  Training Loop ##
    start_time = time.time()
    for episode in range(1,NEPISODES):
        env.resetRobot()
        x = np.array([[env.states_sincos[1][0],env.states_sincos[1][1],env.states_dot[1][3],
                       env.states_sincos[2][0],env.states_sincos[2][1],env.states_dot[2][3]]])
        
        rsum = 0.0 
 
        for step in range(NSTEPS):
            u       = sess.run(policy.policy, feed_dict={ policy.x: x }) # Greedy policy ...
            #u      += np.random.uniform(-range_esp,range_esp) / (1. + episode/epi_expl + step/step_expl )   #add gaussian noise                      # ... with noise
            u      += np.random.normal(loc=0.0,scale=range_esp)
            
            #print(u[0][0])
            env.simulateDyn([u[0][0]])
            x2 = np.array([env.states_sincos[1][0],env.states_sincos[1][1],env.states_dot[1][3],
                            env.states_sincos[2][0],env.states_sincos[2][1],env.states_dot[2][3]])   
            #print(x2)
            r = -np.linalg.norm(goal-np.array([env.states[2][0],env.states[2][1],env.states[2][2]])) 
            done    = False                                              # pendulum scenario is endless.
            #print(r)
            replayDeque.append(ReplayItem(x,u,r,done,x2))                # Feed replay memory ...
            if len(replayDeque)>REPLAY_SIZE: replayDeque.popleft()       # ... with FIFO forgetting.

            rsum   += r
            if done or np.linalg.norm(x-x2)<1e-3: break                  # Break when pendulum is still.
            x       = [x2]

        # Start optimizing networks when memory size > batch size.
            if len(replayDeque) > BATCH_SIZE:     
                batch = random.sample(replayDeque,BATCH_SIZE)            # Random batch from replay memory.
                x_batch    = np.vstack([ b.x      for b in batch ])
                u_batch    = np.vstack([ b.u      for b in batch ])
                r_batch    = np.vstack([ b.reward for b in batch ])
                d_batch    = np.vstack([ b.done   for b in batch ])
                x2_batch   = np.vstack([ b.x2     for b in batch ])

            # Compute Q(x,u) from target network
                u2_batch   = sess.run(policyTarget.policy, feed_dict={ policyTarget .x : x2_batch})
                q2_batch   = sess.run(qvalueTarget.qvalue, feed_dict={ qvalueTarget.x : x2_batch,
                                                                   qvalueTarget.u : u2_batch })
                qref_batch = r_batch + (d_batch==False)*(DECAY_RATE*q2_batch)

            # Update qvalue to solve HJB constraint: q = r + q'
                sess.run(qvalue.optim, feed_dict={ qvalue.x    : x_batch,
                                                   qvalue.u    : u_batch,
                                                   qvalue.qref : qref_batch })

            # Compute approximate policy gradient ...
                u_targ  = sess.run(policy.policy,   feed_dict={ policy.x        : x_batch} )
                qgrad   = sess.run(qvalue.gradient, feed_dict={ qvalue.x        : x_batch,
                                                            qvalue.u        : u_targ })
            # ... and take an optimization step along this gradient.
                sess.run(policy.optim,feed_dict= { policy.x         : x_batch,
                                               policy.qgradient : qgrad })

            # Update target networks by homotopy.
                sess.run(policyTarget. update_variables)
                sess.run(qvalueTarget.update_variables)

        # \\\END_FOR step in range(NSTEPS)
    

    
    ## Display and logging (not mandatory).
        maxq = np.max( sess.run(qvalue.qvalue,feed_dict={ qvalue.x : x_batch,
                                                          qvalue.u : u_batch }) ) \
                                                          if 'x_batch' in locals() else 0
        print('Ep#{:3d}: lasted {:d} steps, reward={:3.0f}, max qvalue={:2.3f}' \
            .format(episode, step,rsum, maxq))
        h_rwd.append(rsum) 
        h_qva.append(maxq)
        h_ste.append(step)
        # env_rend.setupSIm()
        # if not (episode+1) % 15:     rendertrial(env_rend)

    # \\\END_FOR episode in range(NEPISODES)
    end_time=time.time()
    elapsed_time = end_time-start_time
    print('elapsed '+str(elapsed_time)+'s')
    env.stopSim()

    print("Average reward during trials: %.3f" % (sum(h_rwd)/NEPISODES))

    env_rend.SINCOS = 1
    env_rend.GUI_ENABLED = 1
    env_rend.time_step = time_step
    env_rend.setupSim()
    env_rend.video_path = "/home/pasquale/Desktop/thesis/thesis-code/2D_Acrobot/ddpg/Video"
    env_rend.LOGDATA=1   ####@@@@@@@@@@@@@@@@############@@@@@@@@@@@@@@@@@@@@#############@@@@@@@@
    rendertrial(env_rend)
    env_rend.stopSim()



    ##   SAVE DATA  ##
    filepath = '/home/pasquale/Desktop/thesis/thesis-code/2D_Acrobot/ddpg/'
    

    f=open(filepath + 'hrwd{}.txt'.format(SIM_NUMBER), 'w')
    f.write(json.dumps(h_rwd))
    f.close()
    
    
    

    f=open(filepath + 'config{}.txt'.format(SIM_NUMBER), 'w')
    f.write("NEPISODES = "+str(NEPISODES)+"\nNSTEPS = "+str(NSTEPS)+"\nQVALUE_LEARNING_RATE = "+str(QVALUE_LEARNING_RATE)+"\nPOLICY_LEARNING_RATE = "+str(POLICY_LEARNING_RATE)+"\nDECAY_RATE = "+str(DECAY_RATE)+"\nUPDATE_RATE = "+str(UPDATE_RATE)+"\nREPLAY_SIZE"+str(REPLAY_SIZE)+"\nBATCH_SIZE"+str(BATCH_SIZE)+"\nNH1 = "+str(NH1)+"\nNH2 = "+str(NH2) + "\nreward weights = "+str(0)
           +"\nRANDOM RESET = "+str(env.RANDSET)+"\nstep_expl = "+ str(0)+"\nepi_expl = "+ str(0)+"\nrange_esp = "+ str(range_esp)+"\nElapsed time = "+str(elapsed_time)+"\nMean reward (20 eps) = "+str(0)+"\nStd reward = "+str(0))
    f.close() 

#confronta 
#convergenza
#tempo di training
#average reward
#valuta policy con (10x)random reset 
#salvare a che step arriva in posizione verticale (anche con random reset)


    os.system('spd-say "your program has finished"')
    
    plt.plot( np.cumsum(h_rwd)/list(range(1,NEPISODES)))
    plt.grid(True)
    #plt.show()
    plt.savefig(filepath + 'reward{}.png'.format(SIM_NUMBER))
    

    tf_saver.save(sess, model_save)



