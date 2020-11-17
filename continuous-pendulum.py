'''
Deep actor-critic network, 
From "Continuous control with deep reinforcement learning", by Lillicrap et al, arXiv:1509.02971
'''


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
import json

tf.compat.v1.disable_eager_execution()
RANDOM_SEED = int((time.time()%10)*1000)
print("Seed = %d" %  RANDOM_SEED)
np .random.seed     (RANDOM_SEED)
tf.compat.v1.set_random_seed(RANDOM_SEED)
random.seed         (RANDOM_SEED)
n_init = tf.keras.initializers.TruncatedNormal(seed=RANDOM_SEED)
u_init = tf.keras.initializers.RandomUniform(minval=-2., maxval=2., seed=RANDOM_SEED)


NEPISODES               = 500           # Max training steps
NSTEPS                  = 200           # Max episode length
QVALUE_LEARNING_RATE    = 0.001         # Base learning rate for the Q-value Network
POLICY_LEARNING_RATE    = 0.0001        # Base learning rate for the policy network
DECAY_RATE              = 0.99          # Discount factor 
UPDATE_RATE             = 0.01           # Homotopy rate to update the networks
REPLAY_SIZE             = 10000         # Size of replay buffer
BATCH_SIZE              = 64            # Number of points to be fed in stochastic gradient
NH1 = NH2               = 250           # Hidden layer size

reward_wheights  = [1.,0.01,0.001]

sim_number = 16
RANDSET =0
env                 = Robot("single_pendulum.urdf")       
env_rend            = Robot("single_pendulum.urdf",sim_number=sim_number) #for rendering

env.RANDSET = 0


NX                  = 3          # ... training converges with q,qdot with 2x more neurones.
NU                  = 1            # Control is dim-1: joint torque

def angle_normalize(x):
    return x #((x+np.pi) % (2*np.pi)) - np.pi

class QValueNetwork:
    def __init__(self):
        nvars           = len(tf.compat.v1.trainable_variables())

        x       =  keras.Input(shape=(NX,),name="State")
        u       =  keras.Input(shape=(NU,),name="Control")

        netx1 = layers.Dense(NH1, activation="relu", kernel_initializer=n_init, name="netx1")(x)
        netx2 = layers.Dense(NH2, activation="linear", kernel_initializer=n_init, name="netx2")(netx1)
        netu1 = layers.Dense(NH1, activation="linear", kernel_initializer=n_init, name="netu1")(u)
        netu2 = layers.Dense(NH2, activation="linear", kernel_initializer=n_init, name="netu2")(netu1)
        net_act = tf.keras.activations.relu(netx2+netu2)
        qvalue = layers.Dense(1, activation="linear", kernel_initializer=u_init, name="qvalue")(net_act)
        
        qvalue_model = keras.Model(
            inputs=[x,u],
            outputs=[qvalue],
        )

        self.x          = x                                # Network state   <x> input in Q(x,u)
        self.u          = u                                # Network control <u> input in Q(x,u)
        self.qvalue     = qvalue                           # Network output  <Q>
        self.variables  = tf.compat.v1.trainable_variables()[nvars:] # Variables to be trained
        self.hidens = [ netx1, netx2, netu1, netu2 ]                  # Hidden layers for debug
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
        policy = layers.Dense(NU, activation="tanh", kernel_initializer=u_init, name="netu1")(net)*1 #1nm max torque

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

    sess            = tf.compat.v1.InteractiveSession()
    tf.compat.v1.global_variables_initializer().run()

    env_rend.SINCOS = 1
    env_rend.GUI_ENABLED = 1
    env_rend.setupSim() 


    def rendertrial(maxiter=NSTEPS,verbose=True):
    
        env_rend.resetRobot()
        x = np.array([[env_rend.states_sincos[1][0],env_rend.states_sincos[1][1],
                   env_rend.states_dot[1][3]]])
        rsum = 0.
        for i in range(maxiter):
            u = sess.run(policy.policy, feed_dict={ policy.x: x }) 
            env_rend.simulateDyn([u[0][0]])
            x=np.array([[env_rend.states_sincos[0][0],env_rend.states_sincos[0][1],
                       env_rend.states_dot[0][3]]])
            reward =  -np.square(angle_normalize(env_rend.states[1][3]))*reward_wheights[0]-np.square(env_rend.states_dot[1][3]*reward_wheights[1]) - reward_wheights[2] * (u[0][0] ** 2)
            time.sleep(1e-2)
            rsum += reward
        if verbose: print('Lasted ',i,' timestep -- total reward:',rsum)
    
    signal.signal(signal.SIGTSTP, lambda x,y:rendertrial()) # Roll-out when CTRL-Z is pressed
 

    h_rwd = []
    h_qva = []
    h_ste = []    
 
    env.GUI_ENABLED=0 
    env.SINCOS = 1
    env.setupSim() 
    for episode in range(1,NEPISODES):
        env.resetRobot()
        x    = np.array([[env.states_sincos[1][0],env.states_sincos[1][1],
                       env.states_dot[1][3]]])  #remove .T
        rsum = 0.0
 
        for step in range(NSTEPS):
            u       = sess.run(policy.policy, feed_dict={ policy.x: x }) # Greedy policy ...
            u      += 1. / (1. + episode + step)                         # ... with noise
            #print(u[0][0])
            env.simulateDyn([u[0][0]])
            x2 = np.array([env.states_sincos[1][0],env.states_sincos[1][1],
                       env.states_dot[1][3]])   
            r = -np.square(angle_normalize(env.states[1][3]))*reward_wheights[0]-np.square(env.states_dot[1][3]*reward_wheights[1]) - reward_wheights[2] * (u[0][0] ** 2)
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
    

    
    #Display and logging (not mandatory).
        maxq = np.max( sess.run(qvalue.qvalue,feed_dict={ qvalue.x : x_batch,
                                                          qvalue.u : u_batch }) ) \
                                                          if 'x_batch' in locals() else 0
        print('Ep#{:3d}: lasted {:d} steps, reward={:3.0f}, max qvalue={:2.3f}' \
            .format(episode, step,rsum, maxq))
        h_rwd.append(rsum) 
        h_qva.append(maxq)
        h_ste.append(step)
        if not (episode+1) % 15:     rendertrial()

    # \\\END_FOR episode in range(NEPISODES)
    env.stopSim()

    print("Average reward during trials: %.3f" % (sum(h_rwd)/NEPISODES))

    # env_rend.SINCOS = 1
    # env_rend.GUI_ENABLED = 1
    # env_rend.setupSim()
    env_rend.LOGDATA=1   ####@@@@@@@@@@@@@@@@############@@@@@@@@@@@@@@@@@@@@#############@@@@@@@@
    rendertrial()
    env_rend.stopSim()


    f=open('/home/pasquale/Desktop/thesis/thesis-code/1D_pendulum/hrwd{}.txt'.format(sim_number), 'w')
    f.write(json.dumps(h_rwd))
    f.close()

    f=open('/home/pasquale/Desktop/thesis/thesis-code/1D_pendulum/config{}.txt'.format(sim_number), 'w')
    f.write("NEPISODES = "+str(NEPISODES)+", NSTEPS = "+str(NSTEPS)+", QVALUE_LEARNING_RATE = "+str(QVALUE_LEARNING_RATE)+", POLICY_LEARNING_RATE = "+str(POLICY_LEARNING_RATE)+", DECAY_RATE = "+str(DECAY_RATE)+", UPDATE_RATE = "+str(UPDATE_RATE)+", REPLAY_SIZE"+str(REPLAY_SIZE)+", BATCH_SIZE"+str(BATCH_SIZE)+", NH1 = "+str(NH1)+", NH2 = "+str(NH2) + ",reward weights = "+str(reward_wheights)
            +"RANDOM RESET = "+str(RANDSET))
    f.close() 

    plt.plot( np.cumsum(h_rwd)/list(range(1,NEPISODES)))
    plt.grid(True)
    #plt.show()
    plt.savefig('/home/pasquale/Desktop/thesis/thesis-code/1D_pendulum/reward{}.png'.format(sim_number))



#LOG VIDEO

