


NEPISODES               = 200           # Max training steps
NSTEPS                  = 150           # Max episode length
QVALUE_LEARNING_RATE    = 0.001         # Base learning rate for the Q-value Network
POLICY_LEARNING_RATE    = 0.0001        # Base learning rate for the policy network
DECAY_RATE              = 0.99          # Discount factor 
UPDATE_RATE             = 0.01           # Homotopy rate to update the networks
REPLAY_SIZE             = 10000          # Size of replay buffer
BATCH_SIZE              = 64            # Number of points to be fed in stochastic gradient
NH1 = NH2               = 250           # Hidden layer size
range_esp = 2.