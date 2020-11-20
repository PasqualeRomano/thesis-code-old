import pybullet as p
import pybullet_data
import numpy as np
import time

class Robot:
    def __init__(self, load_file,time_step=1/240,GUI_ENABLED=1,sim_number=0,
                 urdf_path = "/home/pasquale/Desktop/thesis/thesis-code/urdf"):
        self.load_file = load_file
        self.urdf_path = urdf_path
        
        self.gravity = [0,0,-9.81]
        self.time_step=time_step
        
        
        self.LOGDATA=0
        self.GUI_ENABLED=GUI_ENABLED
        self.SIM_READDY=0
        self.SINCOS = 0
        self.RANDSET =0
        
        self.video_path = "/home/pasquale/Desktop/thesis/thesis-code/1D_pendulum/continuous/Video"
        
        
        self.client_id = 0
        self.sim_number=sim_number
        
        self.init_pos = [0.0,0.0,0.0]
        self.init_or = p.getQuaternionFromEuler([0.0, 0.0, 0.0])
        self.states = []
        self.states_sincos = []
        self.states_dot = []
        
        self.actuated_index = []
        self.observed_index = []
        
        self.joint_limits_high = []
        self.max_torque = 2.0

    def setupSim(self):
        
        if self.SIM_READDY ==1:
            
            raise Exception("Simulation already set up")

        
        if self.GUI_ENABLED==1:
            self.client_id =self.physicsClient = p.connect(p.GUI)
            p.resetDebugVisualizerCamera( cameraDistance=0.8, cameraYaw=80, cameraPitch=-10, cameraTargetPosition=[0,0,0],physicsClientId = self.client_id)
            if self.LOGDATA==1:
                self.state_seq = []
                self.state_dot_seq = []
                self.actions_seq = []
                print("Starting video")
                p.startStateLogging(p.STATE_LOGGING_VIDEO_MP4, self.video_path + "/Pendulum_training{}.mp4".format(self.sim_number),physicsClientId = self.client_id)               
        else:
            self.client_id =self.physicsClient = p.connect(p.DIRECT)
            
        p.setAdditionalSearchPath(self.urdf_path)  
        p.setGravity = (self.gravity[0],self.gravity[1],self.gravity[2],self.client_id)
        
        
        self.SIM_READDY = 1
        p.setTimeStep(self.time_step,self.client_id)
        self.loadRobot()
        
        p.stepSimulation(physicsClientId = self.client_id) 
        self.initial_frame = p.saveState(physicsClientId = self.client_id)
        
        
        self.observeState()
        
        print("Simulation is ready")
        
        
     
    def observeState(self):
        if self.SIM_READDY==1:
            self.getObservedLink() 
            ss_tmp =p.getLinkStates(self.body_id,self.observed_index,1,physicsClientId = self.client_id)
            self.states = []
            self.states_dot = []
            self.states_sincos=[]
            
        
            for i in range(len(self.observed_index)):
               
                self.states.append(ss_tmp[i][0]+ p.getEulerFromQuaternion(ss_tmp[i][1]))  ##[  [pos,pos,pos,ang,ang,ang] , [pos,pos,pos,ang,ang,ang] , ... ,for each Link]
             
                if self.SINCOS==1:
                    self.states_sincos.append((np.sin(self.states[i][3]),np.cos(self.states[i][3]), # [    [sin,cos,sin,cos,sin,cos]   ,..., for each Link]
                                          np.sin(self.states[i][4]),np.cos(self.states[i][4]),
                                          np.sin(self.states[i][5]),np.cos(self.states[i][5])))
        
                self.states_dot.append(ss_tmp[i][6]+ss_tmp[i][7])                        ##[    [vel,vel,vel,omega,omega,omega]    ,....,  for each link]
            
            #for logging
            if self.LOGDATA==1:
                self.state_seq.append(self.states)
                self.state_dot_seq.append(self.states_dot)
                   
           
                
     
        else:
            raise Exception("Simulation not set up")
           
        
    def simulateDyn(self,actions):
        if self.SIM_READDY==1: 
            
            bounded = np.clip(np.array(actions),-self.max_torque,self.max_torque).tolist()
            
            p.setJointMotorControlArray(self.body_id, self.actuated_index, p.TORQUE_CONTROL, forces=bounded,physicsClientId = self.client_id)
            p.stepSimulation(physicsClientId = self.client_id)
            if self.LOGDATA==1:
                self.actions_seq.append(actions)
            self.observeState()  
        else:
            raise Exception("Simulation not set up") 
        
        
    def loadRobot(self):
        self.body_id = p.loadURDF(self.load_file,self.init_pos,self.init_or,physicsClientId = self.client_id)
        
        self.getActuatedJoints()            
        p.setJointMotorControlArray(self.body_id, self.actuated_index, p.VELOCITY_CONTROL, 
                                    forces=[0.0 for m in self.actuated_index],physicsClientId = self.client_id)
        p.setJointMotorControlArray(self.body_id, self.actuated_index,p.TORQUE_CONTROL, forces=[0.0 for m in self.actuated_index],physicsClientId = self.client_id)
        
    
    def getActuatedJoints(self):
        if len(self.actuated_index) == 0:
            for i in range(p.getNumJoints(self.body_id,physicsClientId = self.client_id)):
                if p.getJointInfo(self.body_id,i,physicsClientId = self.client_id)[2]==0 or p.getJointInfo(self.body_id,i,physicsClientId = self.client_id)[2]==1:
                    self.actuated_index.append(i) 
    
    def getObservedLink(self):
        if len(self.observed_index)==0:
            self.observed_index = [i for i in range(p.getNumJoints(self.body_id,physicsClientId = self.client_id))]
                    

    def resetRobot(self):
        if self.SIM_READDY==1: 
            p.restoreState(self.initial_frame,physicsClientId = self.client_id)
            if self.RANDSET==1:
                
               # for i in range(p.getNumJoints(self.body_id,physicsClientId=self.client_id)):
                    
                
                
                high = np.array([np.pi,0.,0.])
                high_vel= np.array([8.,0,0])
                rnd_pos  = np.random.uniform(low=-high, high=high)
                rnd_vel = np.random.uniform(low=-high_vel, high=high_vel)
                p.resetJointStateMultiDof(self.body_id,self.ACTUATED_JOINTS_INDEX,rnd_pos.tolist(),targetVelocity= rnd_vel.tolist(),physicsClientId = self.client_id) #####
                
            if self.LOGDATA==1:
                self.state_seq = []
                self.state_dot_seq = []
                self.actions_seq = []
                print("Starting video")
                p.startStateLogging(p.STATE_LOGGING_VIDEO_MP4,  self.video_path + "/Pendulum_training{}.mp4".format(self.sim_number),physicsClientId = self.client_id)
            self.observeState()
        else:
            raise Exception("Simulation not set up")
    
    def stopSim(self):
        if self.SIM_READDY==1: 
            p.disconnect(physicsClientId = self.client_id)
            self.SIM_READDY=0
            print('Simulation {:d} stopped'.format(self.client_id))
        else:
            raise Exception("Simulation not set up")
    


if __name__ == "__main__":

    pendulum = Robot("double_pendulum.urdf",time_step=0.01)
    
    #pendulum.LOGDATA=1
    pendulum.setupSim()
    
    for i in range (100):
     pendulum.simulateDyn([1.0,0.0])
    
    pendulum.stopSim



    """ #Start simulation environment
    #or p.DIRECT for non-graphical version
    physicsClient = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath()) #used by loadURDF
    p.setGravity(0,0,-9.81)
    #Load pendulum from urdf
    pendulum = p.loadURDF("single_pendulum.urdf")
    #Set pendulum orientation

    #Setup torque control on joint
    mode1 = p.TORQUE_CONTROL
    mode0 = p.VELOCITY_CONTROL
    torque = 0.1
    p.setJointMotorControl2(pendulum, 1, mode0, force=0.0,targetVelocity=0.0)

    #p.setJointMotorControl2(pendulum, 0, mode0, force=0)   
    #p.setJointMotorControlArray(pendulum,[0,1],)
    button = 0
    #Simulate
    for i in range (10):
   
    p.stepSimulation()
    #print(p.getJointState(pendulum,1)) #pos,vel,react_force(x6),appl_torque
    time.sleep(1./240.)

    
   
    p.setJointMotorControl2(pendulum, 1, mode1, force=0)
        
    
    
    pendPos, pendOr = p.getBasePositionAndOrientation(pendulum)
    print(pendPos,pendOr)
    print(p.getNumJoints(pendulum))
    print(p.getJointInfo(pendulum,0))
    print(p.getJointInfo(pendulum,1))
    ss_tmp =p.getLinkStates(pendulum,[1])
    ss = []
    #   for i in [0,1]:
    #     ss.append(ss_tmp[i][0:2])

    p.disconnect()  """
