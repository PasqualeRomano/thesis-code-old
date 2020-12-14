import matplotlib.pyplot as plt
import csv ,pandas as pd



path_myddpg = "/home/pasquale/Desktop/thesis/thesis-code/1D_pendulum/ddpg/eval/"
path_stbsddpg = "/home/pasquale/Desktop/thesis/thesis-code/1D_pendulum/stable_baselines/eval/"
#Comparison my ddpg vs stablebaselines ddpg

#Extract data

data_myddpg = pd.read_csv(path_myddpg+'ang_vel_act_seq.csv')
data_stbsddpg = pd.read_csv(path_stbsddpg+'ang_vel_act_seq.csv')
#   data from 100 randset episodes 
#   0-> angles  
#   1-> vel
#   2-> actions

ang_seq_myddpg = list(data_myddpg["0"])
vel_seq_myddpg = list(data_myddpg["1"])
act_seq_myddpg =  list(data_myddpg["2"])

ang_seq_stbsddpg = list(data_stbsddpg["0"])
vel_seq_stbsddpg = list(data_stbsddpg["1"])
act_seq_stbsddpg =  list(data_stbsddpg["2"])
# plt.figure()
# plt.subplot(211)
# plt.plot(ang_seq_myddpg, 'b.')
# plt.grid(color='k', linestyle='-', linewidth=.5)



# plt.show()
# plt.subplot(212)
# plt.plot(vel_seq_myddpg, 'r.')
# plt.grid(color='k', linestyle='-', linewidth=.5)

# plt.show()

f, (ax1, ax2) = plt.subplots(2, 1,sharex=True)
ax1.plot(ang_seq_myddpg,'b.')
ax1.set_ylabel('angle [deg]')
ax1.set_xlabel('step')
ax1.grid(color='k', linestyle='-', linewidth=.25)

ax2.plot(vel_seq_myddpg,'r.')
ax2.set_ylabel('angular velocity [rad/s]')
ax2.set_xlabel('step')
ax2.grid(color='k', linestyle='-', linewidth=.25)
#plt.savefig('/home/pasquale/Desktop/thesis/thesis-code/plots/plot_ddpg_my_1Dp.eps', format='eps')


f, (ax1, ax2) = plt.subplots(2, 1,sharex=True)
ax1.plot(ang_seq_stbsddpg,'b.')
ax1.set_ylabel('angle [deg]')
ax1.set_xlabel('step')
ax1.grid(color='k', linestyle='-', linewidth=.25)


ax2.plot(vel_seq_stbsddpg,'r.')
ax2.set_ylabel('angular velocity [rad/s]')
ax2.set_xlabel('step')
ax2.grid(color='k', linestyle='-', linewidth=.25)
#plt.savefig('/home/pasquale/Desktop/thesis/thesis-code/plots/plot_ddpg_stbs_1Dp.eps', format='eps')
#plt.show()


f, (ax) = plt.subplots(1,1)
ax.scatter(list(range(0,len(act_seq_myddpg))),act_seq_myddpg,marker = ".",color = 'darkgreen')

ax.scatter(list(range(0,len(act_seq_myddpg))),act_seq_stbsddpg, marker= '.', color = 'darkorange')

ax.set_ylabel('torque [Nm]')
ax.set_xlabel('step')
ax.grid(color='k', linestyle='-', linewidth=.25)
plt.show()