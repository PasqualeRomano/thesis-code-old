
close all
clc
set(groot,'defaulttextinterpreter','latex');  
set(groot, 'defaultAxesTickLabelInterpreter','latex');  
set(groot, 'defaultLegendInterpreter','latex');
%%
addpath('/home/pasquale/Desktop/thesis/thesis-code/1D_pendulum/ddpg/eval')



%%
data_ang_vel_ddpg = readtable('angles_and_vel_seq.csv', 'HeaderLines',1);
%%DDPG modified implementation
%plot angle and velocity
figure()
tiledlayout(2,1)
ax1 = nexttile;

title(ax1,'Angle')
xlabel(ax1,'step')
ylabel(ax1,'deg')

plot(data_ang_vel_ddpg.Var1,data_ang_vel_ddpg.Var2)
ax2 = nexttile;

title(ax2,'Angular Velocity')
xlabel(ax2,'step')
ylabel(ax2,'\frac{rad}{s}')

plot(data_ang_vel_ddpg.Var1,data_ang_vel_ddpg.Var3)