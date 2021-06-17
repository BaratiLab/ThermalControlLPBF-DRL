import gym
from gym import spaces
from EagarTsaiModel import EagarTsai as ET
import numpy as np
import os
from stable_baselines3.common.env_checker import check_env
from matplotlib import pyplot as plt
from pylab import gca
#This script implements the RL environment as a custom OpenAI Gym environment, using Stable Baselines 3
# Dependencies: Stable Baselines 3, OpenAI Gym, Pytorch



# Specify local figure directory to store plots, diagnostic info
fig_dir = 'results/power_control_square_figures'
if not os.path.exists(fig_dir):
    os.makedirs(fig_dir, exist_ok = True)
def frame_tick(frame_width = 2, tick_width = 1.5):
    ax = gca()
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(frame_width)
    plt.tick_params(direction = 'in', 
                    width = tick_width)

class EnvRLAM(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}
    def __init__(self, plot = False, frameskip = 1, plotlast = True, num_iter = 0):
        super(EnvRLAM, self).__init__()    # Define action and observation space
    # They must be gym.spaces objects    
        self.plot = plot
        self.action_space = spaces.Box(low=np.array([-1]), high=np.array([1]), dtype=np.float64)
        self.squaresize = 10
        self.spacing = 20e-6
        self.observation_space = spaces.Box(low=300, high=20000, shape=(9, self.squaresize, self.squaresize,), dtype=np.float64)
        self.ETenv = ET(20e-6, V = 0.8, bc = 'flux', spacing = self.spacing)
        self.current_step = 0
        self.depths = []
        self.times =[]
        self.indtimes = []
        self.inddepth = []
        self.plotlast = plotlast
        self.indpower = []
        self.indvel = []
        self.power = []
        self.velocity = []
        self.hv = 0
        self.current = 0
        self.angle = 0
        self.dt = 0
        self.distance = 0
        self.dir = 0
        self.residual = False
        self.reward = 0
        self.timesteps = 0
        self.num_iter = num_iter
        self.frameskip = frameskip
        
        buffer = np.stack((self.ETenv.theta[0:self.squaresize, 0:self.squaresize, 0], self.ETenv.theta[0:self.squaresize, 0:self.squaresize, 0], self.ETenv.theta[0:self.squaresize, 0:self.squaresize, 0],
                                self.ETenv.theta[0:self.squaresize, 0:self.squaresize, 0], self.ETenv.theta[0:self.squaresize, 0:self.squaresize, 0], self.ETenv.theta[0:self.squaresize, 0:self.squaresize, 0],
                                self.ETenv.theta[0:self.squaresize, 0:self.squaresize, 0], self.ETenv.theta[0:self.squaresize, 0:self.squaresize, 0], self.ETenv.theta[0:self.squaresize, 0:self.squaresize, 0]))

        self.buffer = (buffer - np.mean(buffer))
    def step(self, action):
       # print("step")
        # time = action[0]*125e-6 + 140e-6
        time = 125e-6
        power = action[0]*250 + 250 
      #  print(power)
       # power = 0
       # power = 145
        for m in range(self.frameskip):
           # print("frame skip")
            
            done = False

            self.current_step += 1

            V =  125e-6*0.8/time
            #dist = V*time
        #
            self.velocity.append(V)
            self.power.append(power)
            
            idx = self.current_step -1
            if idx < 0:
                idx = 0                    
            if self.timesteps % 4 == 0:
                self.dir = 'right'
                angle = 0
               # power += 10
               # print(self.distance)
                if self.distance  >= 1250e-6*0.8 - 125e-6:
                    self.dir = 'up'
                    self.distance = 0
                    angle = np.pi/2
                    self.timesteps += 1
                 #   print(self.distance, 'moving')
                   # breakpoint()
            
            elif self.timesteps % 4 == 1 or self.timesteps % 4 == 3:
                angle = np.pi/2
                self.dir = 'up'
                if self.distance >= 120e-6*0.8:
                    self.timesteps += 1
                    self.distance = 0
                    self.dir = 0
                    if self.timesteps % 4 == 2:
                        angle = np.pi
                    if self.timesteps % 4 == 0:
                        angle = 0
            
            elif self.timesteps % 4 == 2:
                self.dir = 'left'
                angle = np.pi
                if self.distance  >= 1250e-6*0.8 - 125e-6:
                    #breakpoint()
                    self.dir = 'up'
                    self.distance = 0
                    angle = np.pi/2
                    self.timesteps += 1
            
            self.ETenv.forward(time, angle, V = V, P = power)
           
            self.distance += 125e-6*0.8
        
            meltpool = self.ETenv.meltpool()
            self.depths.append(meltpool)
            self.times.append(self.ETenv.time)
            if m == 0:
                self.inddepth.append(meltpool)
                self.indtimes.append(self.ETenv.time)
                self.indvel.append(V)
                self.indpower.append(power)
            reward = 1 - np.abs((55e-6  + meltpool)/25e-6) 
            self.reward += reward

            # Plotting diagnostics
            if self.plot and not self.plotlast:
                    
                np.savetxt(fig_dir + "/" + "timecontrolsquaretimesnorm",np.array(self.times)*1e3)
                np.savetxt(fig_dir + "/" + "timecontrolsquaredepthsframeskip" + str(self.frameskip), np.array(self.depths))

                np.savetxt(fig_dir + "/" + "timecontrolsquarevelocityframeskip" + str(self.frameskip), np.array(self.velocity))
                testfigs  = self.ETenv.plot()
                highxlim = np.max(self.times)
                testfigs[0].savefig(fig_dir + "/" + str(self.frameskip) + 'constanttimecontrolsquare_test' +  '%04d' % self.current_step + ".png")
                plt.clf()
                font_size = 14
                plt.plot(np.array(self.times)*1e3, np.array(self.depths)*1e6, linewidth = 2.0)
                plt.ylim(-120, 10)            
                plt.xlabel(r'Time, $t$ [ms]', fontsize =  font_size)            
                plt.ylabel(r'Melt Depth, $d$, [$\mu$m]', fontsize =  font_size)
                plt.plot(np.array(self.indtimes)*1e3, np.array(self.inddepth)*1e6, 'k.')
                np.max(np.array(self.times))
                plt.xlim(0, highxlim*1e3)
                plt.title(str(round(self.ETenv.time*1e6)) + r'[$\mu$s] ')
                #plt.show()
            #  breakpoint()
            
                plt.plot(np.arange(0, np.max(np.array(self.times))*1e3, 0.01), -55*np.ones(len(np.arange(0, np.max(np.array(self.times))*1e3, 0.01))), 'k--')
                
                plt.savefig(fig_dir + "/" + str(self.frameskip) + 'timecontrolsquaretestdepth'+  '%04d' % self.current_step + ".png")
                plt.clf()            

                plt.plot(np.array(self.times)*1e3, self.velocity)
                plt.plot(np.array(self.indtimes)*1e3, np.array(self.indvel),'k.', linewidth = 2.0)
                plt.xlabel(r'Time, $t$ [ms]', fontsize =  font_size)
                plt.ylabel(r'Velocity, $V$, [m/s]', fontsize =  font_size)
                plt.xlim(0, highxlim*1e3)
                plt.ylim(0, 3.0)
                plt.title(str(round(self.ETenv.time*1e6)) + r'[$\mu$s] ')
                plt.savefig(fig_dir + "/" +str(self.frameskip) + 'timecontrolsquaretestvelocity'+  '%04d' % self.current_step + ".png")
                plt.clf()
                

                plt.plot(np.array(self.times)*1e3, self.power)
                plt.plot(np.array(self.indtimes)*1e3, np.array(self.indpower), 'k.', linewidth = 2.0)
                plt.xlabel(r'Time, $t$ [ms]', fontsize =  font_size)
                plt.ylabel(r'Power, $P$, [W]', fontsize =  font_size)
                plt.xlim(0, highxlim*1e3)
                plt.ylim(-10, 600)
                plt.title(str(round(self.ETenv.time*1e6)) + r'[$\mu$s] ')
                plt.savefig(fig_dir + "/" +str(self.frameskip) + 'timecontrolsquaretestpower'+  '%04d' % self.current_step + ".png")
                plt.clf()
            

            idxx = self.ETenv.location_idx[0]
            idxy = self.ETenv.location_idx[1]
            
            
            self.buffer[8, :, :] = np.copy(self.buffer[5, :, :])
            self.buffer[7, :, :] = np.copy(self.buffer[4, :, :])
            self.buffer[6, :, :] = np.copy(self.buffer[3, :, :])
            
            
            
            self.buffer[5, :, :] = np.copy(self.buffer[2, :, :])
            self.buffer[4, :, :] = np.copy(self.buffer[1, :, :])
            self.buffer[3, :, :] = np.copy(self.buffer[0, :, :])

            padsize = self.squaresize
        
            
            
            theta_pad = np.copy(np.pad(self.ETenv.theta, ((padsize, padsize), (padsize, padsize), (0, 0)), mode = 'reflect'))
            try:
                self.buffer[2, :, :] = theta_pad[padsize+idxx-self.squaresize//2:padsize+idxx+self.squaresize//2, padsize+idxy-self.squaresize//2:padsize+idxy+self.squaresize//2, -1]
            
                self.buffer[1, :, :] = theta_pad[padsize+idxx-self.squaresize//2:padsize+idxx+self.squaresize//2, padsize+idxy, 0:self.squaresize]
                self.buffer[0, :, :] = theta_pad[padsize+idxx, padsize+idxy-self.squaresize//2:padsize+idxy+self.squaresize//2, 0:self.squaresize]
            except:
                breakpoint()
           # self.buffer[0:3] = (self.buffer[0:3] - np.mean(self.buffer[0:3]))/np.std(self.buffer[0:3])
            self.buffer[0:3] = (self.buffer[0:3] - np.mean(self.buffer[0:3], axis = (1,2))[:, None, None])/(np.std(self.buffer[0:3], axis = (1,2))[:, None, None] + 1e-10)
           # self.buffer[self.buffer > 1] = 1
            #self.buffer = self.buffer*2 - 1 

            obs = self.buffer
            if self.timesteps > 19:
                #breakpoint()

              #  reward = self.reward/20 - 0.5*np.std(2*np.array(self.depths)/55e-6 -np.mean(np.array(self.depths)/55e-6))   
            # reward = self.reward/20 - 0.5*np.std(2*np.array(self.depths)/55e-6 -np.mean(np.array(self.depths)/55e-6))   
                minmax_reward = (np.max(self.depths[20:]) - np.min(self.depths[20:]))/25e-6 
                gradient_reward = np.mean(np.abs(np.diff(self.depths[20:])/25e-6))
                reward = self.reward/20 - 0.5*minmax_reward# - gradient_reward# -np.mean(np.array(self.depths)/55e-6))   
                #breakpoint()
                #std_reward = 0.5*np.std(2*np.array(self.depths)/55e-6 -np.mean(np.array(self.depths)/55e-6))
                
                print(reward, "reward", "velocity", V, "power", power, "timestep", self.timesteps, "amplitude", minmax_reward)
                done = True
                print("Episode done")
                #exit()
            else:
                done = False
            if done:
               # breakpoint()
                if self.plotlast:
                    
                    np.savetxt(fig_dir + "/" + "timecontrolsquaretimesnorm",np.array(self.times)*1e3)
                    np.savetxt(fig_dir + "/" + "timecontrolsquaredepthsframeskip" + str(self.frameskip), np.array(self.depths))
                    np.savetxt(fig_dir + "/" + "timecontrolsquarepowerframeskip" + str(self.frameskip)+str(self.num_iter), np.array(self.power))
                    np.savetxt(fig_dir + "/" + "timecontrolsquarevelocityframeskip" + str(self.frameskip)+str(self.num_iter), np.array(self.velocity))
                    testfigs  = self.ETenv.plot()
                    highxlim = np.max(self.times)
                    frame_tick()
                    testfigs[0].savefig(fig_dir + "/" + str(self.frameskip)+str(self.num_iter) + 'timecontrolsquare_test' +  '%04d' % self.current_step + ".png")
                    plt.clf()
                    font_size = 14
                    plt.plot(np.array(self.times)*1e3, np.array(self.depths)*1e6, linewidth = 2.0)
                    plt.ylim(-120, 10)            
                    plt.xlabel(r'Time, $t$ [ms]', fontsize =  font_size)            
                    plt.ylabel(r'Melt Depth, $d$, [$\mu$m]', fontsize =  font_size)
                    plt.plot(np.array(self.indtimes)*1e3, np.array(self.inddepth)*1e6, 'k.')
                    np.max(np.array(self.times))
                    plt.xlim(0, highxlim*1e3)
                    plt.title(str(round(self.ETenv.time*1e6)) + r'[$\mu$s] ')
                    #plt.show()
                    #  breakpoint()

                    plt.plot(np.arange(0, np.max(np.array(self.times))*1e3, 0.01), -55*np.ones(len(np.arange(0, np.max(np.array(self.times))*1e3, 0.01))), 'k--')

                    plt.savefig(fig_dir + "/" + str(self.frameskip) +str(self.num_iter)+ 'timecontrolsquaretestdepth'+  '%04d' % self.current_step + ".png")
                    plt.clf()            

                    plt.plot(np.array(self.times)*1e3, self.velocity)
                    plt.plot(np.array(self.indtimes)*1e3, np.array(self.indvel),'k.', linewidth = 2.0)
                    plt.xlabel(r'Time, $t$ [ms]', fontsize =  font_size)
                    plt.ylabel(r'Velocity, $V$, [m/s]', fontsize =  font_size)
                    plt.xlim(0, highxlim*1e3)
                    plt.ylim(0, 3.0)
                    plt.title(str(round(self.ETenv.time*1e6)) + r'[$\mu$s] ')
                    plt.savefig(fig_dir + "/" +str(self.frameskip)+str(self.num_iter) + 'timecontrolsquaretestvelocity'+  '%04d' % self.current_step + ".png")
                    plt.clf()


                    plt.plot(np.array(self.times)*1e3, self.power)
                    plt.plot(np.array(self.indtimes)*1e3, np.array(self.indpower), 'k.', linewidth = 2.0)
                    plt.xlabel(r'Time, $t$ [ms]', fontsize =  font_size)
                    plt.ylabel(r'Power, $P$, [W]', fontsize =  font_size)
                    plt.xlim(0, highxlim*1e3)
                    plt.ylim(-10, 600)
                    plt.title(str(round(self.ETenv.time*1e6)) + r'[$\mu$s] ')
                    plt.savefig(fig_dir + "/" +str(self.frameskip)+str(self.num_iter) + 'timecontrolsquaretestpower'+  '%04d' % self.current_step + ".png")
                    plt.clf()
                    plt.close('all')
                break
        return obs, reward,  done,{}


    # Execute one time step within the environment
    def reset(self):
        self.ETenv.reset()
        self.current_step = 0
  
        self.hv = 0
        self.current = 0
        self.angle = 0
        self.residual = False
        self.dir = 0

        step = 2
        self.reward = 0
        self.timesteps = 0
        self.dt = 0
        self.distance = 0
        self.buffer = np.stack((self.ETenv.theta[ 0:self.squaresize,  0:self.squaresize, 0], self.ETenv.theta[ 0:self.squaresize,  0:self.squaresize, 0], self.ETenv.theta[ 0:self.squaresize,  0:self.squaresize, 0], 
        self.ETenv.theta[ 0:self.squaresize,  0:self.squaresize, 0], self.ETenv.theta[ 0:self.squaresize, 0:self.squaresize, 0], self.ETenv.theta[ 0:self.squaresize,  0:self.squaresize, 0],
        self.ETenv.theta[0:self.squaresize, 0:self.squaresize, 0], self.ETenv.theta[0:self.squaresize, 0:self.squaresize, 0], self.ETenv.theta[0:self.squaresize, 0:self.squaresize, 0]))
        
        buffer = np.stack((
        self.ETenv.theta[0:self.squaresize, 0:self.squaresize, 0], self.ETenv.theta[0:self.squaresize, 0:self.squaresize, 0], self.ETenv.theta[0:self.squaresize, 0:self.squaresize, 0], 
        self.ETenv.theta[0:self.squaresize, 0:self.squaresize, 0], self.ETenv.theta[0:self.squaresize, 0:self.squaresize, 0], self.ETenv.theta[0:self.squaresize, 0:self.squaresize, 0],
        self.ETenv.theta[0:self.squaresize, 0:self.squaresize, 0], self.ETenv.theta[0:self.squaresize, 0:self.squaresize, 0], self.ETenv.theta[0:self.squaresize, 0:self.squaresize, 0]))

        self.buffer = (buffer - np.mean(buffer))
        #buffer[buffer > 1] = 1
       # buffer = buffer*2 - 1 

        return buffer
    # Reset the state of the environment to an initial state
    def render(self, mode='console', close=False):
        pass
    def plot(self):
        pass
    def plot_buffer(self, action):               
        time = action[0]*125e-6 + 140e-6
       # power = action[1]*250 + 250 
        power = 145
        V = 100e-6/time
        buffer_dir = fig_dir + '/buffer/'
        if not os.path.exists(buffer_dir):
            os.makedirs(buffer_dir)
        for index in range(9):
            plt.close('all')
            try:
                if index % 3 == 2:
                    plt.pcolormesh(self.buffer[index].T,  cmap = 'jet')
                if index % 3 == 1:
                    plt.pcolormesh(self.buffer[index].T, cmap = 'jet')
                if index % 3 == 0:
                    plt.pcolormesh(self.buffer[index].T,  cmap = 'jet')
                # breakpoint()
            except Exception as e:
                print(e)
                breakpoint()
                print('saved')
            plt.title("time: " + str(V) + " power: " + str(power)) 
            plt.savefig(buffer_dir + str(index) + "closesquarebuffer" +  '%04d' % self.current_step + ".png")   
            
            plt.clf()
            plt.close('all')
def main():
    env = EnvRLAM()
# It will check your custom environment and output additional warnings if needed
    check_env(env)
