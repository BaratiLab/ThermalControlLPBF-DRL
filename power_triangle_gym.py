import gym
from gym import spaces
from EagarTsaiModel import EagarTsai as ET
import numpy as np
from stable_baselines3.common.env_checker import check_env
from matplotlib import pyplot as plt
import os
fig_dir = 'results/triangle_power_control_figures'
if not os.path.exists(fig_dir):
    os.makedirs(fig_dir)

class EnvRLAM(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}
    def __init__(self, plot = False, frameskip = 2):
        super(EnvRLAM, self).__init__()    # Define action and observation space
    # They must be gym.spaces objects    
        self.plot = plot
        self.squaresize = 10
        self.action_space = spaces.Box(low=np.array([-1 ]), high=np.array([1]), dtype=np.float16)
        self.observation_space = spaces.Box(low=300, high=20000, shape=(9, self.squaresize, self.squaresize,), dtype=np.float64)
        self.ETenv = ET(20e-6, V = 0.8, bc = 'flux')
        if plot:
            self.constantETenv = ET(20e-6, V = 0.8, bc = 'flux')
        self.velocity = []

        self.current_step = 0
        self.depths = []
        self.times =[]
        self.power = []
        self.hv = 0
        self.current = 0
        self.angle = 0
        self.inddepth = []
        self.indtimes = []
        self.indpower = []
        self.residual = False
        self.reward = 0
        self.timesteps = 0
        self.frameskip = frameskip
        buffer = np.stack((self.ETenv.theta[0:self.squaresize, 0:self.squaresize, 0], self.ETenv.theta[0:self.squaresize, 0:self.squaresize, 0], self.ETenv.theta[0:self.squaresize, 0:self.squaresize, 0],
                                self.ETenv.theta[0:self.squaresize, 0:self.squaresize, 0], self.ETenv.theta[0:self.squaresize, 0:self.squaresize, 0], self.ETenv.theta[0:self.squaresize, 0:self.squaresize, 0],
                                self.ETenv.theta[0:self.squaresize, 0:self.squaresize, 0], self.ETenv.theta[0:self.squaresize, 0:self.squaresize, 0], self.ETenv.theta[0:self.squaresize, 0:self.squaresize, 0]))
        
        self.buffer = (buffer - np.mean(buffer))
        
  
    def step(self, action):

        power = action[0]*250 +250
        time = 50e-6

        for m in range(self.frameskip):
            self.power.append(power)
            self.timesteps += 1                
            V = 50e-6*0.8/time
            self.velocity.append(V)
          #  breakpoint()
            if self.current_step >= 12:
                reward = self.reward/12  - 0.5*(np.max(self.depths[20:]) - np.min(self.depths[20:]))/25e-6        
                print(reward, "reward", power, "timestep", self.timesteps, "RESIDUAL")
                done = True
            else:
                done = False

            if self.residual:
                V = (self.hv - self.current)*0.8/time
                
                self.ETenv.forward(time, self.angle, V = V, P = power)
                meltpool = self.ETenv.meltpool()
                self.depths.append(meltpool)
                self.times.append(self.ETenv.time)
                reward = 1 - np.abs((55e-6  + meltpool)/25e-6) 
                self.reward += reward
                
                self.current_step += 1
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

                self.buffer[0:3] = (self.buffer[0:3] - np.mean(self.buffer[0:3], axis = (1,2))[:, None, None])/(np.std(self.buffer[0:3], axis = (1,2))[:, None, None] + 1e-10)


                obs = self.buffer
                self.residual = False
                self.hv = 0
                self.current = 0
                return obs, reward,  done,{}
            else:
                
                idx = self.current_step -1
                if idx < 0:
                    idx = 0
                self.hv =(750e-6 - idx*70e-6)
                dtprime = 0
                angle = (self.current_step % 3)*2*np.pi/3
                step = 50e-6*0.8
                self.angle = angle
                self.current += step
                self.ETenv.forward(time, angle, V = V, P = power)

                dtprime += step
                meltpool = self.ETenv.meltpool()
                self.depths.append(meltpool)
                self.times.append(self.ETenv.time)
                reward = 1 - np.abs((55e-6  + meltpool)/25e-6) 
                self.reward += reward

                if ((self.hv - self.current) < 50e-6*0.8 and (self.hv - self.current) > 0):   
                    self.residual = True

            if m == 0:
                self.inddepth.append(meltpool)
                self.indtimes.append(self.ETenv.time)
            if self.plot:
                np.savetxt(fig_dir + "/" + "timecontroltriangletimesnorm",np.array(self.times)*1e3)
                np.savetxt(fig_dir + "/" + "timecontroltriangledepthsframeskip" + str(self.frameskip), np.array(self.depths))
                np.savetxt(fig_dir + "/" + "timecontroltrianglevelocityframeskip" + str(self.frameskip), np.array(self.velocity))
                np.savetxt(fig_dir + "/" + "timecontroltrianglepowerframeskip" + str(self.frameskip), np.array(self.power))
                testfigs  = self.ETenv.plot()
                testfigs[0].savefig(fig_dir + "/" + str(self.frameskip) + 'timecontroltriangle_test' +  '%04d' % self.timesteps + ".png")
                plt.clf()

                plt.plot(np.array(self.times)*1e3, np.array(self.depths)*1e6)
                plt.ylim(-120, 10)            
                plt.xlabel(r'Time, $t$ [ms]')            
                plt.ylabel(r'Melt Depth, $d$, [$\mu$m]')
                plt.plot(np.array(self.indtimes)*1e3, np.array(self.inddepth)*1e6, 'k.')
                plt.xlim(0, 0.008*1e3)
                plt.title(str(round(self.ETenv.time*1e6)) + r'[$\mu$s] ')
                plt.plot(np.arange(0, 0.015*1e3, 0.01), -55*np.ones(len(np.arange(0, 0.015*1e3, 0.01))), 'k--')
                plt.savefig(fig_dir + "/" + str(self.frameskip) + 'timecontroltriangletestdepth'+  '%04d' % self.timesteps + ".png")
                plt.clf()            

                plt.plot(np.array(self.times)*1e3, self.velocity)
                plt.xlabel(r'Time, $t$ [ms]')
                plt.ylabel(r'Velocity, $V$, [m/s]')
                plt.xlim(0, 0.008*1e3)
                plt.ylim(0, 10.0)
                plt.title(str(round(self.ETenv.time*1e6)) + r'[$\mu$s] ')
                plt.savefig(fig_dir + "/" +str(self.frameskip) + 'timecontroltriangletestvelocity'+  '%04d' % self.timesteps + ".png")
                plt.clf()
                plt.close('all')

                plt.plot(np.array(self.times)*1e3, self.power)
                plt.xlabel(r'Time, $t$ [ms]')
                plt.ylabel(r'Power, $P$, W')
                plt.xlim(0, 0.008*1e3)
                plt.ylim(0, 500.0)
                plt.title(str(round(self.ETenv.time*1e6)) + r'[$\mu$s] ')
                plt.savefig(fig_dir + "/" +str(self.frameskip) + 'timecontroltriangletestpower'+  '%04d' % self.timesteps + ".png")
                plt.clf()
                plt.close('all')
 
                print(self.timesteps, "PLOT")
            interv = 2
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
            except Exception as e:
                print(e)
                breakpoint()

            self.buffer[0:3] = (self.buffer[0:3] - np.mean(self.buffer[0:3], axis = (1,2))[:, None, None])/(np.std(self.buffer[0:3], axis = (1,2))[:, None, None] + 1e-10)
            obs = self.buffer
        return obs, reward,  done,{}

    
    # Execute one time step within the environment
    def reset(self):
        self.ETenv.reset()
        self.current_step = 0
        
        self.hv = 0
        self.current = 0
        self.angle = 0
        self.residual = False
        
        step = 2
        self.reward = 0
        self.timesteps = 0

        buffer = np.stack((self.ETenv.theta[ 0:self.squaresize,  0:self.squaresize, 0], self.ETenv.theta[ 0:self.squaresize,  0:self.squaresize, 0], self.ETenv.theta[ 0:self.squaresize,  0:self.squaresize, 0], 
        self.ETenv.theta[ 0:self.squaresize,  0:self.squaresize, 0], self.ETenv.theta[ 0:self.squaresize, 0:self.squaresize, 0], self.ETenv.theta[ 0:self.squaresize,  0:self.squaresize, 0],
        self.ETenv.theta[0:self.squaresize, 0:self.squaresize, 0], self.ETenv.theta[0:self.squaresize, 0:self.squaresize, 0], self.ETenv.theta[0:self.squaresize, 0:self.squaresize, 0]))
        self.buffer = (buffer - np.mean(buffer))
        return np.copy(self.buffer)
    # Reset the state of the environment to an initial state
    def render(self, mode='console', close=False):
        pass
    def plot(self):
        plt.close('all')       
        self.ax.plot(self.times, self.depths)
        plt.xlabel(r'Time, $t$ [ms]')
        plt.ylabel(r'Melt Depth, $d$, [$\mu$m]')       
        
        plt.close('all')
        return self.fig
    # Render the environment to the screen
def main():
    env = EnvRLAM()
# It will check your custom environment and output additional warnings if needed
    check_env(env)
