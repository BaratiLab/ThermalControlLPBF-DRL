import gym
import json
import datetime as dt
#from stable_baselines3.common.policies import MlpPolicy
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from stable_baselines3 import PPO, A2C



from power_square_gym import EnvRLAM as powersquareEnvRLAM

from time_square_gym import EnvRLAM as timesquareEnvRLAM

from power_triangle_gym import EnvRLAM as powertriangleEnvRLAM

from time_triangle_gym import EnvRLAM as timetriangleEnvRLAM   



from stable_baselines3.ppo import MlpPolicy, CnnPolicy
from stable_baselines3.common.cmd_util import make_vec_env
import numpy as np
import torch as th
import matplotlib.pyplot as plt

import os
import sys
from stable_baselines3.common import results_plotter
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import BaseCallback


from stable_baselines3 import TD3
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import BaseCallback



from stable_baselines3.common.vec_env import VecEnvWrapper
import numpy as np
import time
from collections import deque
import os.path as osp
import json
import csv
   
def make_env(env_id, rank, seed=0):

    def _init():
        return env_id
    return _init

def main():
    path_toggle =  sys.argv[0] # 0 for HCH path, 1 for triangular path
    control_toggle =  sys.argv[1] # 0 for power, 1 for velocity

    model_filename = input("Enter filename of model zip file: ")
    if path_toggle == 0:
        if control_toggle == 0:
            env = powersquareEnvRLAM(plot = True, frameskip= 1, plotlast = False)
            

        elif control_toggle == 1:
            
            env = timesquareEnvRLAM(plot = True, frameskip= 1, plotlast = False)
    if path_toggle == 1:
    
        if control_toggle == 0:

            env = powertriangleEnvRLAM(plot = True, frameskip= 2, plotlast = False)
        
        elif control_toggle == 1:

            env = timetriangleEnvRLAM(plot = True, frameskip= 1, plotlast = False)        
    
        
    model = PPO.load(model_filename)
    
    constantenv = constantEnvRLAM(plot = True)
    num_cpu = 1

    obs = env.reset()
    c = 0
    while True:
        c = c+ 1
        action, _states = model.predict(obs, deterministic = True)
        obs, rewards, dones, info = env.step(action)
        
        if np.any(dones) == True:
            break
if __name__ == "__main__":
    main()
