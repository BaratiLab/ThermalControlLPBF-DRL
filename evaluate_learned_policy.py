import gym
import json
import datetime as dt
#from stable_baselines3.common.policies import MlpPolicy
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from stable_baselines3 import PPO, A2C
import argparse



from power_square_gym import EnvRLAM as powersquareEnvRLAM

from velocity_square_gym import EnvRLAM as velocitysquareEnvRLAM

from power_triangle_gym import EnvRLAM as powertriangleEnvRLAM

from velocity_triangle_gym import EnvRLAM as velocitytriangleEnvRLAM   



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
import csv
   
def make_env(env_id, rank, seed=0):

    def _init():
        return env_id
    return _init

def parse_arguments():
    parser = argparse.ArgumentParser()  
    #parser = parser.add_mutually_exclusive_group(required=False)
    parser.add_argument('--path', dest='path',
                              default='triangle',
                              help="Which scan path to use, options are \'square\' and \'triangle \'") 
    parser.add_argument('--param', dest='param', default = 'velocity',
        help="Which control parameter to vary, options are \'velocity\' and \'power \' ")

    parser.set_defaults(debug=False, param = 'velocity')

    return parser.parse_args()
def main():
  
    args = parse_arguments()
    path = args.path
    parameter = args.param

    model_filename = input("Enter filename of model zip file: ")
    if path == 'square':
        if parameter == 'power':
            env = powersquareEnvRLAM(plot = True, frameskip= 1, plotlast = False)
            

        elif parameter == 'velocity':
            
            env = velocitysquareEnvRLAM(plot = True, frameskip= 1, plotlast = False)
    if path == 'triangle':
    
        if parameter == 'power':

            env = powertriangleEnvRLAM(plot = True, frameskip= 2, plotlast = False)
        
        elif parameter == 'velocity':

            env = velocitytriangleEnvRLAM(plot = True, frameskip= 1, plotlast = False)        
    
        
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
