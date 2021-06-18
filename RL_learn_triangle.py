import gym
import json
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from stable_baselines3 import PPO, A2C

from power_triangle_gym import EnvRLAM as powertriangleEnvRLAM

from velocity_triangle_gym import EnvRLAM as velocitytriangleEnvRLAM   

from stable_baselines3.ppo import MlpPolicy, CnnPolicy
from stable_baselines3.common.cmd_util import make_vec_env
import numpy as np
import torch as th
import matplotlib.pyplot as plt
import os

from stable_baselines3.common import results_plotter
from stable_baselines3.common.monitor import Monitor
#from stable_baselines3.common.results_plotter import load_results#, ts2xy#, plot_results
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import BaseCallback


from stable_baselines3 import TD3
from stable_baselines3.common.monitor import Monitor
#from stable_baselines3.common.results_plotter import load_results #, ts2xy, plot_results
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import BaseCallback


import argparse

from stable_baselines3.common.vec_env import VecEnvWrapper
import numpy as np
import time
from collections import deque
import os.path as osp
import json
import csv

class VecMonitor(VecEnvWrapper):
    EXT = "monitor.csv"
    
    def __init__(self, venv, filename=None, keep_buf=0, info_keywords=()):
        VecEnvWrapper.__init__(self, venv)
        print('init vecmonitor: ',filename)
        self.eprets = None
        self.eplens = None
        self.epcount = 0
        self.tstart = time.time()
        if filename:
            self.results_writer = ResultsWriter(filename, header={'t_start': self.tstart},
                extra_keys=info_keywords)
        else:
            self.results_writer = None
        self.info_keywords = info_keywords
        self.keep_buf = keep_buf
        if self.keep_buf:
            self.epret_buf = deque([], maxlen=keep_buf)
            self.eplen_buf = deque([], maxlen=keep_buf)

    def reset(self):
        obs = self.venv.reset()
        self.eprets = np.zeros(self.num_envs, 'f')
        self.eplens = np.zeros(self.num_envs, 'i')
        return obs

    def step_wait(self):
        obs, rews, dones, infos = self.venv.step_wait()
        self.eprets += rews
        self.eplens += 1

        newinfos = list(infos[:])
        for i in range(len(dones)):
            if dones[i]:
                info = infos[i].copy()
                ret = self.eprets[i]
                eplen = self.eplens[i]
                epinfo = {'r': ret, 'l': eplen, 't': round(time.time() - self.tstart, 6)}
                for k in self.info_keywords:
                    epinfo[k] = info[k]
                info['episode'] = epinfo
                if self.keep_buf:
                    self.epret_buf.append(ret)
                    self.eplen_buf.append(eplen)
                self.epcount += 1
                self.eprets[i] = 0
                self.eplens[i] = 0
                if self.results_writer:
                    self.results_writer.write_row(epinfo)
                newinfos[i] = info
        return obs, rews, dones, newinfos
        
class ResultsWriter(object):
    def __init__(self, filename, header='', extra_keys=()):
        print('init resultswriter')
        self.extra_keys = extra_keys
        assert filename is not None
        if not filename.endswith(VecMonitor.EXT):
            if osp.isdir(filename):
                filename = osp.join(filename, VecMonitor.EXT)
            else:
                filename = filename #   + "." + VecMonitor.EXT
        self.f = open(filename, "wt")
        if isinstance(header, dict):
            header = '# {} \n'.format(json.dumps(header))
        self.f.write(header)
        self.logger = csv.DictWriter(self.f, fieldnames=('r', 'l', 't')+tuple(extra_keys))
        self.logger.writeheader()
        self.f.flush()

    def write_row(self, epinfo):
        if self.logger:
            self.logger.writerow(epinfo)
            self.f.flush()    






class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq: (int)
    :param log_dir: (str) Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: (int)
    """
    def __init__(self, check_freq: int, log_dir: str, verbose=1):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, 'best_model')
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:

          # Retrieve training reward
          x, y = ts2xy(load_results(self.log_dir), 'timesteps')
          if len(x) > 0:
              # Mean training reward over the last 100 episodes
              mean_reward = np.mean(y[-100:])
              if self.verbose > 0:
                print(f"Num timesteps: {self.num_timesteps}")
                print(f"Best mean reward: {self.best_mean_reward:.2f} - Last mean reward per episode: {mean_reward:.2f}")

              # New best model, you could save the agent here
              if mean_reward > self.best_mean_reward:
                  self.best_mean_reward = mean_reward
                  # Example for saving best model
                  if self.verbose > 0:
                    print(f"Saving new best model to {self.save_path}.zip")
                  self.model.save(self.save_path)

        return True





from typing import Tuple, Callable, List, Optional

import numpy as np
import pandas as pd
# import matplotlib
# matplotlib.use('TkAgg')  # Can change to 'Agg' for non-interactive mode
import matplotlib.pyplot as plt

from stable_baselines3.common.monitor import load_results


X_TIMESTEPS = 'timesteps'
X_EPISODES = 'episodes'
X_WALLTIME = 'walltime_hrs'
POSSIBLE_X_AXES = [X_TIMESTEPS, X_EPISODES, X_WALLTIME]
EPISODES_WINDOW = 100


def rolling_window(array: np.ndarray, window: int) -> np.ndarray:
    """
    Apply a rolling window to a np.ndarray

    :param array: (np.ndarray) the input Array
    :param window: (int) length of the rolling window
    :return: (np.ndarray) rolling window on the input array
    """
    shape = array.shape[:-1] + (array.shape[-1] - window + 1, window)
    strides = array.strides + (array.strides[-1],)
    return np.lib.stride_tricks.as_strided(array, shape=shape, strides=strides)


def window_func(var_1: np.ndarray, var_2: np.ndarray,
                window: int, func: Callable) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply a function to the rolling window of 2 arrays

    :param var_1: (np.ndarray) variable 1
    :param var_2: (np.ndarray) variable 2
    :param window: (int) length of the rolling window
    :param func: (numpy function) function to apply on the rolling window on var
iable 2 (such as np.mean)
    :return: (Tuple[np.ndarray, np.ndarray])  the rolling output with applied fu
nction
    """
    var_2_window = rolling_window(var_2, window)
    function_on_var2 = func(var_2_window, axis=-1)
    return var_1[window - 1:], function_on_var2


def ts2xy(data_frame: pd.DataFrame, x_axis: str) -> Tuple[np.ndarray, np.ndarray
]:
    """
    Decompose a data frame variable to x ans ys

    :param data_frame: (pd.DataFrame) the input data
    :param x_axis: (str) the axis for the x and y output
        (can be X_TIMESTEPS='timesteps', X_EPISODES='episodes' or X_WALLTIME='wa
lltime_hrs')
    :return: (Tuple[np.ndarray, np.ndarray]) the x and y output
    """
    if x_axis == X_TIMESTEPS:
        x_var = np.cumsum(data_frame.l.values)
        y_var = data_frame.r.values
    elif x_axis == X_EPISODES:
        x_var = np.arange(len(data_frame))
        y_var = data_frame.r.values
    elif x_axis == X_WALLTIME:
        # Convert to hours
        x_var = data_frame.t.values / 3600.
        y_var = data_frame.r.values
    else:
        raise NotImplementedError
    return x_var, y_var


def plot_curves(xy_list: List[Tuple[np.ndarray, np.ndarray]],
                x_axis: str, title: str, figsize: Tuple[int, int] = (8, 2)):# -> None:
    """
    plot the curves

    :param xy_list: (List[Tuple[np.ndarray, np.ndarray]]) the x and y coordinate
s to plot
    :param x_axis: (str) the axis for the x and y output
        (can be X_TIMESTEPS='timesteps', X_EPISODES='episodes' or X_WALLTIME='wa
lltime_hrs')
    :param title: (str) the title of the plot
    :param figsize: (Tuple[int, int]) Size of the figure (width, height)
    """

    plt.figure(title, figsize=figsize)
    #breakpoint()
    max_x = max(xy[0][-1] for xy in xy_list)
    min_x = 0
    for (i, (x, y)) in enumerate(xy_list):
        plt.scatter(x, y, s=2)
        # Do not plot the smoothed curve at all if the timeseries is shorter than window size.
        if x.shape[0] >= EPISODES_WINDOW:
            # Compute and plot rolling mean with window of size EPISODE_WINDOW
            x, y_mean = window_func(x, y, EPISODES_WINDOW, np.mean)
            plt.plot(x, y_mean)
    plt.xlim(min_x, max_x)
    plt.title(title)
    plt.xlabel(x_axis)
    plt.ylabel("Episode Rewards")
    plt.tight_layout()


def plot_results(dirs: List[str], num_timesteps: Optional[int],
                 x_axis: str, task_name: str, figsize: Tuple[int, int] = (8, 2)): #-> None:
    """
    Plot the results using csv files from ``Monitor`` wrapper.

    :param dirs: ([str]) the save location of the results to plot
    :param num_timesteps: (int or None) only plot the points below this value
    :param x_axis: (str) the axis for the x and y output
        (can be X_TIMESTEPS='timesteps', X_EPISODES='episodes' or X_WALLTIME='wa
lltime_hrs')
    :param task_name: (str) the title of the task to plot
    :param figsize: (Tuple[int, int]) Size of the figure (width, height)
    """
   
    data_frames = []

    for folder in dirs:
        data_frame = load_results(folder)
        if num_timesteps is not None:
            data_frame = data_frame[data_frame.l.cumsum() <= num_timesteps]
        data_frames.append(data_frame)
    xy_list = [ts2xy(data_frame, x_axis) for data_frame in data_frames]
  #  breakpoint()
    print(xy_list, "xy_list")
    plot_curves(xy_list, x_axis, task_name, figsize)

# Create log dir





th.autograd.set_detect_anomaly(True)
# Parallel environments
def parse_arguments():
    # Command-line flags are defined here.
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', dest='debug',
                              action='store_true',
                              help="Whether to enter debugging mode") 
    parser.add_argument('--param', dest='param', default = 'velocity',
        help="Which control parameter to vary, options are \'velocity\' and \'power \' ")
    parser.add_argument('--verbose', dest='verbose', default = '0',
            help="How much output to display during training ") 
    parser.set_defaults(debug=False, param = 'velocity')

    return parser.parse_args()
def main():
   # num_cpu = 1
    args = parse_arguments()
    debug = args.debug
    parameter = args.param
    num_cpu = 8 # Number of processes to use
    verbose = int(args.verbose)

    log_dir = "training_checkpoints/ppo_triangle_"+parameter
    os.makedirs(log_dir, exist_ok=True)
    model_dir = "trained_models"
    os.makedirs(model_dir, exist_ok=True)
    tb_logdir = 'tensorboard_logs/'
    os.makedirs(tb_logdir, exist_ok=True)
    # Create the vectorized environment
    if parameter == 'velocity':
        frameskip = 1
        env = SubprocVecEnv([make_env(velocitytriangleEnvRLAM(plot = False, frameskip = frameskip, verbose = verbose), i) for i in range(num_cpu)])
        if debug == True:
            print("debugging")
            num_cpu = 1
            
            env = velocitytriangleEnvRLAM(plot = False, frameskip = frameskip, verbose = verbose)
        else:
            env = VecMonitor(env, log_dir)
    elif parameter == 'power':
        frameskip = 2
        env = SubprocVecEnv([make_env(powertriangleEnvRLAM(plot = False, frameskip = frameskip, verbose = verbose), i) for i in range(num_cpu)])
        if debug == True:
            print("debugging")
            num_cpu = 1
            
            env = powertriangleEnvRLAM(plot = False, frameskip = frameskip, verbose = verbose)
        else:
            env = VecMonitor(env, log_dir)

    else:
        raise Exception("Control parameter not found, please enter 'power' or 'velocity' as argument")
    
    

    policy_kwargs = dict(activation_fn=th.nn.Tanh, net_arch=[64, 64])

    model = PPO('MlpPolicy',env, verbose=1, policy_kwargs=policy_kwargs,  tensorboard_log=tb_logdir+"ppo_triangle_"+parameter)
    callback = SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir=log_dir)
    timesteps = 1000000
    model.learn(total_timesteps=timesteps, tb_log_name="ppo_triangle_"+parameter, callback=callback)

    plot_results([log_dir], timesteps, results_plotter.X_TIMESTEPS, "RL_AM")
    plt.show()

    model.save("trained_models/ppo_triangle_" + parameter)


def make_env(env_id, rank, seed=0):

    def _init():
        return env_id
    return _init

if __name__ == "__main__":
    main()