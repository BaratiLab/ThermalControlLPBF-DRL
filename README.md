# Thermal Control of Laser Powder Bed Fusion using Deep Reinforcement Learning


This repository is the implementation of the paper "Thermal Control of Laser Powder Bed Fusion Using Deep Reinforcement Learning", linked [here](https://www.sciencedirect.com/science/article/pii/S2214860421001986). The project makes use of the Deep Reinforcement Library [stable-baselines3](https://github.com/DLR-RM/stable-baselines3) to derive a control policy that maximizes melt pool depth consistency.
![drl_am](https://user-images.githubusercontent.com/45725578/122481556-e3491f00-cf9c-11eb-89c4-f8014b72b9a1.gif)

## Simulation Framework
The Repeated Usage of Stored Line Solutions (RUSLS) method proposed by Wolfer et al. is used to simulate the temperature dynamics in this work. More detail can be found in the following paper:

* Fast solution strategy for transient heat conduction for arbitrary scan paths in additive manufacturing, Additive Manufacturing, Volume 30, 2019 [(link)](https://www.sciencedirect.com/science/article/pii/S2214860419303446)


## Prerequisites 
The following packages are required in order to run the associated code:

* ```gym==0.17.3```
* ```torch==1.5.0```
* ```stable_baselines3==0.7.0```
* ```numba==0.50.1```


These packages can be installed independently, or all at once by running ```pip install -r requirements.txt```. We recommend that these packages are installed in a new conda environment to avoid clashes with existing package installations. Instructions on defining a new conda environment can be found [here](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html).

## Usage
The overall workflow for this project first defines a ```gym``` environment based on the desired scan path, then performs Proximal Policy Optimization to derive a suitable control policy based on the environment. This is done through the following:

### Overview
* ```EagarTsaiModel.py```: implements the RUSLS solution to the Rosenthal equation, as proposed by [Wolfer et al.](https://www.sciencedirect.com/science/article/pii/S2214860419303446)
* ```power_square_gym.py```, ```power_triangle_gym.py```, ```velocity_square_gym.py```, ```velocity_triangle_gym.py```: Defines custom ```gym``` environments for the respective scan paths and control variables. ```square``` is used as shorthand for the predefined horizontal cross-hatching path and ```triangle``` is used as shorthand for the predefined concentric triangular path. 
* ```RL_learn_square.py```, ```RL_learn_triangle.py``` performs Proximal Policy Optimization on the respective scan paths, with command line arguments to change which control parameter is varied.  
* ```evaluate_learned_policy.py``` runs a derived control policy on a specific environment. The environment is specified using command line arguments detailed below.  


### Testing a trained model

To test a trained model on a specific combination of scan path and control parameter, enter this command:

```python evaluate_learned_policy.py --path [scan_path] -param [parameter]``` 

 Note: ```[scan_path]```  should be replaced by ```square``` for the horizontal cross-hatching scan path and ```triangle``` for the concentric triangular path. ```[parameter]``` should be replaced by ```power``` to specify power as a control parameter, and ```velocity``` to specify velocity as a control parameter.
  
  Upon running this command, you will be prompted to enter the path to the ```.zip``` file for the trained model. 


Once the evaluation is complete, the results are stored in the folder ```results/[scan_path]_[parameter]_control/```. This folder will contain plots of the variation of the melt depth and control parameters over time, as well as their raw values for later analysis. 

### Training a new model
In order to train a new model based on the predefined horizontal cross-hatching scan path, enter the command:


```python RL_learn_square.py --param [parameter]```


Here, ```[parameter]``` should be replaced by the control parameter desired. The possible options are ```power``` and ```velocity```.


The process is similar for the predefined concentric triangular scan path. To train a new model, enter the command:


```python RL_learn_triangle.py --param [parameter]```

Again, ```[parameter]``` should be replaced by the control parameter desired. The possible options are ```power``` and ```velocity```.

During training, intermediate model checkpoints will be saved at

```training_checkpoints/ppo_[scan_path]_[parameter]/best_model.zip```
 
 
 
 At the conclusion of training, the finished model is stored at 
 
 ```trained_models/ppo_[scan_path]_[parameter].zip```
 
 






### Defining a custom domain
#### Changing the powder bed features
In order to define a custom domain for use with a different problem configuration, the ```EagarTsaiModel.py``` file should be edited directly. Within the ```EagarTsai()``` class instantiation, the thermodynamic properties and domain dimensions can be specified. Additionally, the resolution and boundary conditions can be provided as arguments to the ```EagarTsai``` class. ```bc = 'flux'``` and ```bc = 'temp'``` implements an adiabatic and constant temperature boundary condition respectively. 

#### Changing the scan path

A new scan path can be defined by creating a new custom gym environment, and writing a custom ```step()``` function to represent the desired scan path, similar to the  ```[parameter]_[scan_path]_gym.py``` scripts in this repository. Considerations for both how the laser moves during a single segment and the placement of each segment within the overall path should be described in this function. More detail on the gym framework for defining custom environments can be found [here](https://stable-baselines.readthedocs.io/en/master/guide/custom_env.html).


### Monitoring the training process with TensorBoard
Tensorboard provides resources for monitoring various metrics of the PPO training process, and can be installed using ```pip install tensorboard```. To open the tensorboard dashboard, enter the command:

```tensorboard --log_dir ./tensorboard_logs/ppo_[scan_path]_[parameter]/ppo_[scan_path]_[parameter]_[run_ID]```

Tensorboard log files are periodically saved during training, with information on various loss metrics, cumulative reward.
