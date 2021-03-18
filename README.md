# Installation 
The framework has been tested to run with Ubuntu20.04 and python3.8. 
## MuJoCo 2.0
Download [MuJoCo 2.0](https://www.roboti.us/index.html) and unzip its contents into `~/.mujoco/mujoco200`.  A license key can be obtained from [here](https://www.roboti.us/license.html). Copy your MuJoCo license key into `~/.mujoco/mjkey.txt`. The finalized folder structure should look like

```
~/.mujoco
│   mjkey.txt   
└───mujoco200
│   │   bin
│   │   doc
│   │   include
|   |   model
|   |   sample
```
Lastly, add the following line to the bottom of `~/.bashrc`
```
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:<path_to_home>/.mujoco/mujoco200/bin
```
## Virtual environment
To avoid package conflicts, it is smart to create a virtual environment. It is possible to create a virtual environment using either pip or conda. Some packages require additional system dependencies, as stated in the "Notes" section. 

## Conda
First, follow the [guidelines](https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html) on how to install Miniconda. The virtual environment can then be created by running 
```
conda env create -f conda.yaml
```
To activate the virtual environment, run 
```
conda activate rl_ultrasound
```

## Pip
Run the following command to set up a virtual environment
```
sudo apt install python-virtualenv

python3 -m venv venv
```
The virtual environment can be activated with
```
source venv/bin/activate
```
The required packages can then be installed with 
```
pip3 install wheel  
pip3 install -r requirements.txt
```
## Notes
The `mujoco-py` package may require additional system dependencies. The full installation process for this package can be found [here](https://github.com/openai/mujoco-py). For instance, if you are going to use a GPU to train RL agents, the `patchelf` package is needed for installing `mujoco-py`:
```
sudo apt-get install -y patchelf
```

 # Train and run an RL agent
 It is possible to train an RL agent to perform the ultrasound task, where the framework has been integrated with the RL algorithms from [stable-baselines](https://github.com/DLR-RM/stable-baselines3). Different settings (e.g. object observations and controller specifications) can be specified in `rl_config.yaml`. Note that the config file is not complete, hence there exists numerous of other settings and hyperparameters that are not specifed in the file. For these parameters, the default values are used. 

 To train (or run) an agent, it is as simple as running
 ```
 python3 rl.py
 ``` 
 Whether to train an agent, or evaluate a trained agent, is specified in `rl_config.yaml`.

 NOTE: The ultrasound task is not complete. That is, the reward function is not properly defined. Hence, as of now, the agent is not able to learn to perform an ultrasound examination correctly.