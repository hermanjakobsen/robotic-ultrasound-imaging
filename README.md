# Installation 
The framework has been tested to run with Ubuntu18.04 and python3.8. 
## MuJoCo 2.0
Download [MuJoCo 2.0](https://www.roboti.us/index.html) and unzip its contents into `~/.mujoco/mujoco200`.  A license key can be obtained from [here](https://www.roboti.us/license.html). Copy your MuJoCo license key into `~/.mujoco/mjkey.txt`. The finalized folder structure should look like

```
~/.mujoco
│   mjkey.txt   
│    
└───mujoco200
│   │   bin
│   │   doc
│   │   include
|   |   model
|   |   sample
```
Lastly, add the following line to the bottom of `~/.bashrc`
```
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/hermankj/.mujoco/mujoco200/bin
```
## Using pip
To avoid package conflictions, it is smart to create a virtual environment. Run the following command to set up a virtual environment
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
pip3 install -r requirements.txt
```
NOTE: The `mujoco-py` package may require additional system dependencies. The full installation process for this package can be found [here](https://github.com/openai/mujoco-py).

# Known issues

## Offscreen rendering
Problems with the GLEW library can occur if you try to benefit from offscreen rendering while using Ubuntu20.04 with a dedicated GPU. This is a known problem with the `mujoco-py` package, and a discussion of solutions can be found [here](https://github.com/openai/mujoco-py/issues/408).
 