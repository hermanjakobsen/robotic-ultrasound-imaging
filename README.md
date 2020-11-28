# Installation 
First download [MuJoCo 2.0](https://www.roboti.us/index.html) and unzip its contents into `~/.mujoco/mujoco200`, and copy your MuJoCo license key into `~/.mujoco/mjkey.txt`. You can obtain a license key from [here](https://www.roboti.us/license.html).

After setting up mujoco, the rest of the requirements can be installed with
```
pip3 install -r requirements.txt
```

## mujoco-py quick fix 
Rendering `skin` for `composite` (soft) objects will result in a [segmentation fault](https://github.com/openai/mujoco-py/issues/373). A quick fix to this problem is given by [fantasyRqq](https://github.com/fantasyRqg/mujoco-py). To resolve the error, do the following
```
pip3 uninstall  -y mujoco_py

git clone git@github.com:fantasyRqg/mujoco-py.git

cd mujoco-py

python3 setup.py build

python3 setup.py install
```
