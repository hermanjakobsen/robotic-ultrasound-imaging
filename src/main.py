import robosuite as suite
import gym
import os.path as osp
import numpy as np

from robosuite.environments.base import register_env
from robosuite import load_controller_config
from robosuite.wrappers import GymWrapper

from stable_baselines3 import PPO
from stable_baselines3.common.save_util import save_to_zip_file, load_from_zip_file
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from my_environments import Ultrasound, FetchPush
from my_models.grippers import UltrasoundProbeGripper
from utils.common import register_gripper
from utils.quaternion import q_log
from demos import contact_btw_probe_and_body_demo, standard_mujoco_py_demo, change_parameters_of_soft_body_demo

import yaml

register_env(Ultrasound)
register_env(FetchPush)
register_gripper(UltrasoundProbeGripper)


# USED FOR TESTING
def get_policy_action(obs):
    # a trained policy could be used here, but we choose a random action
    low, high = env.action_spec
    return np.random.uniform(low, high)

with open("rl_config.yaml", 'r') as stream:
    config = yaml.safe_load(stream)

# Environment specifications
env_options = config["robosuite"]
env_id = env_options.pop("env_id")

env = suite.make(env_id, **env_options)

# reset the environment to prepare for a rollout
obs = env.reset()

done = False
ret = 0.
for t in range(env.horizon):
    action = [0, 0, 0, -1, 0.5 ,0.2]         # use observation to decide on an action
    obs, reward, done, _ = env.step(action) # play action
    #print(obs)
    ret += reward
    env.render()
print("rollout completed with return {}".format(ret))

## Simulations
#controller_demo('UR5e', save_data=False)
#contact_btw_probe_and_body_demo(1, 'main', save_data=False)
#standard_mujoco_py_demo()
#change_parameters_of_soft_body_demo(3)

#plot_joint_pos('data/UR5e_joint_pos_controller_test.csv')
#plot_forces_and_contact('data/main_ee_forces_contact_btw_probe_and_body.csv', 'data/main_contact_contact_btw_probe_and_body.csv')
#plot_gripper_position('data/main_gripper_pos_contact_btw_probe_and_body.csv')
