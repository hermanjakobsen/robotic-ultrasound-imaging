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
from helper import register_gripper, plot_joint_pos, plot_forces_and_contact, plot_gripper_position
from demos import controller_demo, \
    contact_btw_probe_and_body_demo, \
    standard_mujoco_py_demo, \
    drop_cube_on_body_demo, \
    change_parameters_of_soft_body_demo, \
    fetch_push_gym_demo


register_env(Ultrasound)
register_env(FetchPush)
register_gripper(UltrasoundProbeGripper)

## Simulations
#controller_demo('UR5e', save_data=False)
#contact_btw_probe_and_body_demo(1, 'main', save_data=False)
#standard_mujoco_py_demo()
#drop_cube_on_body_demo()
#change_parameters_of_soft_body_demo(3)
#fetch_push_gym_demo()

#plot_joint_pos('data/UR5e_joint_pos_controller_test.csv')
#plot_forces_and_contact('data/main_ee_forces_contact_btw_probe_and_body.csv', 'data/main_contact_contact_btw_probe_and_body.csv')
#plot_gripper_position('data/main_gripper_pos_contact_btw_probe_and_body.csv')
