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

# Environment specifications
env_id = "Ultrasound"
controller_config = {
    "type": "OSC_POSE",
    "input_max": 1,
    "input_min": -1,
    "output_max": [0.05, 0.05, 0.05, 0.5, 0.5, 0.5],
    "output_min": [-0.05, -0.05, -0.05, -0.5, -0.5, -0.5],
    "kp": 150,
    "damping_ratio": 1,
    "impedance_mode": "fixed",
    "kp_limits": [10, 300],
    "damping_ratio_limits": [0, 2],
    "position_limits": None,
    "orientation_limits": None,
    "uncouple_pos_ori": True,
    "control_delta": True,
    "interpolation": "linear",
    "ramp_ratio": 0.2
}


env_options = {}
env_options["robots"] = "UR5e"
env_options["has_renderer"] = True
env_options["render_camera"] = None
env_options["has_offscreen_renderer"] = False
env_options["use_camera_obs"] = False
env_options["controller_configs"] = controller_config
env_options["control_freq"] = 100
 

env = suite.make(env_id, **env_options)

# reset the environment to prepare for a rollout
obs = env.reset()

done = False
ret = 0.
for t in range(env.horizon):
    action = [0, 0, 0, 0.2, 0 ,0]         # use observation to decide on an action
    obs, reward, done, _ = env.step(action) # play action
    eef_pos = obs["robot0_eef_pos"]
    #print(f"eef: {eef_pos}")
    #print(obs["probe_ori_to_desired"])
    print(reward)
    torso_pos = obs["torso_pos"]
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
