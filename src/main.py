import robosuite as suite
import gym
import os.path as osp
import numpy as np

from robosuite.environments.base import register_env

from my_environments import Ultrasound, FetchPush
from my_models.grippers import UltrasoundProbeGripper
from utils.common import register_gripper
import utils.plot as plt

register_env(Ultrasound)
register_env(FetchPush)
register_gripper(UltrasoundProbeGripper)


## Simulation ##

def get_policy_action(obs):
    # a trained policy could be used here, but we choose a random action
    low, high = env.action_spec
    return np.random.uniform(low, high)

env_id = "Ultrasound"

env_options = {}
env_options["robots"] = "UR5e"
env_options["controller_configs"] = {
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
env_options["control_freq"] = 100
env_options["has_renderer"] = True
env_options["has_offscreen_renderer"] = False
env_options["render_camera"] = None
env_options["use_camera_obs"] = False
env_options["horizon"] = 1000
env_options["early_termination"] = False
env_options["save_data"] = True
 

env = suite.make(env_id, **env_options)

# reset the environment to prepare for a rollout
obs = env.reset()

done = False
ret = 0.
for t in range(env.horizon):
    #print(t)
    #action = [0.0, .0, .0, 0, 0, 0]
    action = [0.2, 0, -0.3, 0., 0 ,0]         # use observation to decide on an action
    if t > 110:
        action = [0.2, 0, -0.005, 0, 0, 0]
    obs, reward, done, _ = env.step(action) # play action
    #contact_force = obs["probe_contact_force"]
    #print(f"eef: {contact_force}")
    #print(obs["probe_ori_to_desired"])
    print(reward)
    #torso_pos = obs["torso_pos"]
    #ret += reward
    env.render()
    if done:
        env.close()
        break
print("rollout completed with return {}".format(ret))


## Plotting ##
#plt.plot_eef_pos("simulation_data/ee_pos_1", "simulation_data/ee_traj_pos_1", "simulation_data/time_1")
#plt.plot_eef_vel("simulation_data/ee_vel_1", "simulation_data/ee_traj_vel_1", "simulation_data/time_1")
#plt.plot_contact_and_contact_force("simulation_data/is_contact_1", "simulation_data/ee_z_contact_force_1", \
#    "simulation_data/ee_z_desired_contact_force_1", "simulation_data/time_1")
#plt.plot_rewards("reward_data/pos_1", "reward_data/ori_1", "reward_data/vel_1", "reward_data/force_1", "simulation_data/time_1")