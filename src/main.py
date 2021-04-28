import robosuite as suite
import gym
import os.path as osp
import numpy as np

from robosuite.environments.base import register_env
import robosuite.utils.transform_utils as T

from my_environments import Ultrasound, FetchPush, HMFC
from my_models.grippers import UltrasoundProbeGripper
from utils.common import register_gripper, get_elements_in_obs
import utils.plot as plt

register_env(Ultrasound)
register_env(FetchPush)
register_env(HMFC)
register_gripper(UltrasoundProbeGripper)


## Simulation ##

def run_simulation():
    env_id = "Ultrasound"

    env_options = {}
    env_options["robots"] = "Panda"
    env_options["gripper_types"] = "UltrasoundProbeGripper"
    env_options["controller_configs"] = {
        "type": "OSC_POSE",
        "input_max": 1,
        "input_min": -1,
        "output_max": [0.05, 0.05, 0.05, 0.5, 0.5, 0.5],
        "output_min": [-0.05, -0.05, -0.05, -0.5, -0.5, -0.5],
        "kp": 300,
        "damping_ratio": 1,
        "impedance_mode": "tracking",
        "kp_limits": [0, 1000],
        "damping_ratio_limits": [0, 2],
        "position_limits": None,
        "orientation_limits": None,
        "uncouple_pos_ori": True,
        "control_delta": False,
        "interpolation": None,
        "ramp_ratio": 0.2
    }
    env_options["control_freq"] = 100
    env_options["has_renderer"] = True
    env_options["has_offscreen_renderer"] = False
    env_options["render_camera"] = None
    env_options["use_camera_obs"] = False
    env_options["use_object_obs"] = False
    env_options["horizon"] = 500
    env_options["early_termination"] = True
    env_options["save_data"] = False

    env = suite.make(env_id, **env_options)

    traj = env.get_trajectory()

    # reset the environment to prepare for a rollout
    obs = env.reset()

    done = False
    ret = 0.
    
    for t in range(env.horizon):
        print(t)

        action = [1 for i in range(6)]

        obs, reward, done, _ = env.step(action) # play action

        #print(reward)
        #ret += reward
        env.render()
        if done:
            env.close()
            break
    print("rollout completed with return {}".format(ret))


def test_hmfc():
    env_id = "HMFC"

    env_options = {}
    env_options["robots"] = "Panda"
    env_options["gripper_types"] = "UltrasoundProbeGripper"
    env_options["controller_configs"] = {
        "type": "HMFC",
        "input_max": 1,
        "input_min": -1,
        "output_max": [0.05, 0.05, 0.05, 0.5, 0.5, 0.5],
        "output_min": [-0.05, -0.05, -0.05, -0.5, -0.5, -0.5],
        "interpolation": None,
    }
    env_options["control_freq"] = 100
    env_options["has_renderer"] = True
    env_options["has_offscreen_renderer"] = False
    env_options["render_camera"] = None
    env_options["use_camera_obs"] = False
    env_options["use_object_obs"] = False
    env_options["horizon"] = 1000

    env = suite.make(env_id, **env_options)

    # reset the environment to prepare for a rollout
    obs = env.reset()

    done = False
    ret = 0.
    
    for t in range(env.horizon):
        action = []
        obs, reward, done, _ = env.step(action) # play action

        #print(reward)
        #ret += reward
        env.render()
        if done:
            env.close()
            break
    print("rollout completed with return {}".format(ret))

def plot_data(run_num):
    num = str(run_num)
    plt.plot_eef_pos("simulation_data/ee_pos_" + num + ".csv", "simulation_data/ee_goal_pos_" + num + ".csv", "simulation_data/time_" + num + ".csv")
    plt.plot_eef_vel("simulation_data/ee_vel_" + num + ".csv", "simulation_data/ee_running_mean_vel_" + num + ".csv", "simulation_data/ee_goal_vel_" + num + ".csv", "simulation_data/time_" + num + ".csv")
    plt.plot_contact_and_contact_force("simulation_data/ee_z_contact_force_" + num + ".csv", "simulation_data/ee_z_running_mean_contact_force_" + num + ".csv", "simulation_data/ee_z_desired_contact_force_" + num + ".csv", "simulation_data/time_" + num + ".csv")
    plt.plot_rewards("reward_data/pos_" + num + ".csv", "reward_data/ori_" + num + ".csv", "reward_data/vel_" + num + ".csv", "reward_data/force_" + num + ".csv", "simulation_data/time_" + num + ".csv")
    #plt.plot_controller_delta("policy_data/action_" + num + ".csv", "simulation_data/time_" + num + ".csv")
    plt.plot_controller_gains("policy_data/action_" + num + ".csv", "simulation_data/time_" + num + ".csv")
    #plt.plot_qpos("simulation_data/q_pos_" + num + ".csv", "simulation_data/time_" + num + ".csv")
    #plt.plot_qtorques("simulation_data/q_torques_" + num + ".csv", "simulation_data/time_" + num + ".csv")


#run_simulation()
#test_hmfc()
#plot_data(15)

