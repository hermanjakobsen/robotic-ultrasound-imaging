import robosuite as suite

from robosuite.environments.base import register_env
import robosuite.utils.transform_utils as T

from my_environments import Ultrasound, HMFC
from my_models.grippers import UltrasoundProbeGripper
from utils.common import register_gripper
import utils.plot as plt
import utils.error as error 

register_env(Ultrasound)
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
        "impedance_mode": "fixed",
        "kp_limits": [0, 500],
        "kp_input_max": 1,
        "kp_input_min": 0,
        "damping_ratio_limits": [0, 2],
        "position_limits": None,
        "orientation_limits": None,
        "uncouple_pos_ori": True,
        "control_delta": True,
        "interpolation": None,
        "ramp_ratio": 0.2
    }
    env_options["control_freq"] = 500
    env_options["has_renderer"] = True
    env_options["has_offscreen_renderer"] = False
    env_options["render_camera"] = None
    env_options["use_camera_obs"] = False
    env_options["use_object_obs"] = False
    env_options["horizon"] = 1000
    env_options["early_termination"] = False
    env_options["save_data"] = False
    env_options["torso_solref_randomization"] = False
    env_options["initial_probe_pos_randomization"] = False
    env_options["deterministic_trajectory"] = False
    env_options["use_box_torso"] = True

    env = suite.make(env_id, **env_options)

    # reset the environment to prepare for a rollout
    obs = env.reset()

    done = False
    ret = 0.
    
    for t in range(env.horizon):
        action = [0.0, 0, 0, 0, 0, 0]
        obs, reward, done, _ = env.step(action) # play action
        ret += reward
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
    env_options["control_freq"] = 500
    env_options["has_renderer"] = True
    env_options["has_offscreen_renderer"] = False
    env_options["render_camera"] = None
    env_options["use_camera_obs"] = False
    env_options["horizon"] = 1000
    env_options["save_data"] = True

    env = suite.make(env_id, **env_options)

    # reset the environment to prepare for a rollout
    obs = env.reset()
    done = False
    ret = 0.
    
    for t in range(env.horizon):
        action = []
        obs, reward, done, _ = env.step(action) # play action
        env.render()
        if done:
            env.close()
            break
    print("rollout completed with return {}".format(ret))


## SIMULATION TEST ##
#run_simulation()
#test_hmfc()

## PLOTTING ##
#plt.plot_sim_data("tracking", "test", True)
#plt.plot_training_rew_mean("training_rew_mean/tracking.csv", "training_rew_mean/variable_z.csv", "training_rew_mean/wrench.csv")
#error.calculate_error_metrics("variable_z")
#plt.plot_hmfc_data(1)
