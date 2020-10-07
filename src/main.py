import numpy as np

from mujoco_py import MjSim, MjViewer, MjSimState

import robosuite as suite
from robosuite.environments.base import register_env
from robosuite.models.grippers import GRIPPER_MAPPING

from my_environments import Ultrasound
from my_models.grippers import UltrasoundProbeGripper


def set_initial_state(sim, old_state, robot):
    ''' Used for setting initial state when simulating with mujoco-py directly '''

    old_qvel = old_state.qvel[robot.dof:]
    old_qpos = old_state.qpos[robot.dof:] if robot.gripper.dof < 1 else old_state.qpos[robot.dof-1:]

    init_qvel = [0 for _ in range(robot.dof)]
    initial_model_qvel = np.concatenate((init_qvel, old_qvel), axis=0)
    initial_model_qpos = np.concatenate((robot.init_qpos, old_qpos), axis=0)
        
    new_state = MjSimState(old_state.time, initial_model_qpos, initial_model_qvel, old_state.act, old_state.udd_state)
    sim.set_state(new_state)
    sim.forward()


def register_gripper(gripper_class):
    GRIPPER_MAPPING[gripper_class.__name__] = gripper_class


def robosuite_simulation(env, sim_time):
    # Reset the env
    env.reset()

    # Get action limits
    low, high = env.action_spec

    # Run random policy
    for t in range(sim_time):
        if env.done:
            break
        env.render()
        action = np.random.uniform(low, high)
        observation, reward, done, info = env.step(action)

    # close window
    env.close()


def mujoco_py_simulation(env, sim_time):
    world = env.model 
    model = world.get_model(mode="mujoco_py")

    sim = MjSim(model)
    set_initial_state(sim, sim.get_state(), env.robots[0])
    viewer = MjViewer(sim)

    for _ in range(sim_time):
        sim.step()
        viewer.render()
 

register_env(Ultrasound)
register_gripper(UltrasoundProbeGripper)

'''
env_test = suite.make(
    'Lift',
    robots='Panda',
    gripper_types="PandaGripper",
)
'''

env = suite.make(
            'Ultrasound',
            robots='UR5e',
            gripper_types='UltrasoundProbeGripper',
            has_renderer = True,
            has_offscreen_renderer= False,
            use_camera_obs=False,
            use_object_obs=False,
            control_freq = 50,
            render_camera = None,      
        )


robosuite_simulation(env, env.horizon)
#mujoco_py_simulation(env_test, env_test.horizon)