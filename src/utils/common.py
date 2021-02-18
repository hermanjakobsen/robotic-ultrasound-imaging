import numpy as np
from mujoco_py import MjSimState, MjViewer, MjSim

from robosuite.models.grippers import GRIPPER_MAPPING


def register_gripper(gripper_class):
    GRIPPER_MAPPING[gripper_class.__name__] = gripper_class


def set_initial_robot_state(sim, robot):
    ''' Used for setting initial state when simulating with mujoco-py directly '''

    old_state = sim.get_state()

    old_qvel = old_state.qvel[robot.dof:]
    old_qpos = old_state.qpos[robot.dof:] if robot.gripper.dof < 1 else old_state.qpos[robot.dof-1:]

    init_qvel = [0 for _ in range(robot.dof)]
    init_model_qvel = np.concatenate((init_qvel, old_qvel), axis=0)
    init_model_qpos = np.concatenate((robot.init_qpos, old_qpos), axis=0)
        
    new_state = MjSimState(old_state.time, init_model_qpos, init_model_qvel, old_state.act, old_state.udd_state)
    sim.set_state(new_state)
    sim.forward()


def create_mjsim_and_viewer(env):
    world = env.model 

    model = world.get_model(mode="mujoco_py")
    sim = MjSim(model)
    set_initial_robot_state(sim, env.robots[0])
    viewer = MjViewer(sim)

    return sim, viewer


def transform_ee_frame_axes(measurement):
    # Want z-axis pointing out of probe
    # x (pandaGripper) = -x (probe)
    # y (pandaGripper) = -z (probe)
    # z (pandaGripper) =  y (probe)
    return np.array([-measurement[0], -measurement[2], measurement[1]])