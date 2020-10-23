import numpy as np
from mujoco_py import MjSimState
from robosuite.models.grippers import GRIPPER_MAPPING

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


def relative2absolute_joint_pos_commands(goal_joint_pos, robot, kp, kd):
    assert len(goal_joint_pos) == robot.dof

    action = [0 for _ in range(robot.dof)]
    curr_joint_pos = robot._joint_positions
    curr_joint_vel = robot._joint_velocities

    for i in range(robot.dof):
        action[i] = (goal_joint_pos[i] - curr_joint_pos[i]) * kp - curr_joint_vel[i] * kd
    
    return action