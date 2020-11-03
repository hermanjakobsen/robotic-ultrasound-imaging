import numpy as np
import matplotlib.pyplot as plt
from fractions import Fraction

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
    assert len(goal_joint_pos) == robot.action_dim

    action = [0 for _ in range(robot.action_dim)]
    curr_joint_pos = robot._joint_positions
    curr_joint_vel = robot._joint_velocities

    for i in range(robot.action_dim):
        if i > len(curr_joint_pos) - 1:
            action[i] = goal_joint_pos[i]
        else:    
            action[i] = (goal_joint_pos[i] - curr_joint_pos[i]) * kp - curr_joint_vel[i] * kd
    
    return action


def transform_ee_frame_axes(measurement):
    # Want z-axis pointing out of probe
    # x (pandaGripper) = -x (probe)
    # y (pandaGripper) = -z (probe)
    # z (pandaGripper) = -y (probe)
    return np.array([-measurement[0], -measurement[2], -measurement[1]])


def plot_joint_pos(joint_pos_filepath):
    joint_pos = np.genfromtxt(joint_pos_filepath, delimiter=',')
    ref_values = np.genfromtxt(joint_pos_filepath.replace('joint_pos', 'ref_values'), delimiter=',')

    t = np.array([i for i in range(joint_pos.shape[0])])

    plt.figure()
    for i in range(joint_pos.shape[1]):
        pos = joint_pos[:, i]
        plt.plot(t, pos, label='joint_' + str(i+1))

    plt.hlines(y=ref_values, xmin=0, xmax=t[-1], linestyle='dotted', colors='k', label='ref')

    plt.yticks(plt.yticks()[0],[r"$" + format(str(Fraction(r/np.pi).limit_denominator(4)), )+ r"\pi$" for r in plt.yticks()[0]])

    plt.grid()
    plt.legend()
    plt.title('Test for joint position controller')
    plt.show()

def plot_forces_and_contact(forces_filepath, contact_filepath):
    contact = np.genfromtxt(contact_filepath, delimiter=',')
    forces = np.genfromtxt(forces_filepath, delimiter=',')

    t = np.array([i for i in range(contact.shape[0])])
    axes = ['x', 'y', 'z']

    plt.figure()
    for i in range(forces.shape[1]):
        force = forces[:, i]
        plt.plot(t, force, label='force_' + axes[i])
    
    plt.plot(t, contact, label='contact')
    plt.grid()
    plt.legend()
    plt.title('Test for contact between gripper and object')
    plt.show()




