import numpy as np
import matplotlib.pyplot as plt
from fractions import Fraction

from mujoco_py import MjSimState, MjSim, MjViewer

from robosuite.models.grippers import GRIPPER_MAPPING


def print_world_xml_and_soft_torso_params(world):
    soft_torso = world.mujoco_objects[0]
    composite = soft_torso._get_composite_element()
    print(world.get_xml())
    print(composite.get('solrefsmooth'))
    print('\n\n')


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


def plot_forces_and_contact(forces_filepath, contact_filepath, time_filepath):
    contact = np.genfromtxt(contact_filepath, delimiter=',')
    forces = np.genfromtxt(forces_filepath, delimiter=',')
    time = np.genfromtxt(time_filepath, delimiter=',')
    time = time - time[300]
    time = time[300:]

    axes = ['x', 'y', 'z']
    plt.figure()
    for i in range(forces.shape[1]):
        force = forces[300:, i]
        plt.plot(time, force, label=axes[i])
    
    plt.plot(time, contact[300:], label='contact')
    plt.grid()
    plt.legend()
    plt.xlabel('Time (s)')
    plt.ylabel('Force (N)')
    plt.title('Interaction forces between the probe soft body')
    plt.show()


def plot_gripper_position(position_filepath, time_filepath):
    position = np.genfromtxt(position_filepath, delimiter=',')
    time = np.genfromtxt(time_filepath, delimiter=',')
    time = time - time[300]
    time = time[300:]

    axes = ['x', 'y', 'z']
    plt.figure()
    for i in range(position.shape[1]):
        pos = position[300:, i]
        plt.plot(time, pos, label=axes[i])
    
    plt.grid()
    plt.legend()
    plt.xlabel('Time (s)')
    plt.ylabel('Position (m)')
    plt.title('Position of gripper')
    plt.show()


