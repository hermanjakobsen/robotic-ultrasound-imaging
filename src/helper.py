import numpy as np
import matplotlib.pyplot as plt
from fractions import Fraction

from mujoco_py import MjSimState, MjSim, MjViewer

from robosuite.models.grippers import GRIPPER_MAPPING


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


def print_world_xml_and_soft_torso_params(world):
    soft_torso = world.other_mujoco_objects['soft_torso']
    composite = soft_torso._get_composite_element()
    print(composite.get('solrefsmooth'))
    print(world.get_xml())


def create_mjsim_and_viewer(env):
    world = env.model 

    model = world.get_model(mode="mujoco_py")
    sim = MjSim(model)
    set_initial_robot_state(sim, env.robots[0])
    viewer = MjViewer(sim)

    return sim, viewer


def capture_image_frame(viewer, folderpath):
    # better solution would be to use off-screen renderer.  
    img = viewer._read_pixels_as_in_window()
    plt.imsave(folderpath + 'frame'+ '{0:06}'.format(viewer._image_idx) + '.png', img)
    viewer._image_idx += 1


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
        plt.plot(t, force, label=axes[i])
    
    plt.plot(t, contact, label='contact')
    plt.grid()
    plt.legend()
    plt.xlabel('Time (steps)')
    plt.ylabel('Force (N)')
    plt.title('Interaction forces between the probe and the end effector base')
    plt.show()


def plot_gripper_position(filepath):
    position = np.genfromtxt(filepath, delimiter=',')

    t = np.array([i for i in range(position.shape[0])])
    axes = ['x', 'y', 'z']

    plt.figure()
    for i in range(position.shape[1]):
        force = position[:, i]
        plt.plot(t, force, label=axes[i])
    
    plt.grid()
    plt.legend()
    plt.xlabel('Time (steps)')
    plt.ylabel('Position (m)')
    plt.title('Position of gripper')
    plt.show()


