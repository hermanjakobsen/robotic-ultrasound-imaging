import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def plot_eef_pos(pos_filename, pos_desired_filename, time_filename):
    pos = pd.read_csv(pos_filename, header=None) 
    pos_des = pd.read_csv(pos_desired_filename, header=None)
    time = pd.read_csv(time_filename, header=None)

    labels = ["x", "y", "z"]
    labels_des = ["x_goal", "y_goal", "z_goal"]               # labels for desired values

    plt.figure()
    for i in range(len(pos.columns)):
        plt.plot(time, pos[i], label=labels[i])
        plt.plot(time, pos_des[i], "--", label=labels_des[i])

    plt.legend()
    plt.xlabel("Completed episode (%)")
    plt.title("End-effector position")

    plt.show()


def plot_eef_vel(vel_filename, vel_desired_filename, time_filename):
    vel = pd.read_csv(vel_filename, header=None)
    vel = vel.apply(np.linalg.norm, axis=1)
    vel_des = pd.read_csv(vel_desired_filename, header=None)
    time = pd.read_csv(time_filename, header=None)

    plt.figure()
    plt.plot(time, vel, label="vel")
    plt.plot(time, vel_des, label="vel_goal")

    plt.legend()
    plt.xlabel("Completed episode (%)")
    plt.title("End-effector velocity")

    plt.show()


def plot_contact_and_contact_force(contact_filename, force_filename, desired_force_filename, time_filename):
    contact = pd.read_csv(contact_filename, header=None) 
    force = pd.read_csv(force_filename, header=None)
    desired_force = pd.read_csv(desired_force_filename, header=None)
    time = pd.read_csv(time_filename, header=None) 

    plt.figure()
    plt.plot(time, contact, label="is_contact")
    plt.plot(time, force, label="z_force")
    plt.plot(time, desired_force, "--", label="desired_force")

    plt.legend()
    plt.xlabel("Completed episode (%)")
    plt.title("Contact force")
    plt.show()


def plot_rewards(pos_reward_filename, ori_reward_filename, vel_reward_filename, force_reward_filename, time_filename):
    pos_reward = pd.read_csv(pos_reward_filename, header=None)
    ori_reward = pd.read_csv(ori_reward_filename, header=None)
    vel_reward = pd.read_csv(vel_reward_filename, header=None)
    force_reward = pd.read_csv(force_reward_filename, header=None)

    time = pd.read_csv(time_filename, header=None)

    plt.figure()
    plt.plot(time, pos_reward, label="pos")
    plt.plot(time, ori_reward, label="ori")
    plt.plot(time, vel_reward, label="vel")
    plt.plot(time, force_reward, label="force")
    
    plt.legend()
    plt.xlabel("Completed episode (%)")
    plt.title("Rewards")
    plt.show()


def plot_qpos(qpos_filename, time_filename):
    qpos = pd.read_csv(qpos_filename, header=None)
    time = pd.read_csv(time_filename, header=None)

    plt.figure()
    for i in range(len(qpos.columns)):
        plt.plot(time, qpos[i], label="q" + str(i + 1))
    
    plt.legend()
    plt.xlabel("Completed episode (%)")
    plt.title("Joint positions")
    plt.show()


def plot_controller_actions(action_filename, time_filename):
    action = pd.read_csv(action_filename, header=None)
    time = pd.read_csv(time_filename, header=None)

    controller_action = action.iloc[:, -6:]
    controller_action.columns = [i for i in range(len(controller_action.columns))]  # reset column indices

    labels = ["x", "y", "z", "ax", "ay", "az"]

    for i in range(len(controller_action.columns)):
        plt.figure(i)
        plt.plot(time, controller_action[i])
        plt.xlabel("Completed episode (%)")
        plt.title("Controller action, " + str(labels[i]))

    plt.show()


def plot_controller_gains(action_filename, time_filename):
    action = pd.read_csv(action_filename, header=None)
    time = pd.read_csv(time_filename, header=None)

    labels = ["x", "y", "z", "ax", "ay", "az"]

    # variable mode
    if len(action.columns) == 18:
        damping_ratio = action.iloc[:, :6]
        kp = action.iloc[:, 6:12]

        # reset column indices
        damping_ratio.columns = [i for i in range(len(damping_ratio.columns))]
        plt.figure(0)
        for i in range(len(damping_ratio.columns)):
            plt.plot(time, damping_ratio[i], label=labels[i])

        plt.legend()
        plt.xlabel("Completed episode (%)")
        plt.title("Controller damping ratio")
    
    # variable_kp mode
    elif len(action.columns) == 12:
        kp = action.iloc[:, :6]

    else:
        print("Unknown action dim!")
        return

    # reset column indices
    kp.columns = [i for i in range(len(kp.columns))]  

    plt.figure(1)
    for i in range(len(kp.columns)):
        plt.plot(time, kp[i], label=labels[i])

    plt.legend()
    plt.xlabel("Completed episode (%)")
    plt.title("Controller gains (Kp)")

    plt.show()
