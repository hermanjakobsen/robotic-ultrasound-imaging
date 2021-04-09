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
    plt.plot(time, contact, label="contact")
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

    labels = ["pos", "ori", "vel", "force"]
    colors = ['red', 'black', 'blue', 'green']
    rewards = [pos_reward, ori_reward, vel_reward, force_reward]

    fig, axs = plt.subplots(len(rewards))
    fig.suptitle("Rewards")
    for i in range(len(rewards)):
        axs[i].plot(time, rewards[i], color=colors[i])
        axs[i].set_title(labels[i])

    plt.xlabel("Completed episode (%)")
    plt.show()


def plot_qpos(qpos_filename, time_filename):
    qpos = pd.read_csv(qpos_filename, header=None)
    time = pd.read_csv(time_filename, header=None)

    colors = ['red', 'black', 'blue', 'brown', 'green', "purple", "olive"]

    fig, axs = plt.subplots(len(qpos.columns))
    fig.suptitle("Joint positions")
    for i in range(len(qpos.columns)):
        axs[i].plot(time, qpos[i], color=colors[i])
        axs[i].set_title("q" + str(i + 1))
    
    plt.xlabel("Completed episode (%)")
    plt.show()



def plot_qtorques(qtorque_filename, time_filename):
    torques = pd.read_csv(qtorque_filename, header=None)
    time = pd.read_csv(time_filename, header=None)

    colors = ['red', 'black', 'blue', 'brown', 'green', "purple", "olive"]

    fig, axs = plt.subplots(len(torques.columns))
    fig.suptitle("Joint torques")
    for i in range(len(torques.columns)):
        axs[i].plot(time, torques[i], color=colors[i])
        axs[i].set_title("q" + str(i + 1))
        axs[i].set(ylabel="N")
    
    plt.xlabel("Completed episode (%)")
    plt.show()


def plot_controller_delta(action_filename, time_filename):
    action = pd.read_csv(action_filename, header=None)
    time = pd.read_csv(time_filename, header=None)

    controller_delta = action.iloc[:, -6:]
    controller_delta.columns = [i for i in range(len(controller_delta.columns))]  # reset column indices

    labels = ["x", "y", "z", "ax", "ay", "az"]

    fig , axs = plt.subplots(len(controller_delta.columns))
    fig.suptitle("Controller action")
    for i in range(len(controller_delta.columns)):
        axs[i].plot(time, controller_delta[i])
        axs[i].set_title(labels[i])
    plt.xlabel("Completed episode (%)")
    plt.show()


def plot_controller_gains(action_filename, time_filename):
    action = pd.read_csv(action_filename, header=None)
    time = pd.read_csv(time_filename, header=None)

    labels = ["x", "y", "z", "ax", "ay", "az"]
    colors = ['red', 'black', 'blue', 'brown', 'green', "purple"]

    # variable mode
    if len(action.columns) == 18:
        damping_ratio = action.iloc[:, :6]
        kp = action.iloc[:, 6:12]

        # reset column indices
        damping_ratio.columns = [i for i in range(len(damping_ratio.columns))]
        fig1, axs1 = plt.subplots(len(damping_ratio.columns))
        fig1.suptitle("Controller damping ratio")
        for i in range(len(damping_ratio.columns)):
            axs1[i].plot(time, damping_ratio[i], color=colors[i])
            axs1[i].set_title(labels[i])

        plt.xlabel("Completed episode (%)")
    
    # variable_kp mode
    elif len(action.columns) == 12:
        kp = action.iloc[:, :6]

    else:
        print("Unknown action dim!")
        return

    # reset column indices
    kp.columns = [i for i in range(len(kp.columns))]  

    fig2, axs2 = plt.subplots(len(kp.columns))
    fig2.suptitle("Controller gains (Kp)")
    for i in range(len(kp.columns)):
        axs2[i].plot(time, kp[i], color=colors[i])
        axs2[i].set_title(labels[i])

    plt.xlabel("Completed episode (%)")

    plt.show()
