import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def plot_eef_data(data_filename, data_desired_filename, time_filename, pos_title):
    data = pd.read_csv(data_filename, header=None) 
    data_des = pd.read_csv(data_desired_filename, header=None)
    time = pd.read_csv(time_filename, header=None)

    axes = ["x", "y", "z"]
    axes_des = ["x_des", "y_des", "z_des"]               # labels for desired values
    plt.figure()
    for i in range(data.shape[1]):
        plt.plot(time, data[i], label=axes[i])
        plt.plot(time, data_des[i], "--", label=axes_des[i])

    plt.legend()
    plt.xlabel("Completed episode (%)")
    if pos_title:
        plt.title("End-effector position")
    else:
        plt.title("End-effector velocity")
    plt.show()


def plot_eef_pos(pos_filename, pos_desired_filename, time_filename):
    plot_eef_data(pos_filename, pos_desired_filename, time_filename, True)


def plot_eef_vel(vel_filename, vel_desired_filename, time_filename):
    plot_eef_data(vel_filename, vel_desired_filename, time_filename, False)


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
