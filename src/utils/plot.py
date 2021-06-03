import numpy as np
import matplotlib.pyplot as plt
from numpy.lib import save
import pandas as pd
import seaborn as sns
import matplotlib as mpl
import os

mpl.style.use("seaborn")

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "DejaVu Sans",
    "font.sans-serif": ["Helvetica"],
    "font.size": 16,
    "figure.figsize": (15, 10),
    "figure.autolayout": True,
    "legend.fontsize": 'x-large',
    'axes.labelsize': 'x-large',
    'axes.titlesize':'x-large',
    'xtick.labelsize':'x-large',
    'ytick.labelsize':'x-large'})

def save_fig(model_name, filename):
    fldr = "/home/hermankj/Documents/master_thesis_figures/results/"
    fldr = os.path.join(fldr, model_name)
    os.makedirs(fldr, exist_ok=True)

    save_path = os.path.join(fldr, filename)
    plt.savefig(save_path, bbox_inches="tight")
    


def plot_eef_pos(pos_filepath, pos_desired_filepath, time_filepath, model_name):
    pos = pd.read_csv(pos_filepath, header=None) 
    pos_des = pd.read_csv(pos_desired_filepath, header=None)
    time = pd.read_csv(time_filepath, header=None)

    labels = [r"$x$", r"$y$", r"$z$"]
    labels_des = [r"$x_{goal}$", r"$y_{goal}$", r"$z_{traj}$"]              # labels for desired values

    for i in range(len(labels)):
        plt.figure()
        plt.plot(time, pos[i], label=labels[i])
        plt.plot(time, pos_des[i], "--", label=labels_des[i])

        plt.legend()
        plt.xlabel(r"Completed episode (\%)")
        plt.title(labels[i] + "-position")

        pos_dir = labels[i].replace("$", "")

        save_fig(model_name, pos_dir + "_pos.eps")


def plot_eef_quat_diff(quat_diff_filepath, time_filepath, model_name):
    quat_diff = pd.read_csv(quat_diff_filepath, header=None)
    time = pd.read_csv(time_filepath, header=None)

    plt.figure()
    plt.plot(time, quat_diff)

    plt.xlabel(r"Completed episode (\%)")
    plt.title("Quaternion distance")

    save_fig(model_name, "quat_diff.eps")


def plot_eef_vel(vel_filepath, mean_vel_filepath, desired_vel_filepath, time_filepath, model_name):
    vel = pd.read_csv(vel_filepath, header=None)
    vel = vel.apply(np.linalg.norm, axis=1)
    mean_vel = pd.read_csv(mean_vel_filepath, header=None)
    des_vel = pd.read_csv(desired_vel_filepath, header=None)
    time = pd.read_csv(time_filepath, header=None)

    plt.figure()
    plt.plot(time, vel, label=r"$\left\Vert \mathbf{v} \right\Vert$")
    plt.plot(time, des_vel, "--", label=r"$v_{goal}$")
    plt.plot(time, mean_vel, label=r"$\bar{v}$")

    plt.legend()
    plt.xlabel(r"Completed episode (\%)")
    plt.title("End-effector velocity")

    save_fig(model_name, "vel.eps")


def plot_contact_force(
        force_filepath, 
        mean_force_filepath, 
        derivative_force_filepath, 
        desired_force_filepath,
        desired_derivative_filepath, 
        time_filepath,
        model_name):

    force = pd.read_csv(force_filepath, header=None)
    mean_force = pd.read_csv(mean_force_filepath, header=None)
    desired_force = pd.read_csv(desired_force_filepath, header=None)
    derivative_force = pd.read_csv(derivative_force_filepath, header=None)
    desired_derivative = pd.read_csv(desired_derivative_filepath, header=None)
    time = pd.read_csv(time_filepath, header=None) 

    plt.figure()
    plt.plot(time, force, label=r"$f$")
    plt.plot(time, desired_force, "--", label=r"$f_{goal}$")
    plt.plot(time, mean_force, label=r"$\bar{f}$")

    plt.legend()
    plt.xlabel(r"Completed episode (\%)")
    plt.title(r"Contact force $z$-direction")

    save_fig(model_name, "force.eps")

    plt.figure()
    plt.plot(time, derivative_force, label=r"$f'$")
    plt.plot(time, desired_derivative, "--", label=r"$f'_{goal}$")

    plt.legend()
    plt.xlabel(r"Completed episode (\%)")
    plt.title(r"Derivative of contact force $z$-direction")

    save_fig(model_name, "der_force.eps")


def plot_rewards(
    pos_reward_filepath, 
    ori_reward_filepath, 
    vel_reward_filepath, 
    force_reward_filepath,
    der_force_reward_filepath, 
    time_filepath,
    model_name):

    pos_reward = pd.read_csv(pos_reward_filepath, header=None)
    ori_reward = pd.read_csv(ori_reward_filepath, header=None)
    vel_reward = pd.read_csv(vel_reward_filepath, header=None)
    force_reward = pd.read_csv(force_reward_filepath, header=None)
    der_force_reward = pd.read_csv(der_force_reward_filepath, header=None)

    time = pd.read_csv(time_filepath, header=None)

    labels = ["position", "orientation", "force", "derivative force", "velocity"]
    colors = ['red', 'black', 'blue', 'green', "purple"]
    rewards = [pos_reward, ori_reward, force_reward, der_force_reward, vel_reward]

    fig, axs = plt.subplots(len(rewards))

    fig.suptitle("Rewards")
    for i in range(len(rewards)):
        if i < len(rewards) - 1:
            axs[i].xaxis.set_major_locator(plt.NullLocator())
        axs[i].plot(time, rewards[i], color=colors[i])
        axs[i].set_title(labels[i])

    plt.xlabel("Completed episode (\%)")

    save_fig(model_name, "rewards.eps")


def plot_qpos(qpos_filepath, time_filepath):
    qpos = pd.read_csv(qpos_filepath, header=None)
    time = pd.read_csv(time_filepath, header=None)

    colors = ['red', 'black', 'blue', 'brown', 'green', "purple", "olive"]

    fig, axs = plt.subplots(len(qpos.columns))
    fig.suptitle("Joint positions")
    for i in range(len(qpos.columns)):
        axs[i].plot(time, qpos[i], color=colors[i])
        axs[i].set_title("q" + str(i + 1))
    
    plt.xlabel("Completed episode (\%)")


def plot_qtorques(qtorque_filepath, time_filepath, title=None):
    torques = pd.read_csv(qtorque_filepath, header=None)
    time = pd.read_csv(time_filepath, header=None)

    colors = ['red', 'black', 'blue', 'brown', 'green', "purple", "olive"]

    fig, axs = plt.subplots(len(torques.columns))
    if title is None:
        fig.suptitle("Joint torques") 
    else:
        fig.suptitle(title)

    for i in range(len(torques.columns)):
        axs[i].plot(time, torques[i], color=colors[i])
        axs[i].set_title("q" + str(i + 1))
        axs[i].set(ylabel="N")
    
    plt.xlabel("Completed episode (\%)")


def plot_controller_gains(action_filepath, time_filepath, model_name, use_subfig=True):
    action = pd.read_csv(action_filepath, header=None)
    time = pd.read_csv(time_filepath, header=None)

    labels = ["$x$", "$y$", "$z$", "$ax$", "$ay$", "$az$"]
    colors = ['red', 'black', 'blue', 'brown', 'green', "purple"]

    # tracking mode
    if len(action.columns) == 6:
        kp = scale_input(action, 0, 500, 0, 1)
        kd = 2 * np.sqrt(kp)

    # variable_z mode
    elif len(action.columns) == 7:
        kp = scale_input(action.iloc[:, :-1], 0, 500, 0, 1)
        kd = 2 * np.sqrt(kp)

    else:
        print("Unknown action dim!")
        return

    # reset column indices
    kd.columns = [i for i in range(len(kd.columns))] 
    kp.columns = [i for i in range(len(kp.columns))]
    
    if use_subfig:
        fig1, axs1 = plt.subplots(len(kd.columns))
        fig1.suptitle("Controller $k_d$ gains")
        for i in range(len(kd.columns)):
            if i < len(kd.columns) - 1:
                axs1[i].xaxis.set_major_locator(plt.NullLocator())

            axs1[i].plot(time, kd[i], color=colors[i])
            axs1[i].set_title(labels[i])

        plt.xlabel("Completed episode (\%)")

        save_fig(model_name, "kd.eps")

        fig2, axs2 = plt.subplots(len(kp.columns))
        fig2.suptitle("Controller $k_p$ gains")
        for i in range(len(kp.columns)):
            if i < len(kp.columns) - 1:
                axs2[i].xaxis.set_major_locator(plt.NullLocator())
            
            axs2[i].plot(time, kp[i], color=colors[i])
            axs2[i].set_title(labels[i])

        plt.xlabel("Completed episode (\%)")
        
        save_fig(model_name, "kp.eps")

    else:
        for i in range(len(kd.columns)):
            plt.figure()
            plt.plot(time, kd[i])
            plt.title(labels[i])
            plt.xlabel("Completed episode (\%)")

            save_fig(model_name, "kd_" + labels[i] +".eps")

        for i in range(len(kp.columns)):
            plt.figure()
            plt.plot(time, kp[i])
            plt.title(labels[i])
            plt.xlabel("Completed episode (\%)")

            save_fig(model_name, "kp_" + labels[i] + ".eps")



def plot_wrench(action_filepath, time_filepath, model_name):
    action = pd.read_csv(action_filepath, header=None)      # [force_x, force_y, force_z, torque_x, torque_y, torque_z]
    time = pd.read_csv(time_filepath, header=None)

    labels = [r"$f_x$", r"$f_y$", r"$f_z$", r"$\tau_x$", r"$\tau_y$", r"$\tau_z$"]

    dim = int(len(action.columns) / 2)
    fig_f , ax_f = plt.subplots(dim)

    for i in range(dim):
        # plot force
        ax_f[i].plot(time, action.iloc[:, i])
        ax_f[i].set_title(labels[i])
        if i < dim - 1:
            ax_f[i].xaxis.set_major_locator(plt.NullLocator())
    
    fig_f.suptitle(r"Controller $f_{des}$")
    plt.xlabel(r"Completed episode (\%)")

    save_fig(model_name, "force_des.eps")

    fig_t , ax_t = plt.subplots(dim)
    for i in range(dim, len(action.columns)):
        # plot torque
        ax_t[i - dim].plot(time, action.iloc[:, i])
        ax_t[i - dim].set_title(labels[i])

        if i < len(action.columns) - 1:
            ax_t[i- dim].xaxis.set_major_locator(plt.NullLocator())
    
    fig_t.suptitle(r"Controller $\tau_{des}$")
    plt.xlabel(r"Completed episode (\%)")

    save_fig(model_name, "torque_des.eps")
    

def plot_delta_z(action_filepath, time_filepath, model_name):
    action = pd.read_csv(action_filepath, header=None)
    time = pd.read_csv(time_filepath, header=None)

    # Not variable_z mode
    if len(action.columns) != 7:
        return 

    delta_z = action.iloc[:, -1]
    delta_z.columns = 0             # reset column index
    delta_z = scale_input(delta_z, -0.05, 0.05, -1, 1)  

    plt.figure()
    plt.plot(time, delta_z)

    plt.xlabel(r"Completed episode (\%)")
    plt.title(r"$\Delta_z$")

    save_fig(model_name, "delta_z")


def hmfc_plot_ee_pos(pos_filepath, pos_desired_filepath, time_filepath):
    pos = pd.read_csv(pos_filepath, header=None) 
    pos_des = pd.read_csv(pos_desired_filepath, header=None)
    time = pd.read_csv(time_filepath, header=None)

    labels = ["x", "y"]
    labels_des = ["x_goal", "y_goal"]               # labels for desired values

    plt.figure()
    for i in range(len(pos.columns)):
        plt.plot(time, pos[i], label=labels[i])
        plt.plot(time, pos_des[i], "--", label=labels_des[i])

    plt.legend()
    plt.xlabel("Completed episode (%)")
    plt.title("End-effector position")


def plot_sim_data(identifier, model_name, show_figs):
    id = str(identifier)

    sim_data_fldr = "simulation_data"
    reward_data_fldr = "reward_data"
    policy_data_fldr = "policy_data"

    time_path = os.path.join(sim_data_fldr, "time_" + id + ".csv")

    ee_pos_path = os.path.join(sim_data_fldr, "ee_pos_" + id + ".csv")
    ee_goal_pos_path = os.path.join(sim_data_fldr, "ee_goal_pos_" + id + ".csv")

    ee_z_force_path = os.path.join(sim_data_fldr, "ee_z_contact_force_" + id + ".csv")
    ee_mean_z_force_path = os.path.join(sim_data_fldr, "ee_z_running_mean_contact_force_" + id + ".csv")
    ee_goal_z_force_path = os.path.join(sim_data_fldr, "ee_z_goal_contact_force_" + id + ".csv")

    ee_z_derivative_force_path = os.path.join(sim_data_fldr, "ee_z_derivative_contact_force_" + id + ".csv")
    ee_goal_derivative_z_force_path = os.path.join(sim_data_fldr, "ee_z_goal_derivative_contact_force_" + id + ".csv")

    ee_vel_path = os.path.join(sim_data_fldr, "ee_vel_" + id + ".csv")
    ee_mean_vel_path = os.path.join(sim_data_fldr, "ee_running_mean_vel_" + id + ".csv")
    ee_goal_vel_path = os.path.join(sim_data_fldr, "ee_goal_vel_" + id + ".csv")

    ee_diff_quat_path = os.path.join(sim_data_fldr, "ee_diff_quat_" + id + ".csv")

    pos_reward_path = os.path.join(reward_data_fldr, "pos_" + id + ".csv")
    ori_reward_path = os.path.join(reward_data_fldr, "ori_" + id + ".csv")
    force_reward_path = os.path.join(reward_data_fldr, "force_" + id + ".csv")
    der_reward_path = os.path.join(reward_data_fldr, "derivative_force_" + id + ".csv")
    vel_reward_path = os.path.join(reward_data_fldr, "vel_" + id + ".csv")

    action_path = os.path.join(policy_data_fldr, "action_" + id + ".csv")

    plot_eef_pos(
        ee_pos_path, 
        ee_goal_pos_path, 
        time_path, 
        model_name)
    plot_eef_quat_diff(
        ee_diff_quat_path,
        time_path,
        model_name)
    plot_eef_vel(
        ee_vel_path, 
        ee_mean_vel_path, 
        ee_goal_vel_path, 
        time_path,
        model_name)
    plot_contact_force(
        ee_z_force_path, 
        ee_mean_z_force_path, 
        ee_z_derivative_force_path, 
        ee_goal_z_force_path, 
        ee_goal_derivative_z_force_path, 
        time_path,
        model_name)
    plot_rewards(
        pos_reward_path, 
        ori_reward_path, 
        vel_reward_path, 
        force_reward_path,
        der_reward_path, 
        time_path,
        model_name
        )
    if model_name == "baseline":    
        plot_wrench(action_path, time_path, model_name)
    else:
        plot_controller_gains(action_path, time_path, model_name, False)
        plot_delta_z(action_path, time_path, model_name)

    #plot_qpos("simulation_data/q_pos_" + id + ".csv", "simulation_data/time_" + id + ".csv")
    #plot_qtorques("simulation_data/q_torques_" + id + ".csv", "simulation_data/time_" + id + ".csv")

    if show_figs:
        plt.show()


def plot_training_rew_mean(tracking_filepath, variable_z_filepath, wrench_filepath):
    tracking = pd.read_csv(tracking_filepath)
    variable_z = pd.read_csv(variable_z_filepath)
    wrench = pd.read_csv(wrench_filepath)

    plt.figure()
    plt.plot(wrench["Step"], wrench["Value"], label="Baseline")
    plt.plot(tracking["Step"], tracking["Value"], label="Variable impedance")
    plt.plot(variable_z["Step"], variable_z["Value"], label="Augmented variable impedance")

    plt.legend()
    plt.xlabel(r"Step")
    plt.ylabel(r"Episodic mean reward")
    plt.title(r"Training curves")

    plt.savefig("/home/hermankj/Documents/master_thesis_figures/results/training_curves.eps", bbox_inches="tight")


def plot_training_rew_mean_obs_space(tracking_full_obs_filepath, tracking_reduced_obs_filepath):
    full_obs = pd.read_csv(tracking_full_obs_filepath)
    reduced_obs = pd.read_csv(tracking_reduced_obs_filepath)

    plt.figure()
    plt.plot(full_obs["Step"], full_obs["Value"], label="Full observation space")
    plt.plot(reduced_obs["Step"], reduced_obs["Value"], label="Reduced observation space")

    plt.legend()
    plt.xlabel(r"Step")
    plt.ylabel(r"Episodic mean reward")
    plt.title(r"Variable impedance model")

    plt.savefig("/home/hermankj/Documents/master_thesis_figures/results/effect_obs_space.eps", bbox_inches="tight")
    

def hmfc_plot_z_force(force_filepath, mean, desired_force_filepath, time_filepath):
    force = pd.read_csv(force_filepath, header=None)
    des_force = pd.read_csv(desired_force_filepath, header=None)
    mean_force = pd.read_csv(mean, header=None)
    time = pd.read_csv(time_filepath, header=None) 

    plt.figure()
    plt.plot(time, force, label="force")
    plt.plot(time, des_force, "--", label="goal_force")
    #plt.plot(time, mean_force, label="mean_force")

    plt.legend()
    plt.xlabel("Completed episode (%)")
    plt.title("z force")
    plt.show()


def hmfc_plot_z_pos(pos_filepath, time_filepath):
    pos = pd.read_csv(pos_filepath, header=None) 
    time = pd.read_csv(time_filepath, header=None)

    plt.figure()
    plt.plot(time, pos, label="z_pos")

    plt.legend()
    plt.xlabel("Completed episode (%)")
    plt.title("Position in z-direction")
    plt.show()


def hmfc_plot_torques(desired_torque_filename, comp_torque_filename, ext_torque_filename, time_filepath):
    plot_qtorques(desired_torque_filename, time_filepath, title="Desired torques")
    plot_qtorques(comp_torque_filename, time_filepath, title="Compensation torques")
    plot_qtorques(ext_torque_filename, time_filepath, title="External torques")
    

def plot_hmfc_data(run_num):
    num = str(run_num)
    hmfc_plot_ee_pos("hmfc_test_data/ee_pos_" + num + ".csv", "hmfc_test_data/ee_goal_pos_" + num + ".csv", "hmfc_test_data/time_" + num + ".csv")
    hmfc_plot_z_force("hmfc_test_data/ee_force_" + num + ".csv",  "hmfc_test_data/ee_force_mean_" + num + ".csv", "hmfc_test_data/ee_goal_force_" + num + ".csv", "hmfc_test_data/time_" + num + ".csv")
    hmfc_plot_z_pos("hmfc_test_data/ee_z_pos_" + num + ".csv", "hmfc_test_data/time_" + num + ".csv")
    hmfc_plot_torques("hmfc_test_data/desired_torque_" + num + ".csv", "hmfc_test_data/compensation_torque_" + num + ".csv", "hmfc_test_data/external_torque_" + num + ".csv", "hmfc_test_data/time_" + num + ".csv")


def scale_input(input, output_min, output_max, input_min, input_max):
    input_scale = abs(output_max - output_min) / abs(input_max - input_min)
    output_transform = (output_max + output_min) / 2.0
    input_transform = (input_max + input_min) / 2.0
    scaled_kp = (input - input_transform) * input_scale + output_transform

    return scaled_kp
