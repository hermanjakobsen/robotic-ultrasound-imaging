import numpy as np
import matplotlib.pyplot as plt
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


def plot_eef_pos(pos_filename, pos_desired_filename, time_filename, model_name):
    pos = pd.read_csv(pos_filename, header=None) 
    pos_des = pd.read_csv(pos_desired_filename, header=None)
    time = pd.read_csv(time_filename, header=None)

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
        plt.savefig("/home/hermankj/Documents/master_thesis_figures/results/" + model_name + "/" + pos_dir+ "_pos.eps", bbox_inches="tight")
        

def plot_eef_quat_diff(quat_diff_filename, time_filename, model_name):
    quat_diff = pd.read_csv(quat_diff_filename, header=None)
    time = pd.read_csv(time_filename, header=None)

    plt.figure()
    plt.plot(time, quat_diff)

    plt.xlabel(r"Completed episode (\%)")
    plt.title("Quaternion distance")

    plt.savefig("/home/hermankj/Documents/master_thesis_figures/results/" + model_name + "/quat_diff.eps", bbox_inches="tight")


def plot_eef_vel(vel_filename, mean_vel_filename, desired_vel_filename, time_filename, model_name):
    vel = pd.read_csv(vel_filename, header=None)
    vel = vel.apply(np.linalg.norm, axis=1)
    mean_vel = pd.read_csv(mean_vel_filename, header=None)
    des_vel = pd.read_csv(desired_vel_filename, header=None)
    time = pd.read_csv(time_filename, header=None)

    plt.figure()
    plt.plot(time, vel, label=r"$\left\Vert \mathbf{v} \right\Vert$")
    plt.plot(time, des_vel, "--", label=r"$v_{goal}$")
    plt.plot(time, mean_vel, label=r"$\bar{v}$")

    plt.legend()
    plt.xlabel(r"Completed episode (\%)")
    plt.title("End-effector velocity")

    plt.savefig("/home/hermankj/Documents/master_thesis_figures/results/" + model_name + "/vel.eps", bbox_inches="tight")


def plot_contact_and_contact_force(
        force_filename, mean_force_filename, 
        derivative_force_filename, 
        desired_force_filename,
        desired_derivative_filename, 
        time_filename,
        model_name):

    force = pd.read_csv(force_filename, header=None)
    mean_force = pd.read_csv(mean_force_filename, header=None)
    desired_force = pd.read_csv(desired_force_filename, header=None)
    derivative_force = pd.read_csv(derivative_force_filename, header=None)
    desired_derivative = pd.read_csv(desired_derivative_filename, header=None)
    time = pd.read_csv(time_filename, header=None) 

    plt.figure()
    plt.plot(time, force, label=r"$f$")
    plt.plot(time, desired_force, "--", label=r"$f_{goal}$")
    plt.plot(time, mean_force, label=r"$\bar{f}$")

    plt.legend()
    plt.xlabel(r"Completed episode (\%)")
    plt.title(r"Contact force $z$-direction")

    plt.savefig("/home/hermankj/Documents/master_thesis_figures/results/" + model_name + "/force.eps", bbox_inches="tight")

    plt.figure()
    plt.plot(time, derivative_force, label=r"$f'$")
    plt.plot(time, desired_derivative, "--", label=r"$f'_{goal}$")

    plt.legend()
    plt.xlabel(r"Completed episode (\%)")
    plt.title(r"Derivative of contact force $z$-direction")

    plt.savefig("/home/hermankj/Documents/master_thesis_figures/results/" + model_name + "/der_force.eps", bbox_inches="tight")



def plot_rewards(
    pos_reward_filename, 
    ori_reward_filename, 
    vel_reward_filename, 
    force_reward_filename,
    der_force_reward_filename, 
    time_filename,
    model_name):

    pos_reward = pd.read_csv(pos_reward_filename, header=None)
    ori_reward = pd.read_csv(ori_reward_filename, header=None)
    vel_reward = pd.read_csv(vel_reward_filename, header=None)
    force_reward = pd.read_csv(force_reward_filename, header=None)
    der_force_reward = pd.read_csv(der_force_reward_filename, header=None)

    time = pd.read_csv(time_filename, header=None)

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

    plt.savefig("/home/hermankj/Documents/master_thesis_figures/results/" + model_name + "/rewards.eps", bbox_inches="tight")


def plot_qpos(qpos_filename, time_filename):
    qpos = pd.read_csv(qpos_filename, header=None)
    time = pd.read_csv(time_filename, header=None)

    colors = ['red', 'black', 'blue', 'brown', 'green', "purple", "olive"]

    fig, axs = plt.subplots(len(qpos.columns))
    fig.suptitle("Joint positions")
    for i in range(len(qpos.columns)):
        axs[i].plot(time, qpos[i], color=colors[i])
        axs[i].set_title("q" + str(i + 1))
    
    plt.xlabel("Completed episode (\%)")


def plot_qtorques(qtorque_filename, time_filename, title=None):
    torques = pd.read_csv(qtorque_filename, header=None)
    time = pd.read_csv(time_filename, header=None)

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


def plot_controller_delta(action_filename, time_filename):
    action = pd.read_csv(action_filename, header=None)
    time = pd.read_csv(time_filename, header=None)

    # tracking mode
    if len(action.columns) == 6:
        print("Joint commands are not part of the action space for this controller mode")
        return

    controller_delta = action.iloc[:, -6:]
    controller_delta.columns = [i for i in range(len(controller_delta.columns))]  # reset column indices

    labels = ["$x$", "$y$", "$z$", "$ax$", "$ay$", "$az$"]

    fig , axs = plt.subplots(len(controller_delta.columns))
    fig.suptitle("Joint commands")
    for i in range(len(controller_delta.columns)):
        axs[i].plot(time, controller_delta[i])
        axs[i].set_title(labels[i])
    plt.xlabel("Completed episode (\%)")


def plot_controller_gains(action_filename, time_filename, model_name):
    action = pd.read_csv(action_filename, header=None)
    time = pd.read_csv(time_filename, header=None)

    labels = ["$x$", "$y$", "$z$", "$ax$", "$ay$", "$az$"]
    colors = ['red', 'black', 'blue', 'brown', 'green', "purple"]

    # variable mode
    if len(action.columns) == 18:
        damping_ratio = action.iloc[:, :6]
        damping_ratio.columns = [i for i in range(len(damping_ratio.columns))]         # reset column indices

        kp = action.iloc[:, 6:12]
        kp.columns = [i for i in range(len(kp.columns))]         # reset column indices

        kd = 2 * np.sqrt(kp)
        kd = kd.multiply(damping_ratio)

        fig1, axs1 = plt.subplots(len(damping_ratio.columns))
        fig1.suptitle("Controller damping ratio")
        for i in range(len(damping_ratio.columns)):
            if i < len(damping_ratio.columns) - 1:
                axs1[i].xaxis.set_major_locator(plt.NullLocator())

            axs1[i].plot(time, damping_ratio[i], color=colors[i])
            axs1[i].set_title(labels[i])

        plt.xlabel("Completed episode (\%)")

    # variable_kp mode
    elif len(action.columns) == 12:
        kp = action.iloc[:, :6]
        kd = 2 * np.sqrt(kp)

    # tracking mode
    elif len(action.columns) == 6:
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

    fig2, axs2 = plt.subplots(len(kd.columns))
    fig2.suptitle("Controller $k_d$ gains")
    for i in range(len(kd.columns)):
        if i < len(kd.columns) - 1:
            axs2[i].xaxis.set_major_locator(plt.NullLocator())

        axs2[i].plot(time, kd[i], color=colors[i])
        axs2[i].set_title(labels[i])

    plt.xlabel("Completed episode (\%)")
    plt.savefig("/home/hermankj/Documents/master_thesis_figures/results/" + model_name + "/kd.eps", bbox_inches="tight")


    # reset column indices
    kp.columns = [i for i in range(len(kp.columns))] 

    fig3, axs3 = plt.subplots(len(kp.columns))
    fig3.suptitle("Controller $k_p$ gains")
    for i in range(len(kp.columns)):
        if i < len(kp.columns) - 1:
            axs3[i].xaxis.set_major_locator(plt.NullLocator())
        
        axs3[i].plot(time, kp[i], color=colors[i])
        axs3[i].set_title(labels[i])

    plt.xlabel("Completed episode (\%)")
    plt.savefig("/home/hermankj/Documents/master_thesis_figures/results/" + model_name + "/kp.eps", bbox_inches="tight")


def plot_wrench(action_filename, time_filename, model_name):
    action = pd.read_csv(action_filename, header=None)      # [force_x, force_y, force_z, torque_x, torque_y, torque_z]
    time = pd.read_csv(time_filename, header=None)

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
    plt.savefig("/home/hermankj/Documents/master_thesis_figures/results/" + model_name + "/force_des.eps", bbox_inches="tight")

    fig_t , ax_t = plt.subplots(dim)
    for i in range(dim, len(action.columns)):
        # plot torque
        ax_t[i - dim].plot(time, action.iloc[:, i])
        ax_t[i - dim].set_title(labels[i])

        if i < len(action.columns) - 1:
            ax_t[i- dim].xaxis.set_major_locator(plt.NullLocator())
    
    fig_t.suptitle(r"Controller $\tau_{des}$")
    plt.xlabel(r"Completed episode (\%)")
    plt.savefig("/home/hermankj/Documents/master_thesis_figures/results/" + model_name + "/torque_des.eps", bbox_inches="tight")
    

def plot_delta_z(action_filename, time_filename, model_name):
    action = pd.read_csv(action_filename, header=None)
    time = pd.read_csv(time_filename, header=None)

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

    plt.savefig("/home/hermankj/Documents/master_thesis_figures/results/" + model_name + "/delta_z.eps", bbox_inches="tight")


def hmfc_plot_ee_pos(pos_filename, pos_desired_filename, time_filename):
    pos = pd.read_csv(pos_filename, header=None) 
    pos_des = pd.read_csv(pos_desired_filename, header=None)
    time = pd.read_csv(time_filename, header=None)

    labels = ["x", "y"]
    labels_des = ["x_goal", "y_goal"]               # labels for desired values

    plt.figure()
    for i in range(len(pos.columns)):
        plt.plot(time, pos[i], label=labels[i])
        plt.plot(time, pos_des[i], "--", label=labels_des[i])

    plt.legend()
    plt.xlabel("Completed episode (%)")
    plt.title("End-effector position")


def plot_training_rew_mean(rew_mean_filename):
    rew_mean = pd.read_csv(rew_mean_filename)

    res = sns.lineplot(x="Step", y="Value", data=rew_mean)
    plt.title("Training - Episodic mean reward")


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
    plot_contact_and_contact_force(
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
        plot_controller_gains(action_path, time_path, model_name)
        plot_delta_z(action_path, time_path, model_name)

    #plot_qpos("simulation_data/q_pos_" + id + ".csv", "simulation_data/time_" + id + ".csv")
    #plot_qtorques("simulation_data/q_torques_" + id + ".csv", "simulation_data/time_" + id + ".csv")

    if show_figs:
        plt.show()


def hmfc_plot_z_force(force_filename, mean, desired_force_filename, time_filename):
    force = pd.read_csv(force_filename, header=None)
    des_force = pd.read_csv(desired_force_filename, header=None)
    mean_force = pd.read_csv(mean, header=None)
    time = pd.read_csv(time_filename, header=None) 

    plt.figure()
    plt.plot(time, force, label="force")
    plt.plot(time, des_force, "--", label="goal_force")
    #plt.plot(time, mean_force, label="mean_force")

    plt.legend()
    plt.xlabel("Completed episode (%)")
    plt.title("z force")
    plt.show()


def hmfc_plot_z_pos(pos_filename, time_filename):
    pos = pd.read_csv(pos_filename, header=None) 
    time = pd.read_csv(time_filename, header=None)

    plt.figure()
    plt.plot(time, pos, label="z_pos")

    plt.legend()
    plt.xlabel("Completed episode (%)")
    plt.title("Position in z-direction")
    plt.show()


def hmfc_plot_torques(desired_torque_filename, comp_torque_filename, ext_torque_filename, time_filename):
    plot_qtorques(desired_torque_filename, time_filename, title="Desired torques")
    plot_qtorques(comp_torque_filename, time_filename, title="Compensation torques")
    plot_qtorques(ext_torque_filename, time_filename, title="External torques")
    

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


def plot_training_rew_mean(tracking_filepath, variable_z_filepath, wrench_filepath):
    tracking = pd.read_csv(tracking_filepath)
    variable_z = pd.read_csv(variable_z_filepath)
    wrench = pd.read_csv(wrench_filepath)

    tracking.drop(columns="Wall time", inplace=True)
    variable_z.drop(columns="Wall time", inplace=True)
    wrench.drop(columns="Wall time", inplace=True)

    plt.figure()
    plt.plot(wrench["Step"], wrench["Value"], label="Baseline")
    plt.plot(tracking["Step"], tracking["Value"], label="Variable impedance")
    plt.plot(variable_z["Step"], variable_z["Value"], label="Augmented variable impedance")

    plt.legend()
    plt.xlabel(r"Step")
    plt.ylabel(r"Episodic mean reward")
    plt.title(r"Training curves")

    plt.savefig("/home/hermankj/Documents/master_thesis_figures/results/training_curves.eps", bbox_inches="tight")
    