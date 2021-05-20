import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import matplotlib as mpl

#mpl.style.use("seaborn")

def plot_eef_pos(pos_filename, pos_desired_filename, time_filename):
    pos = pd.read_csv(pos_filename, header=None) 
    pos_des = pd.read_csv(pos_desired_filename, header=None)
    time = pd.read_csv(time_filename, header=None)

    labels = ["x", "y"]
    labels_des = ["x_goal", "y_goal"]              # labels for desired values

    plt.figure()
    plt.plot(time, pos[0], label=labels[0])
    plt.plot(time, pos_des[0], "--", label=labels_des[0])

    plt.legend()
    plt.xlabel("Completed episode (%)")
    plt.title("End-effector position")

    plt.figure()
    plt.plot(time, pos[1], label=labels[1])
    plt.plot(time, pos_des[1], "--", label=labels_des[1])

    plt.legend()
    plt.xlabel("Completed episode (%)")
    plt.title("End-effector position")

    plt.show()


def plot_eef_vel(vel_filename, mean_vel_filename, desired_vel_filename, time_filename):
    vel = pd.read_csv(vel_filename, header=None)
    vel = vel.apply(np.linalg.norm, axis=1)
    mean_vel = pd.read_csv(mean_vel_filename, header=None)
    des_vel = pd.read_csv(desired_vel_filename, header=None)
    time = pd.read_csv(time_filename, header=None)

    plt.figure()
    plt.plot(time, vel, label="vel")
    plt.plot(time, mean_vel, label="mean_vel")
    plt.plot(time, des_vel, "--", label="goal_vel")

    plt.legend()
    plt.xlabel("Completed episode (%)")
    plt.title("End-effector velocity")

    plt.show()


def plot_contact_and_contact_force(
        force_filename, mean_force_filename, 
        derivative_force_filename, 
        desired_force_filename, 
        time_filename):

    force = pd.read_csv(force_filename, header=None)
    mean_force = pd.read_csv(mean_force_filename, header=None)
    desired_force = pd.read_csv(desired_force_filename, header=None)
    derivative_force = pd.read_csv(derivative_force_filename, header=None)
    time = pd.read_csv(time_filename, header=None) 

    plt.figure()
    plt.plot(time, force, label="force")
    plt.plot(time, mean_force, label="mean_force")
    plt.plot(time, desired_force, "--", label="goal_force")

    plt.legend()
    plt.xlabel("Completed episode (%)")
    plt.title("Contact force z")

    plt.figure()
    plt.plot(time, derivative_force)

    plt.xlabel("Completed episode (%)")
    plt.title("Derivative contact force z")

    plt.show()


def plot_rewards(
    pos_reward_filename, 
    ori_reward_filename, 
    vel_reward_filename, 
    force_reward_filename,
    der_force_reward_filename, 
    time_filename):

    pos_reward = pd.read_csv(pos_reward_filename, header=None)
    ori_reward = pd.read_csv(ori_reward_filename, header=None)
    vel_reward = pd.read_csv(vel_reward_filename, header=None)
    force_reward = pd.read_csv(force_reward_filename, header=None)
    der_force_reward = pd.read_csv(der_force_reward_filename, header=None)

    time = pd.read_csv(time_filename, header=None)

    labels = ["pos", "ori", "vel", "force", "der_force"]
    colors = ['red', 'black', 'blue', 'green', "purple"]
    rewards = [pos_reward, ori_reward, vel_reward, force_reward, der_force_reward]

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
    
    plt.xlabel("Completed episode (%)")
    plt.show()


def plot_controller_delta(action_filename, time_filename):
    action = pd.read_csv(action_filename, header=None)
    time = pd.read_csv(time_filename, header=None)

    # tracking mode
    if len(action.columns) == 6:
        print("Joint commands are not part of the action space for this controller mode")
        return

    controller_delta = action.iloc[:, -6:]
    controller_delta.columns = [i for i in range(len(controller_delta.columns))]  # reset column indices

    labels = ["x", "y", "z", "ax", "ay", "az"]

    fig , axs = plt.subplots(len(controller_delta.columns))
    fig.suptitle("Joint commands")
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
        damping_ratio.columns = [i for i in range(len(damping_ratio.columns))]         # reset column indices

        kp = action.iloc[:, 6:12]
        kp.columns = [i for i in range(len(kp.columns))]         # reset column indices

        kd = 2 * np.sqrt(kp)
        kd = kd.multiply(damping_ratio)

        fig1, axs1 = plt.subplots(len(damping_ratio.columns))
        fig1.suptitle("Controller damping ratio")
        for i in range(len(damping_ratio.columns)):
            axs1[i].plot(time, damping_ratio[i], color=colors[i])
            axs1[i].set_title(labels[i])

        plt.xlabel("Completed episode (%)")

    # variable_kp mode
    elif len(action.columns) == 12:
        kp = action.iloc[:, :6]
        kd = 2 * np.sqrt(kp)

    # tracking mode
    elif len(action.columns) == 6:
        kp = scale_gain(action, 1, 500)
        kd = 2 * np.sqrt(kp)

    # variable_z mode
    elif len(action.columns) == 7:
        kp = scale_gain(action.iloc[:, :-1], 1, 500)
        kd = 2 * np.sqrt(kp)

    else:
        print("Unknown action dim!")
        return

    # reset column indices
    kd.columns = [i for i in range(len(kd.columns))] 

    fig2, axs2 = plt.subplots(len(kd.columns))
    fig2.suptitle("Controller Kd")
    for i in range(len(kd.columns)):
        axs2[i].plot(time, kd[i], color=colors[i])
        axs2[i].set_title(labels[i])

    plt.xlabel("Completed episode (%)")

    # reset column indices
    kp.columns = [i for i in range(len(kp.columns))] 

    fig3, axs3 = plt.subplots(len(kp.columns))
    fig3.suptitle("Controller Kp")
    for i in range(len(kp.columns)):
        axs3[i].plot(time, kp[i], color=colors[i])
        axs3[i].set_title(labels[i])

    plt.xlabel("Completed episode (%)")

    plt.show()


def plot_delta_z(action_filename, time_filename):
    action = pd.read_csv(action_filename, header=None)
    time = pd.read_csv(time_filename, header=None)

    # Not variable_z mode
    if len(action.columns) != 7:
        return 

    delta_z = action.iloc[:, -1]
    delta_z.columns = 0             # reset column index

    plt.figure()
    plt.plot(time, delta_z, label="delta_z")

    plt.legend()
    plt.xlabel("Completed episode (%)")
    plt.title("Action - delta_z")
    plt.show()


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

    plt.show()


def plot_training_rew_mean(rew_mean_filename):
    rew_mean = pd.read_csv(rew_mean_filename)

    res = sns.lineplot(x="Step", y="Value", data=rew_mean)
    plt.title("Training - Episodic mean reward")
    plt.show()


def plot_sim_data(run_num):
    num = str(run_num)
    plot_eef_pos(
        "simulation_data/ee_pos_" + num + ".csv", 
        "simulation_data/ee_goal_pos_" + num + ".csv", 
        "simulation_data/time_" + num + ".csv")
    plot_eef_vel("simulation_data/ee_vel_" + num + ".csv", 
        "simulation_data/ee_running_mean_vel_" + num + ".csv", 
        "simulation_data/ee_goal_vel_" + num + ".csv", 
        "simulation_data/time_" + num + ".csv")
    plot_contact_and_contact_force(
        "simulation_data/ee_z_contact_force_" + num + ".csv", 
        "simulation_data/ee_z_running_mean_contact_force_" + num + ".csv", 
        "simulation_data/ee_z_derivative_contact_force_" + num + ".csv", 
        "simulation_data/ee_z_desired_contact_force_" + num + ".csv", 
        "simulation_data/time_" + num + ".csv")
    plot_rewards(
        "reward_data/pos_" + num + ".csv", 
        "reward_data/ori_" + num + ".csv", 
        "reward_data/vel_" + num + ".csv", 
        "reward_data/force_" + num + ".csv",
        "reward_data/derivative_force_" + num + ".csv", 
        "simulation_data/time_" + num + ".csv")
    #plot_controller_delta("policy_data/action_" + num + ".csv", "simulation_data/time_" + num + ".csv")
    plot_controller_gains("policy_data/action_" + num + ".csv", "simulation_data/time_" + num + ".csv")
    plot_delta_z("policy_data/action_" + num + ".csv", "simulation_data/time_" + num + ".csv")
    #plot_qpos("simulation_data/q_pos_" + num + ".csv", "simulation_data/time_" + num + ".csv")
    #plot_qtorques("simulation_data/q_torques_" + num + ".csv", "simulation_data/time_" + num + ".csv")


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


def scale_gain(kp, kp_min, kp_max):
    kp_scale = abs(kp_max - kp_min) / abs(1 - 0)
    kp_output_transform = (kp_max + kp_min) / 2.0
    kp_input_transform = (1 + 0) / 2.0
    scaled_kp = (kp - kp_input_transform) * kp_scale + kp_output_transform

    return scaled_kp
