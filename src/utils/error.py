import numpy as np
import pandas as pd
import os

def save_data(data, model_name, save_filename):
    """
    Saves data to 'errors/@model_name' folder as @save_filename_mse.csv.
    Args:
        data (np.array): Data to be saved
        model_name (string): Name of the model
        save_filename (string): What to call the saved results
    """
    fldr = os.path.join("error_data", model_name)
    os.makedirs(fldr, exist_ok=True)
    save_path = os.path.join(fldr, save_filename + ".csv")
    pd.DataFrame(np.array(data)).to_csv(save_path, header=None, index=None)


def mse(actual, goal, model_name, save_filename):
    """
    Calculates the mean squared error between two arrays and saves the result to file.

    Args:
        actual (np.array): The actual measurements
        goal (np.array): The goal values 
        model_name (string): Name of the model
        save_filename (string): What to call the saved results
    """
    mse = np.square(np.subtract(actual, goal)).mean()
    save_data(np.array([mse]), model_name, save_filename)
    

def mse_ee_pos(ee_pos_filepath, goal_pos_filepath, model_name):
    """
    Calculates the mean squared error of end-effector positions respectively and saves results to file.

    Args:
        ee_pos_filepath (string): Filepath for ee pos measurements
        goal_pos_filepath (string): Filepath for ee goal position values 
        model_name (string): Name of model the measurements are extracted with.
    """
    ee_pos = pd.read_csv(ee_pos_filepath, header=None)
    goal_pos = pd.read_csv(goal_pos_filepath, header=None)

    x_pos = ee_pos.iloc[:, 0] 
    y_pos = ee_pos.iloc[:, 1]
    
    x_goal_pos = goal_pos.iloc[:, 0] 
    y_goal_pos = goal_pos.iloc[:, 1]

    mse(x_pos, x_goal_pos, model_name, "x_pos_mse")
    mse(y_pos, y_goal_pos, model_name, "y_pos_mse")
 

def mse_ee_force(ee_force_filepath, ee_mean_force_filepath, goal_force_filepath, model_name):
    """
    Calculates the mean squared error of end-effector contact force in z-direction and saves results to file.

    Args:
        ee_force_filepath (string): Filepath for ee force measurements
        ee_mean_force_filepath (string): Filepath for ee mean force measurements
        goal_force_filepath (string): Filepath for ee goal force values 
        model_name (string): Name of model the measurements are extracted with.
    """
    force = pd.read_csv(ee_force_filepath, header=None)
    mean_force = pd.read_csv(ee_mean_force_filepath, header=None)
    goal = pd.read_csv(goal_force_filepath, header=None)

    mse(force, goal, model_name, "force_mse")
    mse(mean_force, goal, model_name, "mean_force_mse")


def mse_ee_der_force(ee_der_force_filepath, goal_der_force_filepath, model_name):
    """
    Calculates the mean squared error of end-effector contact force in z-direction and saves results to file.

    Args:
        ee_der_force_filepath (string): Filepath for ee force measurements
        ee_mean_force_filepath (string): Filepath for ee mean force measurements
        goal_force_filepath (string): Filepath for ee goal force values 
        model_name (string): Name of model the measurements are extracted with.
    """
    der_force = pd.read_csv(ee_der_force_filepath, header=None)
    goal = pd.read_csv(goal_der_force_filepath, header=None)

    mse(der_force, goal, model_name, "der_force_mse")
    

def mse_ee_velocity(ee_vel_filepath, ee_mean_vel_filepath, goal_force_filepath, model_name):
    """
    Calculates the mean squared error of end-effector velocity and saves results to file.

    Args:
        ee_vel_filepath (string): Filepath for ee velocity measurements
        ee_mean_vel_filepath (string): Filepath for ee mean velocity measurements
        goal_force_filepath (string): Filepath for ee goal velocity values 
        model_name (string): Name of model the measurements are extracted with.
    """
    velocity = pd.read_csv(ee_vel_filepath, header=None)
    velocity = pd.DataFrame(velocity.apply(np.linalg.norm, axis=1))
    mean_vel = pd.read_csv(ee_mean_vel_filepath, header=None)
    goal = pd.read_csv(goal_force_filepath, header=None)

    mse(velocity, goal, model_name, "velocity_mse")
    mse(mean_vel, goal, model_name, "mean_velocity_mse")


def mean_rewards(
    pos_reward_filepath, 
    ori_reward_filepath,
    force_reward_filepath,
    der_force_reward_filepath,
    vel_reward_filepath,
    model_name):
    """
    Calculates the mean of the rewards and saves results to file.

    Args:
        *_reward_filepath (string): Filepath to * reward.
        model_name (string): Name of model the rewards are extracted with.
    """
    pos_reward_mean = pd.read_csv(pos_reward_filepath, header=None).mean()
    ori_reward_mean = pd.read_csv(ori_reward_filepath, header=None).mean()
    force_reward_mean = pd.read_csv(force_reward_filepath, header=None).mean()
    der_force_reward_mean = pd.read_csv(der_force_reward_filepath, header=None).mean()
    vel_reward_mean = pd.read_csv(vel_reward_filepath, header=None).mean()

    means = [pos_reward_mean, ori_reward_mean, force_reward_mean, der_force_reward_mean, vel_reward_mean]
    save_names = ["pos_reward_mean", "ori_reward_mean", "force_reward_mean", "der_force_reward_mean", "vel_reward_mean"]

    for i in range(len(means)):
        save_data(means[i], model_name, save_names[i])


def mean_ee_quat_diff(ee_quat_diff_filepath, model_name):
    """
    Calculates the mean of the end-effector quaternion difference between the end-effector quaternion and
    the goal quaternion. The results are saved to file.

    Args:
        ee_quat_diff_filepath (string): Filepath to end-effector quaternion difference.
        model_name (string): Name of model the measurements are extracted with.
    """
    quat_diff_mean = pd.read_csv(ee_quat_diff_filepath, header=None).mean()
    save_data(np.array([quat_diff_mean]), model_name, "quat_diff_mean")


def calculate_error_metrics(model_name):
    """
    Calculate relevant error metrics (i.e. mse and mean) for the given model. Saves results to files.

    Args:
        model_name (string): Name of model the measurements are extracted with.
    """
    sim_data_fldr = "simulation_data"
    reward_data_fldr = "reward_data"

    ee_pos_path = os.path.join(sim_data_fldr, "ee_pos_" + model_name + ".csv")
    ee_goal_pos_path = os.path.join(sim_data_fldr, "ee_goal_pos_" + model_name + ".csv")

    ee_z_force_path = os.path.join(sim_data_fldr, "ee_z_contact_force_" + model_name + ".csv")
    ee_mean_z_force_path = os.path.join(sim_data_fldr, "ee_z_running_mean_contact_force_" + model_name + ".csv")
    ee_goal_z_force_path = os.path.join(sim_data_fldr, "ee_z_goal_contact_force_" + model_name + ".csv")

    ee_z_derivative_force_path = os.path.join(sim_data_fldr, "ee_z_derivative_contact_force_" + model_name + ".csv")
    ee_goal_derivative_z_force_path = os.path.join(sim_data_fldr, "ee_z_goal_derivative_contact_force_" + model_name + ".csv")

    ee_vel_path = os.path.join(sim_data_fldr, "ee_vel_" + model_name + ".csv")
    ee_mean_vel_path = os.path.join(sim_data_fldr, "ee_running_mean_vel_" + model_name + ".csv")
    ee_goal_vel_path = os.path.join(sim_data_fldr, "ee_goal_vel_" + model_name + ".csv")

    ee_diff_quat_path = os.path.join(sim_data_fldr, "ee_diff_quat_" + model_name + ".csv")

    pos_reward_path = os.path.join(reward_data_fldr, "pos_" + model_name + ".csv")
    ori_reward_path = os.path.join(reward_data_fldr, "ori_" + model_name + ".csv")
    force_reward_path = os.path.join(reward_data_fldr, "force_" + model_name + ".csv")
    der_reward_path = os.path.join(reward_data_fldr, "derivative_force_" + model_name + ".csv")
    vel_reward_path = os.path.join(reward_data_fldr, "vel_" + model_name + ".csv")

    mse_ee_pos(ee_pos_path, ee_goal_pos_path, model_name)
    mse_ee_force(ee_z_force_path, ee_mean_z_force_path, ee_goal_z_force_path, model_name)
    mse_ee_der_force(ee_z_derivative_force_path, ee_goal_derivative_z_force_path, model_name)
    mse_ee_velocity(ee_vel_path, ee_mean_vel_path, ee_goal_vel_path, model_name)
    mean_ee_quat_diff(ee_diff_quat_path, model_name)
    mean_rewards(
        pos_reward_path,
        ori_reward_path,
        force_reward_path,
        der_reward_path,
        vel_reward_path,
        model_name)