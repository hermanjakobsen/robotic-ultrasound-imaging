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
    fldr = os.path.join("errors", model_name)
    os.makedirs(fldr, exist_ok=True)
    save_path = os.path.join(fldr, save_filename + "_mse" + ".csv")
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

    mse(x_pos, x_goal_pos, model_name, "x_pos")
    mse(y_pos, y_goal_pos, model_name, "y_pos")
 

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

    mse(force, goal, model_name, "force")
    mse(mean_force, goal, model_name, "mean_force")


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

    mse(der_force, goal, model_name, "der_force")
    

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

    mse(velocity, goal, model_name, "velocity")
    mse(mean_vel, goal, model_name, "mean_velocity")


def mean_ee_quat_diff(ee_quat_diff_filepath, model_name):
    """
    Calculates the mean of the end-effector quaternion difference between the end-effector quaternion and
    the goal quaternion. The results are saved to file.

    Args:
        ee_quat_diff_filepath (string): Filepath to end-effector quaternion difference.
        model_name (string): Name of model the measurements are extracted with.
    """
    quat_diff = pd.read_csv(ee_quat_diff_filepath, header=None)
    mean = quat_diff.mean()
    save_data(np.array([mean]), model_name, "quat_diff_mean")

