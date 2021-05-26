import numpy as np
from mujoco_py import MjSimState, MjViewer, MjSim

from robosuite.models.grippers import GRIPPER_MAPPING


def register_gripper(gripper_class):
    """
    Register @gripper_class in GRIPPER_MAPPING.

    Args:
        gripper_class (GripperModel): Gripper class which should be registered
    """
    GRIPPER_MAPPING[gripper_class.__name__] = gripper_class


def get_number_of_elements_in_obs(obs):
    """
    Counts the number of elements in an environment's observation space

    Args:
        obs (dict): Observation space

    """
    num_el = 0
    for key in obs:
        num_el += obs[key].size
    return int(num_el / 2)               # every observation is added twice (default robosuite)