"""
Defines a string based method of initializing grippers
"""
from robosuite.models.grippers.two_finger_gripper import TwoFingerGripper, LeftTwoFingerGripper
from robosuite.models.grippers.pr2_gripper import PR2Gripper
from robosuite.models.grippers.robotiq_gripper import RobotiqGripper
from robosuite.models.grippers.pushing_gripper import PushingGripper
from robosuite.models.grippers.robotiq_three_finger_gripper import RobotiqThreeFingerGripper

from my_models.grippers.panda_gripper import PandaGripper
from my_models.grippers.ultrasoundprobe_gripper import UltrasoundProbeGripper


def gripper_factory(name):
    """
    Genreator for grippers
    Creates a Gripper instance with the provided name.
    Args:
        name: the name of the gripper class
    Returns:
        gripper: Gripper instance
    Raises:
        XMLError: [description]
    """
    if name == "TwoFingerGripper":
        return TwoFingerGripper()
    if name == "LeftTwoFingerGripper":
        return LeftTwoFingerGripper()
    if name == "PR2Gripper":
        return PR2Gripper()
    if name == "RobotiqGripper":
        return RobotiqGripper()
    if name == "PushingGripper":
        return PushingGripper()
    if name == "RobotiqThreeFingerGripper":
        return RobotiqThreeFingerGripper()
    if name == "PandaGripper":
        return PandaGripper()
    if name == "UltrasoundProbe":
        return UltrasoundProbeGripper()
    raise ValueError("Unknown gripper name {}".format(name))