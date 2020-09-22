"""
Intended for attaching an ultrasound probe.
"""
import numpy as np
from robosuite.utils.mjcf_utils import xml_path_completion
from robosuite.models.grippers.gripper import Gripper


class UltrasoundProbeGripper(Gripper):
    """
    Intended for attaching an ultrasound probe.
    """

    def __init__(self):
        super().__init__("my_models/assets/grippers/ultrasound_probe.xml")

    def format_action(self, action):
        return action

    @property
    def init_qpos(self):
        return np.array([])

    @property
    def joints(self):
        return []

    @property
    def dof(self):
        return 0

    @property
    def visualization_sites(self):
        return ["grip_site", "grip_site_cylinder"]

    def contact_geoms(self):
        return [
            "collision"
        ]
