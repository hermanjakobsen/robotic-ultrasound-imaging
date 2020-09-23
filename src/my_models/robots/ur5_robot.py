import numpy as np
from robosuite.models.robots.robot import Robot
from robosuite.utils.mjcf_utils import array_to_string


class UR5(Robot):
    "Universal Robots UR5 is an industrial robotic arm designed to simulate repetitive manual tasks weighing up to 5 kg."

    def __init__(self):
        super().__init__("my_models/assets/robots/ur5/robot.xml")

        self.bottom_offset = np.array([0, 0, -0.913])
        self.set_joint_damping()
        self._model_name = "UR5"

    def set_base_xpos(self, pos):
        """Places the robot on position @pos."""
        node = self.worldbody.find("./body[@name='link0']")
        node.set("pos", array_to_string(pos - self.bottom_offset))

    def set_joint_damping(self, damping=np.array((0.1, 0.1, 0.1, 0.1, 0.1, 0.01, 0.01))):
        """Set joint damping """
        body = self._base_body
        for i in range(len(self._link_body)):
            body = body.find("./body[@name='{}']".format(self._link_body[i]))
            joint = body.find("./joint[@name='{}']".format(self._joints[i]))
            joint.set("damping", array_to_string(np.array([damping[i]])))

    def set_joint_frictionloss(self, friction=np.array((0.1, 0.1, 0.1, 0.1, 0.1, 0.01, 0.01))):
        """Set joint friction loss (static friction)"""
        body = self._base_body
        for i in range(len(self._link_body)):
            body = body.find("./body[@name='{}']".format(self._link_body[i]))
            joint = body.find("./joint[@name='{}']".format(self._joints[i]))
            joint.set("frictionloss", array_to_string(np.array([friction[i]])))

    @property
    def dof(self):
        return 6

    @property
    def joints(self):
        return ["joint{}".format(x) for x in range(1, 7)]

    @property
    def init_qpos(self):
        return np.array([-np.pi / 10, -np.pi / 3, np.pi / 4, 0, 0, 0])

    @property
    def init_qvel(self):
        return np.array([0, 0, 0, 0, 0, 0])

    @property
    def contact_geoms(self):
        return ["link{}_collision".format(x) for x in range(1, 7)]

    @property
    def _base_body(self):
        node = self.worldbody.find("./body[@name='link0']")
        return node

    @property
    def _link_body(self):
        return ["link1", "link2", "link3", "link4", "link5", "link6"]

    @property
    def _joints(self):
        return ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]