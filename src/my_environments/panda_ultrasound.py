from collections import OrderedDict
import numpy as np

from robosuite.utils.transform_utils import convert_quat

from my_environments import PandaEnv
from my_models.robots import Panda
from my_models.objects import TorsoObject
from my_models.tasks import UltrasoundTask
from my_models.arenas import UltrasoundArena

class PandaUltrasound(PandaEnv):
    """
    This class corresponds to the ultrasound task for the Panda robot arm.
    """

    def __init__(
        self,
        gripper_type="UltrasoundProbe",
        table_full_size=(0.8, 0.8, 0.8),
        table_friction=(1., 5e-3, 1e-4),
        use_camera_obs=True,
        use_object_obs=True,
        reward_shaping=False,
        placement_initializer=None,
        gripper_visualization=False,
        use_indicator_object=False,
        has_renderer=False,
        has_offscreen_renderer=True,
        render_collision_mesh=False,
        render_visual_mesh=True,
        control_freq=10,
        horizon=1000,
        ignore_done=False,
        camera_name="frontview",
        camera_height=256,
        camera_width=256,
        camera_depth=False,
    ):
        """
        Args:
            gripper_type (str): type of gripper, used to instantiate
                gripper models from gripper factory.
            table_full_size (3-tuple): x, y, and z dimensions of the table.
            table_friction (3-tuple): the three mujoco friction parameters for
                the table.
            use_camera_obs (bool): if True, every observation includes a
                rendered image.
            use_object_obs (bool): if True, include object (cube) information in
                the observation.
            reward_shaping (bool): if True, use dense rewards.
            placement_initializer (ObjectPositionSampler instance): if provided, will
                be used to place objects on every reset, else a UniformRandomSampler
                is used by default.
            gripper_visualization (bool): True if using gripper visualization.
                Useful for teleoperation.
            use_indicator_object (bool): if True, sets up an indicator object that
                is useful for debugging.
            has_renderer (bool): If true, render the simulation state in
                a viewer instead of headless mode.
            has_offscreen_renderer (bool): True if using off-screen rendering.
            render_collision_mesh (bool): True if rendering collision meshes
                in camera. False otherwise.
            render_visual_mesh (bool): True if rendering visual meshes
                in camera. False otherwise.
            control_freq (float): how many control signals to receive
                in every second. This sets the amount of simulation time
                that passes between every action input.
            horizon (int): Every episode lasts for exactly @horizon timesteps.
            ignore_done (bool): True if never terminating the environment (ignore @horizon).
            camera_name (str): name of camera to be rendered. Must be
                set if @use_camera_obs is True.
            camera_height (int): height of camera frame.
            camera_width (int): width of camera frame.
            camera_depth (bool): True if rendering RGB-D, and RGB otherwise.
        """

        # settings for table top
        self.table_full_size = table_full_size
        self.table_friction = table_friction

        # whether to use ground-truth object states
        self.use_object_obs = use_object_obs

        # reward configuration
        self.reward_shaping = reward_shaping

        super().__init__(
            gripper_type=gripper_type,
            gripper_visualization=gripper_visualization,
            use_indicator_object=use_indicator_object,
            has_renderer=has_renderer,
            has_offscreen_renderer=has_offscreen_renderer,
            render_collision_mesh=render_collision_mesh,
            render_visual_mesh=render_visual_mesh,
            control_freq=control_freq,
            horizon=horizon,
            ignore_done=ignore_done,
            use_camera_obs=use_camera_obs,
            camera_name=camera_name,
            camera_height=camera_height,
            camera_width=camera_width,
            camera_depth=camera_depth,
        )

    def _load_model(self):
        """
        Loads an xml model, puts it in self.model
        """
        super()._load_model()
        self.mujoco_robot.set_base_xpos([0, 0, 0])

        # load model for table top workspace
        self.mujoco_arena = UltrasoundArena(
            table_full_size=self.table_full_size, table_friction=self.table_friction
        )
        if self.use_indicator_object:
            self.mujoco_arena.add_pos_indicator()

        # The panda robot has a pedestal, we want to align it with the table
        self.mujoco_arena.set_origin([0.16 + self.table_full_size[0] / 2, 0, 0])

        # initialize objects of interest
        torso = TorsoObject()

        self.mujoco_objects = OrderedDict([("human_torso", torso)])

        # task includes arena, robot, and objects of interest
        self.model = UltrasoundTask(
            self.mujoco_arena,
            self.mujoco_robot,
            self.mujoco_objects,
        )

    def _check_contact(self):
        """
        Returns True if gripper is in contact with an object.
        """
        collision = False
        for contact in self.sim.data.contact[: self.sim.data.ncon]:
            if (
                self.sim.model.geom_id2name(contact.geom1)
                in self.gripper.contact_geoms()
                or self.sim.model.geom_id2name(contact.geom2)
                in self.gripper.contact_geoms()
            ):
                collision = True
                break
        return collision
