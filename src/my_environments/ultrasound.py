from collections import OrderedDict
import numpy as np

from robosuite.utils.transform_utils import convert_quat
from robosuite.utils.mjcf_utils import CustomMaterial

from robosuite.environments.robot_env import RobotEnv
from robosuite.robots import SingleArm

from robosuite.models.tasks import  UniformRandomSampler

from my_models.tasks import UltrasoundTask
from my_models.arenas import UltrasoundArena
from my_models.objects import SoftTorsoObject


class Ultrasound(RobotEnv):
    """
    This class corresponds to the lifting task for a single robot arm.
    Args:
        robots (str or list of str): Specification for specific robot arm(s) to be instantiated within this env
            (e.g: "Sawyer" would generate one arm; ["Panda", "Panda", "Sawyer"] would generate three robot arms)
            Note: Must be a single single-arm robot!
        controller_configs (str or list of dict): If set, contains relevant controller parameters for creating a
            custom controller. Else, uses the default controller for this specific task. Should either be single
            dict if same controller is to be used for all robots or else it should be a list of the same length as
            "robots" param
        gripper_types (str or list of str): type of gripper, used to instantiate
            gripper models from gripper factory. Default is "default", which is the default grippers(s) associated
            with the robot(s) the 'robots' specification. None removes the gripper, and any other (valid) model
            overrides the default gripper. Should either be single str if same gripper type is to be used for all
            robots or else it should be a list of the same length as "robots" param
        gripper_visualizations (bool or list of bool): True if using gripper visualization.
            Useful for teleoperation. Should either be single bool if gripper visualization is to be used for all
            robots or else it should be a list of the same length as "robots" param
        initialization_noise (dict or list of dict): Dict containing the initialization noise parameters.
            The expected keys and corresponding value types are specified below:
            :`'magnitude'`: The scale factor of uni-variate random noise applied to each of a robot's given initial
                joint positions. Setting this value to `None` or 0.0 results in no noise being applied.
                If "gaussian" type of noise is applied then this magnitude scales the standard deviation applied,
                If "uniform" type of noise is applied then this magnitude sets the bounds of the sampling range
            :`'type'`: Type of noise to apply. Can either specify "gaussian" or "uniform"
            Should either be single dict if same noise value is to be used for all robots or else it should be a
            list of the same length as "robots" param
            :Note: Specifying "default" will automatically use the default noise settings.
                Specifying None will automatically create the required dict with "magnitude" set to 0.0.
        table_full_size (3-tuple): x, y, and z dimensions of the table.
        table_friction (3-tuple): the three mujoco friction parameters for
            the table.
        use_camera_obs (bool): if True, every observation includes rendered image(s)
        use_object_obs (bool): if True, include object (torso) information in
            the observation.
        reward_scale (None or float): Scales the normalized reward function by the amount specified.
            If None, environment reward remains unnormalized
        reward_shaping (bool): if True, use dense rewards.
        placement_initializer (ObjectPositionSampler instance): if provided, will
            be used to place objects on every reset, else a UniformRandomSampler
            is used by default.
        use_indicator_object (bool): if True, sets up an indicator object that
            is useful for debugging.
        has_renderer (bool): If true, render the simulation state in
            a viewer instead of headless mode.
        has_offscreen_renderer (bool): True if using off-screen rendering
        render_camera (str): Name of camera to render if `has_renderer` is True. Setting this value to 'None'
            will result in the default angle being applied, which is useful as it can be dragged / panned by
            the user using the mouse
        render_collision_mesh (bool): True if rendering collision meshes in camera. False otherwise.
        render_visual_mesh (bool): True if rendering visual meshes in camera. False otherwise.
        control_freq (float): how many control signals to receive in every second. This sets the amount of
            simulation time that passes between every action input.
        horizon (int): Every episode lasts for exactly @horizon timesteps.
        ignore_done (bool): True if never terminating the environment (ignore @horizon).
        hard_reset (bool): If True, re-loads model, sim, and render object upon a reset call, else,
            only calls sim.reset and resets all robosuite-internal variables
        camera_names (str or list of str): name of camera to be rendered. Should either be single str if
            same name is to be used for all cameras' rendering or else it should be a list of cameras to render.
            :Note: At least one camera must be specified if @use_camera_obs is True.
            :Note: To render all robots' cameras of a certain type (e.g.: "robotview" or "eye_in_hand"), use the
                convention "all-{name}" (e.g.: "all-robotview") to automatically render all camera images from each
                robot's camera list).
        camera_heights (int or list of int): height of camera frame. Should either be single int if
            same height is to be used for all cameras' frames or else it should be a list of the same length as
            "camera names" param.
        camera_widths (int or list of int): width of camera frame. Should either be single int if
            same width is to be used for all cameras' frames or else it should be a list of the same length as
            "camera names" param.
        camera_depths (bool or list of bool): True if rendering RGB-D, and RGB otherwise. Should either be single
            bool if same depth setting is to be used for all cameras or else it should be a list of the same length as
            "camera names" param.
    Raises:
        AssertionError: [Invalid number of robots specified]
    """

    def __init__(
        self,
        robots,
        controller_configs=None,
        gripper_types="default",
        gripper_visualizations=False,
        initialization_noise="default",
        table_full_size=(0.8, 0.8, 0.05),
        table_friction=(1., 5e-3, 1e-4),
        use_camera_obs=True,
        use_object_obs=True,
        reward_scale=1.0,
        reward_shaping=False,
        placement_initializer=None,
        use_indicator_object=False,
        has_renderer=False,
        has_offscreen_renderer=True,
        render_camera="frontview",
        render_collision_mesh=False,
        render_visual_mesh=True,
        control_freq=20,
        horizon=5000,
        ignore_done=False,
        hard_reset=True,
        camera_names="agentview",
        camera_heights=256,
        camera_widths=256,
        camera_depths=False,
    ):
        # First, verify that only one robot is being inputted
        self._check_robot_configuration(robots)

        # settings for table top
        self.table_full_size = table_full_size
        self.table_friction = table_friction

        # reward configuration
        self.reward_scale = reward_scale
        self.reward_shaping = reward_shaping

        # whether to use ground-truth object states
        self.use_object_obs = use_object_obs

        # object placement initializer
        if placement_initializer:
            self.placement_initializer = placement_initializer
        else:
            self.placement_initializer = UniformRandomSampler()

        super().__init__(
            robots=robots,
            controller_configs=controller_configs,
            gripper_types=gripper_types,
            gripper_visualizations=gripper_visualizations,
            initialization_noise=initialization_noise,
            use_camera_obs=use_camera_obs,
            use_indicator_object=use_indicator_object,
            has_renderer=has_renderer,
            has_offscreen_renderer=has_offscreen_renderer,
            render_camera=render_camera,
            render_collision_mesh=render_collision_mesh,
            render_visual_mesh=render_visual_mesh,
            control_freq=control_freq,
            horizon=horizon,
            ignore_done=ignore_done,
            hard_reset=hard_reset,
            camera_names=camera_names,
            camera_heights=camera_heights,
            camera_widths=camera_widths,
            camera_depths=camera_depths
        )

    def reward(self, action=None):
        """
        Reward function for the task.
        Sparse un-normalized reward:
            - a discrete reward of 2.25 is provided if the cube is lifted
        Un-normalized summed components if using reward shaping:
            - Reaching: in [0, 1], to encourage the arm to reach the cube
            - Grasping: in {0, 0.25}, non-zero if arm is grasping the cube
            - Lifting: in {0, 1}, non-zero if arm has lifted the cube
        The sparse reward only consists of the lifting component.
        Note that the final reward is normalized and scaled by
        reward_scale / 2.25 as well so that the max score is equal to reward_scale
        Args:
            action (np array): [NOT USED]
        Returns:
            float: reward value
        """
        return 0


    def _load_model(self):
        """
        Loads an xml model, puts it in self.model
        """
        super()._load_model()

        # Verify the correct robot has been loaded
        assert isinstance(self.robots[0], SingleArm), \
            "Error: Expected one single-armed robot! Got {} type instead.".format(type(self.robots[0]))

        # Adjust base pose accordingly
        xpos = self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0])
        self.robots[0].robot_model.set_base_xpos(xpos)

        # load model for table top workspace
        self.mujoco_arena = UltrasoundArena(
            table_full_size=self.table_full_size,
            table_friction=self.table_friction,
            table_offset=(0, 0, 0.8),
        )
        if self.use_indicator_object:
            self.mujoco_arena.add_pos_indicator()

        # Arena always gets set to zero origin
        self.mujoco_arena.set_origin([0, 0, 0])

        # initialize soft torso object
        soft_torso = SoftTorsoObject()

        stiffness = np.array([0.001, 0.1, 1, 50, 100])
        damping = np.array([0.001, 0.1, 1, 50, 100])
        rand_damp_idx = int(np.random.choice(damping.shape[0], 1, replace=False))
        rand_stiff_idx = int(np.random.choice(stiffness.shape[0], 1, replace=False))

        soft_torso._set_damping(damping[rand_damp_idx])
        soft_torso._set_stiffness(stiffness[rand_stiff_idx])

        self.mujoco_objects_on_table = OrderedDict([])#('soft_torso', soft_torso)])
        self.other_mujoco_objects = OrderedDict([('soft_torso', soft_torso)])

        self.n_objects = len(self.mujoco_objects_on_table) + len(self.other_mujoco_objects)

        # task includes arena, robot, and objects of interest
        self.model = UltrasoundTask(
            mujoco_arena=self.mujoco_arena, 
            mujoco_robots=[robot.robot_model for robot in self.robots], 
            mujoco_objects_on_table=self.mujoco_objects_on_table,
            other_mujoco_objects=self.other_mujoco_objects,
            initializer=self.placement_initializer,
        )
        self.model.place_objects()

    def _get_reference(self):
        """
        Sets up references to important components. A reference is typically an
        index or a list of indices that point to the corresponding elements
        in a flatten array, which is how MuJoCo stores physical simulation data.
        """
        super()._get_reference()
        #self.torso_body_id = self.sim.model.body_name2id('B3_2_4') # approx middle body
        #self.torso_geom_id = self.sim.model.geom_name2id('G3_2_4')

    def _reset_internal(self):
        """
        Resets simulation internal configurations.
        """
        super()._reset_internal()

    def _check_gripper_contact(self):
        """
        Checks whether each gripper is in contact with an object.
        Returns:
            list of bool: True if the specific gripper is in contact with an object
        """
        collisions = [False] * self.num_robots
        for idx, robot in enumerate(self.robots):
            for contact in self.sim.data.contact[: self.sim.data.ncon]:
                # Single arm case
                if robot.robot_model.arm_type == "single":
                    if (
                        self.sim.model.geom_id2name(contact.geom1)
                        in robot.gripper.contact_geoms
                        or self.sim.model.geom_id2name(contact.geom2)
                        in robot.gripper.contact_geoms
                    ):
                        collisions[idx] = True
                        break
                # Bimanual case
                else:
                    for arm in robot.arms:
                        if (
                                self.sim.model.geom_id2name(contact.geom1)
                                in robot.gripper[arm].contact_geoms
                                or self.sim.model.geom_id2name(contact.geom2)
                                in robot.gripper[arm].contact_geoms
                        ):
                            collisions[idx] = True
                            break
        return collisions

    def _get_observation(self):
        """
        Returns an OrderedDict containing observations [(name_string, np.array), ...].
        Important keys:
            `'robot-state'`: contains robot-centric information.
            `'object-state'`: requires @self.use_object_obs to be True. Contains object-centric information.
            `'image'`: requires @self.use_camera_obs to be True. Contains a rendered frame from the simulation.
            `'depth'`: requires @self.use_camera_obs and @self.camera_depth to be True.
            Contains a rendered depth map from the simulation
        Returns:
            OrderedDict: Observations from the environment
        """
        di = super()._get_observation()

        robot = self.robots[0]

        if robot.has_gripper:
            
            # Checking if the UltrasoundProbeGripper is used
            if robot.gripper.dof == 0:
                # Remove unused keys (no joints in gripper)
                di.pop('robot0_gripper_qpos', None)
                di.pop('robot0_gripper_qvel', None)
                di['gripper_pos'] = np.array(self.sim.data.get_body_xpos('gripper0_gripper_base'))
                di['gripper_velp'] = np.array(self.sim.data.get_body_xvelp('gripper0_gripper_base'))
                di['gripper_quat'] = convert_quat(self.sim.data.get_body_xquat('gripper0_gripper_base'), to="xyzw")

            di['contact'] = self._check_gripper_contact()

        return di


    def _check_robot_configuration(self, robots):
        """
        Sanity check to make sure the inputted robots and configuration is acceptable
        Args:
            robots (str or list of str): Robots to instantiate within this env
        """
        if type(robots) is list:
            assert len(robots) == 1, "Error: Only one robot should be inputted for this task!"