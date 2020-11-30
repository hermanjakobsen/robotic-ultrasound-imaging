from collections import OrderedDict
import numpy as np

from robosuite.utils.transform_utils import convert_quat
from robosuite.utils.mjcf_utils import CustomMaterial

from robosuite.environments.robot_env import RobotEnv
from robosuite.robots import SingleArm

from robosuite.models.arenas import TableArena
from robosuite.models.objects import BoxObject
from robosuite.models.tasks import ManipulationTask, UniformRandomSampler


class FetchPush(RobotEnv):
    """
    This class corresponds to the fetch/push task for a single robot arm. Many similarities with lifting task.
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
        use_object_obs (bool): if True, include object (cube) information in
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
        distance_threshold = 0.06,
        placement_initializer=None,
        use_indicator_object=False,
        has_renderer=False,
        has_offscreen_renderer=True,
        render_camera="frontview",
        render_collision_mesh=False,
        render_visual_mesh=True,
        control_freq=10,
        horizon=1000,
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
        self.distance_threshold = distance_threshold

        # whether to use ground-truth object states
        self.use_object_obs = use_object_obs

        # object placement initializer
        if placement_initializer:
            self.placement_initializer = placement_initializer
        else:
            self.placement_initializer = UniformRandomSampler(
                x_range=[-0.2, 0.2],
                y_range=[-0.2, 0.2],
                ensure_object_boundary_in_range=False,
                rotation=None,
                z_offset=0.01,
            )

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
            camera_depths=camera_depths,
        )

    def _distance(self, goal_a, goal_b):
        assert goal_a.shape == goal_b.shape
        return np.linalg.norm(goal_a - goal_b)

    def reward(self, action=None):
        """
        Reward function for the task.

        Args:
            action (np array): [NOT USED]
        Returns:
            float: reward value
        """
        reward = .0

        cube_pos = np.array(self.sim.data.body_xpos[self.cube_body_id])
        cube_to_goal_dist = self._distance(cube_pos, self.goal_pos)
        gripper_pos = self.sim.data.site_xpos[self.robots[0].eef_site_id]
        gripper_to_cube_dist = self._distance(gripper_pos, cube_pos)

        if self.reward_shaping:

            # Return large penalty if robot has moved significantly away from cube
            if self._has_moved_significantly_away_from_cube():
                return -100

            # Return large reward if cube has been moved to goal
            if self._check_success():
                return 100

            # Give penalty for touching table
            if self._is_gripper_touching_table():
                reward -= 1

            # Reward for moving closer to cube
            if gripper_to_cube_dist < self.previous_gripper_to_cube_dist:
                reward += 1
    
            # Penalty for moving away from cube
            if gripper_to_cube_dist > self.previous_gripper_to_cube_dist:
                reward -= 1
            
            # Reward for touching cube
            if self._is_gripper_touching_cube():
                reward += 1

            # Reward for pushing cube closer to goal
            if cube_to_goal_dist < self.previous_cube_to_goal_dist:
                reward += 2

            # Penalty for pushing the cube further away
            if cube_to_goal_dist > self.previous_cube_to_goal_dist:
                reward -= 2

            # Update measurements
            self.previous_gripper_to_cube_dist = gripper_to_cube_dist
            self.previous_cube_to_goal_dist = cube_to_goal_dist

            return reward
        else:
            return -float(not self._check_success())

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
        self.mujoco_arena = TableArena(
            table_full_size=self.table_full_size,
            table_friction=self.table_friction,
            table_offset=(0, 0, 0.8),
        )
        if self.use_indicator_object:
            self.mujoco_arena.add_pos_indicator()

        # Arena always gets set to zero origin
        self.mujoco_arena.set_origin([0, 0, 0])

        # initialize objects of interest
        tex_attrib = {
            "type": "cube",
        }
        mat_attrib = {
            "texrepeat": "1 1",
            "specular": "0.4",
            "shininess": "0.1",
        }
        redwood = CustomMaterial(
            texture="WoodRed",
            tex_name="redwood",
            mat_name="redwood_mat",
            tex_attrib=tex_attrib,
            mat_attrib=mat_attrib,
        )
        cube = BoxObject(
            name="cube",
            size_min=[0.020, 0.020, 0.020],  # [0.015, 0.015, 0.015],
            size_max=[0.022, 0.022, 0.022],  # [0.018, 0.018, 0.018])
            rgba=[1, 0, 0, 1],
            material=redwood,
        )

        goal_cube = BoxObject(
            name="goal_cube",
            size_min=[0.005, 0.005, 0.005],
            size_max=[0.005, 0.005, 0.005], 
            rgba=[0, 0, 1, 1],
        )

        self.mujoco_objects = OrderedDict([("cube", cube), ("goal_cube", goal_cube)])
        self.n_objects = len(self.mujoco_objects)

        # task includes arena, robot, and objects of interest
        self.model = ManipulationTask(
            mujoco_arena=self.mujoco_arena, 
            mujoco_robots=[robot.robot_model for robot in self.robots], 
            mujoco_objects=self.mujoco_objects, 
            visual_objects=None, 
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

        # Additional object references from this env
        self.cube_body_id = self.sim.model.body_name2id("cube")
        self.goal_cube_body_id = self.sim.model.body_name2id("goal_cube")

        if self.robots[0].gripper_type == 'UltrasoundProbeGripper': 
            self.probe_geom_id = [
                self.sim.model.geom_name2id(x) for x in self.robots[0].gripper.important_geoms["probe"]
            ]
        elif self.robots[0].has_gripper:
            self.l_finger_geom_ids = [
                self.sim.model.geom_name2id(x) for x in self.robots[0].gripper.important_geoms["left_finger"]
            ]
            self.r_finger_geom_ids = [
                self.sim.model.geom_name2id(x) for x in self.robots[0].gripper.important_geoms["right_finger"]
            ]
        self.cube_geom_id = self.sim.model.geom_name2id("cube")
        self.goal_cube_geom_id = self.sim.model.geom_name2id("goal_cube")
        self.table_geom_id = self.sim.model.geom_name2id("table_collision")

    def _reset_internal(self):
        """
        Resets simulation internal configurations.
        """
        super()._reset_internal()

        # Reset all object positions using initializer sampler if we're not directly loading from an xml
        if not self.deterministic_reset:

            # Sample from the placement initializer for all objects
            obj_pos, obj_quat = self.model.place_objects()

            # Loop through all objects and reset their positions
            for i, (obj_name, _) in enumerate(self.mujoco_objects.items()):
                self.sim.data.set_joint_qpos(obj_name + "_jnt0", np.concatenate([np.array(obj_pos[i]), np.array(obj_quat[i])]))

        self.initial_cube_pos = np.array(self.sim.data.body_xpos[self.cube_body_id])
        self.initial_gripper_pos = np.array(self.sim.data.site_xpos[self.robots[0].eef_site_id])
        self.goal_pos = np.array(self.sim.data.body_xpos[self.goal_cube_body_id])

        self.initial_cube_to_goal_dist = self._distance(self.initial_cube_pos, self.goal_pos)
        self.initial_gripper_to_cube_dist = self._distance(self.initial_cube_pos, self.initial_gripper_pos)

        self.previous_cube_to_goal_dist = self.initial_cube_to_goal_dist
        self.previous_gripper_to_cube_dist = self.initial_gripper_to_cube_dist
        

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

        # Get robot prefix
        pr = self.robots[0].robot_model.naming_prefix

        if self.robots[0].has_gripper:
            # Checking if the UltrasoundProbeGripper is used
            if self.robots[0].gripper.dof == 0:
                # Remove unused keys (no joints in gripper)
                di.pop('robot0_gripper_qpos', None)
                di.pop('robot0_gripper_qvel', None)

        # low-level object information
        if self.use_object_obs:

            # position and rotation of object
            cube_pos = np.array(self.sim.data.body_xpos[self.cube_body_id])
            di["cube_pos"] = cube_pos
            di["goal_pos"] = self.goal_pos

            cube_to_goal_dist = [self._distance(cube_pos, self.goal_pos)]

            gripper_pos = di[pr + "eef_pos"]
            di[pr + "gripper_to_cube_dist"] = [self._distance(gripper_pos, cube_pos)]

            # Used for GymWrapper observations (Robot state will also default be added e.g. eef position)
            di["object-state"] = np.concatenate(
                [cube_pos, self.goal_pos, cube_to_goal_dist, di[pr + "gripper_to_cube_dist"]]
            )

        return di

    def _check_success(self):
        """
        Check if cube has been pushed to goal.
        Returns:
            bool: True if cube has been pushed to goal
        """
        cube_pos = self.sim.data.body_xpos[self.cube_body_id]

        # Cube is close to the goal_position
        return self._distance(self.goal_pos, cube_pos) < self.distance_threshold

    def _has_moved_significantly_away_from_cube(self):
        """
        Check if the robot has moved away from cube between steps.
        Returns:
            bool: True if episode is terminated
        """
        cube_pos = np.array(self.sim.data.body_xpos[self.cube_body_id])
        gripper_pos = np.array(self.sim.data.site_xpos[self.robots[0].eef_site_id])
        gripper_to_cube_dist = self._distance(cube_pos, gripper_pos)

        return gripper_to_cube_dist > self.initial_gripper_to_cube_dist + 0.12


    def _check_terminated(self):
        """
        Check if the task has completed one way or another. The following conditions lead to termination:
            - Task completion (cube pushed to goal) Should this be added?
            - Robot moving away from cube
        Returns:
            bool: True if episode is terminated
        """
        return self._has_moved_significantly_away_from_cube() or self._check_success()

    def _is_gripper_touching_cube(self):
        """
        Check if the gripper is in contact with the cube
        Returns:
            bool: True if contact
        """
        for contact in self.sim.data.contact[: self.sim.data.ncon]:
            geom_name1 =  self.sim.model.geom_id2name(contact.geom1)
            geom_name2 = self.sim.model.geom_id2name(contact.geom2)

            if (
                geom_name1 in self.robots[0].gripper.contact_geoms and geom_name2 == self.sim.model.geom_id2name(self.cube_geom_id)
                or geom_name2 in self.robots[0].gripper.contact_geoms and geom_name1 == self.sim.model.geom_id2name(self.cube_geom_id)
            ):
                return True

        return False

    def _is_gripper_touching_table(self):
        """
        Check if the gripper is in contact with the tabletop
        Returns:
            bool: True if contact
        """
        for contact in self.sim.data.contact[: self.sim.data.ncon]:
            geom_name1 =  self.sim.model.geom_id2name(contact.geom1)
            geom_name2 = self.sim.model.geom_id2name(contact.geom2)

            if (
                geom_name1 in self.robots[0].gripper.contact_geoms and geom_name2 == self.sim.model.geom_id2name(self.table_geom_id)
                or geom_name2 in self.robots[0].gripper.contact_geoms and geom_name1 == self.sim.model.geom_id2name(self.table_geom_id)
            ):
                return True

        return False

    def _post_action(self, action):
        """
        In addition to super method, add additional info if requested
        Args:
            action (np.array): Action to execute within the environment
        Returns:
            3-tuple:
                - (float) reward from the environment
                - (bool) whether the current episode is completed or not
                - (dict) info about current env step
        """
        reward, done, info = super()._post_action(action)
        done = done or self._check_terminated()
        return reward, done, info

    def _visualization(self):
        """
        Do any needed visualization here. Overrides superclass implementations.
        """

        # color the gripper site appropriately based on distance to cube
        if self.robots[0].gripper_visualization:
            # get distance to cube
            cube_site_id = self.sim.model.site_name2id("cube")
            dist = np.sum(
                np.square(
                    self.sim.data.site_xpos[cube_site_id]
                    - self.sim.data.get_site_xpos(self.robots[0].gripper.visualization_sites["grip_site"])
                )
            )

            # set RGBA for the EEF site here
            max_dist = 0.1
            scaled = (1.0 - min(dist / max_dist, 1.)) ** 15
            rgba = np.zeros(4)
            rgba[0] = 1 - scaled
            rgba[1] = scaled
            rgba[3] = 0.5

            self.sim.model.site_rgba[self.robots[0].eef_site_id] = rgba

    def _check_robot_configuration(self, robots):
        """
        Sanity check to make sure the inputted robots and configuration is acceptable
        Args:
            robots (str or list of str): Robots to instantiate within this env
        """
        if type(robots) is list:
            assert len(robots) == 1, "Error: Only one robot should be inputted for this task!"