from collections import OrderedDict
import numpy as np
import re

from robosuite.utils.transform_utils import convert_quat
from robosuite.utils.mjcf_utils import CustomMaterial

from robosuite.environments.manipulation.single_arm_env import SingleArmEnv

from robosuite.models.tasks import ManipulationTask
from robosuite.utils.placement_samplers import UniformRandomSampler
from robosuite.utils.observables import Observable, sensor

from my_models.objects import SoftTorsoObject, BoxObject
from my_models.tasks import UltrasoundTask
from my_models.arenas import UltrasoundArena
from utils.quaternion import distance_quat


class Ultrasound(SingleArmEnv):
    """
    This class corresponds to the ultrasound task for a single robot arm.

    Args:
        robots (str or list of str): Specification for specific robot arm(s) to be instantiated within this env
            (e.g: "Sawyer" would generate one arm; ["Panda", "Panda", "Sawyer"] would generate three robot arms)
            Note: Must be a single single-arm robot!
        env_configuration (str): Specifies how to position the robots within the environment (default is "default").
            For most single arm environments, this argument has no impact on the robot setup.
        controller_configs (str or list of dict): If set, contains relevant controller parameters for creating a
            custom controller. Else, uses the default controller for this specific task. Should either be single
            dict if same controller is to be used for all robots or else it should be a list of the same length as
            "robots" param
        gripper_types (str or list of str): type of gripper, used to instantiate
            gripper models from gripper factory. Default is "default", which is the default grippers(s) associated
            with the robot(s) the 'robots' specification. None removes the gripper, and any other (valid) model
            overrides the default gripper. Should either be single str if same gripper type is to be used for all
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
        placement_initializer (ObjectPositionSampler): if provided, will
            be used to place objects on every reset, else a UniformRandomSampler
            is used by default.
        has_renderer (bool): If true, render the simulation state in
            a viewer instead of headless mode.
        has_offscreen_renderer (bool): True if using off-screen rendering
        render_camera (str): Name of camera to render if `has_renderer` is True. Setting this value to 'None'
            will result in the default angle being applied, which is useful as it can be dragged / panned by
            the user using the mouse
        render_collision_mesh (bool): True if rendering collision meshes in camera. False otherwise.
        render_visual_mesh (bool): True if rendering visual meshes in camera. False otherwise.
        render_gpu_device_id (int): corresponds to the GPU device id to use for offscreen rendering.
            Defaults to -1, in which case the device will be inferred from environment variables
            (GPUS or CUDA_VISIBLE_DEVICES).
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
        env_configuration="default",
        controller_configs=None,
        gripper_types="UltrasoundProbeGripper",
        initialization_noise="default",
        table_full_size=(0.8, 0.8, 0.05),
        table_friction=(1., 5e-3, 1e-4),
        use_camera_obs=True,
        use_object_obs=True,
        reward_scale=1.0,
        reward_shaping=False,
        placement_initializer=None,
        has_renderer=False,
        has_offscreen_renderer=True,
        render_camera="frontview",
        render_collision_mesh=False,
        render_visual_mesh=True,
        render_gpu_device_id=-1,
        control_freq=20,
        horizon=1000,
        ignore_done=False,
        hard_reset=True,
        camera_names="agentview",
        camera_heights=256,
        camera_widths=256,
        camera_depths=False,
    ):
        assert gripper_types =="UltrasoundProbeGripper",\
            "Tried to specify gripper other than UltrasoundProbeGripper in Ultrasound environment!"

        # settings for table top
        self.table_full_size = table_full_size
        self.table_friction = table_friction
        self.table_offset = np.array((0, 0, 0.8))

        # reward configuration
        self.reward_scale = reward_scale
        self.reward_shaping = reward_shaping
        self.contact_force_upper_threshold = 60
        self.contact_force_lower_threshold = 40
        self.timer_threshold = 60                   # how many steps probe must be in contact with torso to yield success

        # whether to use ground-truth object states
        self.use_object_obs = use_object_obs

        # object placement initializer
        self.placement_initializer = placement_initializer

        super().__init__(
            robots=robots,
            env_configuration=env_configuration,
            controller_configs=controller_configs,
            mount_types="default",
            gripper_types=gripper_types,
            initialization_noise=initialization_noise,
            use_camera_obs=use_camera_obs,
            has_renderer=has_renderer,
            has_offscreen_renderer=has_offscreen_renderer,
            render_camera=render_camera,
            render_collision_mesh=render_collision_mesh,
            render_visual_mesh=render_visual_mesh,
            render_gpu_device_id=render_gpu_device_id,
            control_freq=control_freq,
            horizon=horizon,
            ignore_done=ignore_done,
            hard_reset=hard_reset,
            camera_names=camera_names,
            camera_heights=camera_heights,
            camera_widths=camera_widths,
            camera_depths=camera_depths,
        )


    def reward(self, action=None):
        """
        Reward function for the task.

        Args:
            action (np array): [NOT USED]

        Returns:
            float: reward value
        """

        reward = 0. 

        total_force_ee = np.linalg.norm(np.array(self.robots[0].recent_ee_forcetorques.current[:3]))
        dist_to_torso_center = np.linalg.norm(self._eef_xpos - self._torso_xpos)
        ee_orientation = convert_quat(self._eef_xquat, to="wxyz")   # (w, x, y, z) quaternion


        # reward for probe touching torso
        if self._check_probe_contact_with_upper_part_torso():
            reward += 2.25

        # reaching reward
        reward += 1.5 * (1 - np.tanh(5 * dist_to_torso_center))

        # probe orientation penalty
        ori_deviation = distance_quat(self.ee_inital_orientation, ee_orientation)
        reward -= np.tanh(ori_deviation)

        # touching table penalty (will also end the episode)
        if self._check_probe_contact_with_table():
            return -100

        return reward


    def _load_model(self):
        """
        Loads an xml model, puts it in self.model
        """
        super()._load_model()

        # Adjust base pose accordingly
        xpos = self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0])
        self.robots[0].robot_model.set_base_xpos(xpos)

        # load model for table top workspace
        mujoco_arena = UltrasoundArena()

        # Arena always gets set to zero origin
        mujoco_arena.set_origin([0, 0, 0])

        # initialize objects of interest
        self.torso = SoftTorsoObject(name="torso")

        # Create placement initializer
        if self.placement_initializer is not None:
            self.placement_initializer.reset()
            self.placement_initializer.add_objects(self.torso)
        else:
            self.placement_initializer = UniformRandomSampler(
                name="ObjectSampler",
                mujoco_objects=[self.torso],
                x_range=[-0.12, 0.12],
                y_range=[-0.12, 0.12],
                rotation=None,
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=True,
                reference_pos=self.table_offset,
                z_offset=0.005,
            )

        # task includes arena, robot, and objects of interest
        self.model = UltrasoundTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots], 
            mujoco_objects=[self.torso]
        )


    def _setup_references(self):
        """
        Sets up references to important components. A reference is typically an
        index or a list of indices that point to the corresponding elements
        in a flatten array, which is how MuJoCo stores physical simulation data.
        """
        super()._setup_references()

        # additional object references from this env
        self.torso_body_id = self.sim.model.body_name2id(self.torso.root_body)


    def _setup_observables(self):
        """
        Sets up observables to be used for this environment. Creates object-based observables if enabled

        Returns:
            OrderedDict: Dictionary mapping observable names to its corresponding Observable object
        """
        observables = super()._setup_observables()

        # probe information
        modality = "probe"

        @sensor(modality=modality)
        def probe_contact_with_upper_part_torso(obs_cache):
            return self._check_probe_contact_with_upper_part_torso()

        @sensor(modality=modality)
        def probe_force(obs_cache):
            return self.robots[0].ee_force

        @sensor(modality=modality)
        def probe_torque(obs_cache):
            return self.robots[0].ee_torque

        sensors = [probe_contact_with_upper_part_torso, probe_force, probe_torque]
        
        # low-level object information
        if self.use_object_obs:
            modality = "object"

            @sensor(modality=modality)
            def torso_pos(obs_cache):
                return self._torso_xpos

            @sensor(modality=modality)
            def torso_quat(obs_cache):
                return convert_quat(np.array(self.sim.data.body_xquat[self.torso_body_id]), to="xyzw")

            sensors += [torso_pos, torso_quat]

        names = [s.__name__ for s in sensors]

        # Create observables
        for name, s in zip(names, sensors):
            observables[name] = Observable(
                name=name,
                sensor=s,
                sampling_rate=self.control_freq,
            )

        return observables


    def _reset_internal(self):
        """
        Resets simulation internal configurations.
        """
        super()._reset_internal()

        # Reset all object positions using initializer sampler if we're not directly loading from an xml
        if not self.deterministic_reset:

            # Sample from the placement initializer for all objects
            object_placements = self.placement_initializer.sample()

            # Loop through all objects and reset their positions
            for obj_pos, _, obj in object_placements.values():
                self.sim.data.set_joint_qpos(obj.joints[0], np.concatenate([np.array(obj_pos), np.array([0.5, 0.5, -0.5, -0.5])]))
        
        # ee resets - bias at initial state
        self.ee_force_bias = np.zeros(3)
        self.ee_torque_bias = np.zeros(3)

        # probe resets - orientation at initial state
        self.ee_inital_orientation = convert_quat(self._eef_xquat, to="wxyz")    # (w, x, y, z) quaternion

        # initialize timer
        self.timer = 0


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

        # Update force bias
        if np.linalg.norm(self.ee_force_bias) == 0:
            self.ee_force_bias = self.robots[0].ee_force
            self.ee_torque_bias = self.robots[0].ee_torque

        done = done or self._check_terminated()

        return reward, done, info


    def visualize(self, vis_settings):
        """
        Args:
            vis_settings (dict): Visualization keywords mapped to T/F, determining whether that specific
                component should be visualized. Should have "grippers" keyword as well as any other relevant
                options specified.
        """
        # Run superclass method first
        super().visualize(vis_settings=vis_settings)


    def _check_success(self):
        """
        Check if the probe is in contact with the upper/top part of torso for a given amount of time.

        Returns:
            bool: True if probe touched upper part of torso for a given amount of time. 
        """ 
        if self._check_probe_contact_with_torso():
            self.timer += 1
            return self.timer >= self.timer_threshold
            
        self.timer = 0
        return False


    def _check_probe_contact_with_upper_part_torso(self):
        """
        Check if the probe is in contact with the upper/top part of torso. Touching the torso on the sides should not count as contact.

        Returns:
            bool: True if probe both is in contact with upper part of torso and inside dsitance threshold from the torso center.
        """     
        # check for contact only if probe is in contact with upper part and close to torso center
        if  self._eef_xpos[-1] >= self._torso_xpos[-1] and np.linalg.norm(self._eef_xpos[:2] - self._torso_xpos[:2]) < 0.12:
            return self._check_probe_contact_with_torso()

        return False


    def _check_probe_contact_with_torso(self):
        """
        Check if the probe is in contact with the torso.

        NOTE This method utilizes the autogenerated geom names for MuJoCo-native composite objects
        
        Returns:
            bool: True if probe is in contact with torso
        """     
        # check contact with torso geoms based on autogenerated names
        gripper_contacts = self.get_contacts(self.robots[0].gripper)
        for contact in gripper_contacts:
            match = re.search("[G]\d+[_]\d+[_]\d+$", contact)
            if match != None:
                return True
    
        return False

    
    def _check_probe_contact_with_table(self):
        """
        Check if the probe is in contact with the tabletop.

        Returns:
            bool: True if probe is in contact with table
        """
        return self.check_contact(self.robots[0].gripper, "table_collision")


    def _check_terminated(self):
        """
        Check if the task has completed one way or another. The following conditions lead to termination:
            - Collision with table
            - Joint Limit reached

        Returns:
            bool: True if episode is terminated
        """

        terminated = False

        # Prematurely terminate if contacting the table with the probe
        if self._check_probe_contact_with_table():
            print(40 * '-' + " COLLIDED WITH TABLE " + 40 * '-')
            terminated = True

        # Prematurely terminate if reaching joint limits
        if self.robots[0].check_q_limits():
            print(40 * '-' + " JOINT LIMIT " + 40 * '-')
            #terminated = True

        # Prematurely terminate if task is success
       # if self._check_success():
       #     print(40 * '+' + " TASK SUCCESS " + 40 * '+')
        #    terminated = True

        return terminated

    
    def _get_examination_trajectory(self, n_pts):
        """
        Calculates the examination trajectory along the torso. The trajectory is calculated as a line 
        between the start and end position in the xy-plane.

        Args:
            n_pts (int): number of points along the trajectory

        Returns:
            [np.array]:  trajectory points (x,y,z)
        """
        assert n_pts > 1, "The number of points must be atleast 2"

        trajectory = np.zeros((n_pts, 3))
        start_pos = self._examination_start_xpos
        end_pos = self._examination_end_xpos
        
        dir_vec = np.array([
            end_pos[0] - start_pos[0], 
            end_pos[1] - start_pos[1], 
            end_pos[2] - start_pos[2]
            ])

        for i in range(n_pts):
            t = 1 / (n_pts-1) * i
            traj_pt = np.array([
                start_pos[0] + dir_vec[0] * t, 
                start_pos[1] + dir_vec[1] * t,
                start_pos[2] + dir_vec[2] * t
                ])
            trajectory[i] = traj_pt

        return trajectory


    @property
    def _torso_xpos(self):
        """
        Grabs torso center position

        Returns:
            np.array: torso pos (x,y,z)
        """
        return np.array(self.sim.data.body_xpos[self.torso_body_id])


    @property
    def _examination_start_xpos(self):
        """
        Grabs start position for ultrasound examination

        Returns:
            np.array: start pos (x,y,z)

        NOTE This function has several shortcomings:
            - The overall size of the torso is not known, hence hard-coded values must be used. 
            - The numeric values used in the function have been found through testing, and are prone to:
                * Changes in the torso size.
                * Rotation of the torso.
        """
        pos_x = self._torso_xpos[0] - 0.17 / 2
        pos_y = self._torso_xpos[1]
        pos_z = self._torso_xpos[2] + 0.056

        return np.array([pos_x, pos_y, pos_z])
    

    @property
    def _examination_end_xpos(self):
        """
        Grabs end position for ultrasound examination

        Returns:
            np.array: end pos (x,y,z)

        NOTE This function has several shortcomings:
            - The overall size of the torso is not known, hence hard-coded values must be used. 
            - The numeric values used in the function have been found through testing, and are prone to:
                * Changes in the torso size.
                * Rotation of the torso.
        """
        pos_x = self._torso_xpos[0] + 0.17 / 2
        pos_y = self._torso_xpos[1]
        pos_z = self._torso_xpos[2] + 0.056

        return np.array([pos_x, pos_y, pos_z])
    