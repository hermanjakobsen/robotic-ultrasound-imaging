from collections import OrderedDict
import os
import numpy as np
import pandas as pd
import re
from klampt.model import trajectory
import roboticstoolbox as rtb

from spatialmath import SE3

from robosuite.utils.transform_utils import convert_quat, quat2mat, mat2euler
from robosuite.utils.mjcf_utils import CustomMaterial

from robosuite.environments.manipulation.single_arm_env import SingleArmEnv

from robosuite.models.tasks import ManipulationTask
from robosuite.utils.placement_samplers import UniformRandomSampler
from robosuite.utils.observables import Observable, sensor
from robosuite.models.base import MujocoModel

import robosuite.utils.transform_utils as T

from my_models.objects import SoftTorsoObject, BoxObject
from my_models.tasks import UltrasoundTask
from my_models.arenas import UltrasoundArena
from utils.quaternion import distance_quat, difference_quat


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
        table_full_size (3-tuple): x, y, and z dimensions of the table.
        table_friction (3-tuple): the three mujoco friction parameters for
            the table.
        use_camera_obs (bool): if True, every observation includes rendered image(s)
        use_object_obs (bool): if True, include object (cube) information in
            the observation.
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
        early_termination (bool): True if episode is allowed to finish early.
        save_data (bool): True if data from episode is collected and saved.
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
        table_friction=100*(1., 5e-3, 1e-4),
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
        early_termination=False,
        save_data=False,
    ):
        assert gripper_types == "UltrasoundProbeGripper",\
            "Tried to specify gripper other than UltrasoundProbeGripper in Ultrasound environment!"

        assert robots == "UR5e" or robots == "Panda", \
            "Robot must be UR5e or Panda!"

        assert "OSC" in controller_configs["type"], \
            "The robot controller must be of type OSC"

        # settings for table top
        self.table_full_size = table_full_size
        self.table_friction = table_friction
        self.table_offset = np.array((0, 0, 0.8))

        # settings for joint initialization noise (Gaussian)
        self.mu = 0
        self.sigma = 0.00

        # settings for contact force running mean
        self.alpha = 0.1    # decay factor (high alpha -> discounts older observations faster). Must be in (0, 1)

        # reward configuration 
        self.reward_scale = reward_scale
        self.reward_shaping = reward_shaping

        # error multipliers
        self.pos_error_mul = 50
        self.ori_error_mul = 0.2
        self.vel_error_mul = 45
        self.force_error_mul = 0.7

        # reward multipliers
        self.pos_reward_mul = 5
        self.ori_reward_mul = 1
        self.vel_reward_mul = 1
        self.force_reward_mul = 3

        # desired states
        self.goal_quat = np.array([-0.69192486,  0.72186726, -0.00514253, -0.01100909]) # Upright probe orientation found from experimenting (x,y,z,w)
        self.goal_velocity = 0.04           # norm of velocity vector
        self.goal_contact_z_force = 5       # (N)  

        # early termination configuration
        self.pos_error_threshold = 0.40
        self.ori_error_threshold = 0.10

        # examination trajectory
        self.traj_x_offset = 0.17         # offset from x_center of torso as to where to begin examination
        self.top_torso_offset = 0.044     # offset from z_center of torso to top of torso
        self.x_range = 0.15               # how large the torso is from center to end in x-direction
        self.y_range = 0.05 #0.11               # how large the torso is from center to end in y-direction
        self.grid_pts = 50                # how many points in the grid
                                            
        # whether to use ground-truth object states
        self.use_object_obs = use_object_obs

        # object placement initializer
        self.placement_initializer = placement_initializer

        # misc settings
        self.early_termination = early_termination
        self.save_data = save_data

        super().__init__(
            robots=robots,
            env_configuration=env_configuration,
            controller_configs=controller_configs,
            mount_types="default",
            gripper_types=gripper_types,
            initialization_noise=None,
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

        ee_current_ori = convert_quat(self._eef_xquat, to="wxyz")   # (w, x, y, z) quaternion
        ee_desired_ori = convert_quat(self.goal_quat, to="wxyz")

        # position
        self.pos_error = np.square(self.pos_error_mul * (self._eef_xpos[0:-1] - self.traj_pt[0:-1]))
        self.pos_reward = self.pos_reward_mul * np.exp(-1 * np.linalg.norm(self.pos_error))

        # orientation
        self.ori_error = self.ori_error_mul * distance_quat(ee_current_ori, ee_desired_ori)
        self.ori_reward = self.ori_reward_mul * np.exp(-1 * self.ori_error)

        # velocity
        self.vel_error =  np.square(self.vel_error_mul * (self.vel_running_mean - self.goal_velocity))
        self.vel_reward = self.vel_reward_mul * np.exp(-1 * np.linalg.norm(self.vel_error))
        
        # force
        self.force_error = np.square(self.force_error_mul * (self.z_contact_force_running_mean - self.goal_contact_z_force))
        self.force_reward = self.force_reward_mul * np.exp(-1 * self.force_error) if self._check_probe_contact_with_torso() else 0

        # add rewards
        reward += (self.pos_reward + self.ori_reward + self.vel_reward + self.force_reward)

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
                x_range=[0, 0], #[-0.12, 0.12],
                y_range=[0, 0], #[-0.12, 0.12],
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
        self.probe_id = self.sim.model.body_name2id(self.robots[0].gripper.root_body)
        

    def _setup_observables(self):
        """
        Sets up observables to be used for this environment. Creates object-based observables if enabled

        Returns:
            OrderedDict: Dictionary mapping observable names to its corresponding Observable object
        """
        observables = super()._setup_observables()

        pf = self.robots[0].robot_model.naming_prefix

        # Remove unnecessary observables
        del observables[pf + "joint_pos"]
        del observables[pf + "joint_pos_cos"]
        del observables[pf + "joint_pos_sin"]
        del observables[pf + "joint_vel"]
        del observables[pf + "gripper_qvel"]
        del observables[pf + "gripper_qpos"]
        del observables[pf + "eef_pos"]
        del observables[pf + "eef_quat"]

        sensors = []

        # probe information
        modality = f"{pf}proprio"       # Need to use this modality since proprio obs cannot be empty in GymWrapper

        @sensor(modality=modality)
        def eef_contact_force(obs_cache):
            return self.sim.data.cfrc_ext[self.probe_id][-3:]

        @sensor(modality=modality)
        def eef_torque(obs_cache):
            return self.robots[0].ee_torque

        @sensor(modality=modality)
        def eef_vel(obs_cache):
            return self.robots[0]._hand_vel

        @sensor(modality=modality)
        def eef_contact_force_z_diff(obs_cache):
            return self.z_contact_force_running_mean - self.goal_contact_z_force

        @sensor(modality=modality)
        def eef_vel_diff(obs_cache):
            return self.vel_running_mean - self.goal_velocity

        @sensor(modality=modality)
        def eef_pose_diff(obs_cache):
            pos_error = self._eef_xpos - self.traj_pt
            quat_error = difference_quat(self._eef_xquat, self.goal_quat)
            pose_error = np.concatenate((pos_error, quat_error))
            return pose_error

        sensors += [eef_contact_force, eef_torque, eef_vel, eef_contact_force_z_diff, eef_vel_diff, eef_pose_diff]

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
                self.sim.forward()      # update sim states
                
        # says if probe has been in touch with torso
        self.has_touched_torso = False

        # initial position of end-effector
        self.ee_initial_pos = self._eef_xpos

        # create trajectory
        self.trajectory = self.get_trajectory()
        
        # initialize trajectory step
        self.initial_traj_step = np.random.default_rng().uniform(low=0, high=self.num_waypoints - 1)
        self.traj_step = self.initial_traj_step                                    # step at which to evaluate trajectory. Must be in interval [0, num_waypoints - 1]
        
        # set first trajectory point
        self.traj_pt = self.trajectory.eval(self.traj_step)
        self.traj_pt_vel = self.trajectory.deriv(self.traj_step)

        # initialize controller's trajectory
        self.robots[0].controller.traj_pos = self.traj_pt
        self.robots[0].controller.traj_ori = T.quat2axisangle(self.goal_quat)

        # get initial joint positions for robot
        init_qpos = self._get_initial_qpos()
        init_qpos = self._add_noise_to_qpos(init_qpos, self.mu, self.sigma)

        # override initial robot joint positions
        self.robots[0].set_robot_joint_positions(init_qpos)

        # update controller with new initial joints
        self.robots[0].controller.update_initial_joints(init_qpos)

        # initialize running mean of velocity 
        self.vel_running_mean = np.linalg.norm(self.robots[0]._hand_vel)

        # initialize running mean of contact force
        self.z_contact_force_running_mean = self.sim.data.cfrc_ext[self.probe_id][-1]

        # initialize data collection
        if self.save_data:
            # simulation data
            self.data_ee_pos = np.array(np.zeros((self.horizon, 3)))
            self.data_ee_goal_pos = np.array(np.zeros((self.horizon, 3)))
            self.data_ee_vel = np.array(np.zeros((self.horizon, 3)))
            self.data_ee_goal_vel = np.array(np.zeros(self.horizon))
            self.data_ee_running_mean_vel = np.array(np.zeros(self.horizon))
            self.data_ee_quat = np.array(np.zeros((self.horizon, 4)))               # (x,y,z,w)
            self.data_ee_desired_quat = np.array(np.zeros((self.horizon, 4)))       # (x,y,z,w)
            self.data_ee_z_contact_force = np.array(np.zeros(self.horizon))
            self.data_ee_z_desired_contact_force = np.array(np.zeros(self.horizon))
            self.data_ee_z_running_mean_contact_force = np.array(np.zeros(self.horizon))
            self.data_is_contact = np.array(np.zeros(self.horizon))
            self.data_q_pos = np.array(np.zeros((self.horizon, self.robots[0].dof)))
            self.data_q_torques = np.array(np.zeros((self.horizon, self.robots[0].dof)))
            self.data_time = np.array(np.zeros(self.horizon))

            # reward data
            self.data_pos_reward = np.array(np.zeros(self.horizon))
            self.data_ori_reward = np.array(np.zeros(self.horizon))
            self.data_vel_reward = np.array(np.zeros(self.horizon))
            self.data_force_reward = np.array(np.zeros(self.horizon))

            # policy/controller data
            self.data_action = np.array(np.zeros((self.horizon, self.robots[0].action_dim)))


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

        # Convert to trajectory timstep
        normalizer = (self.horizon / (self.num_waypoints - 1))                  # equally many timesteps to reach each waypoint
        self.traj_step = self.timestep / normalizer + self.initial_traj_step

        # update trajectory point
        self.traj_pt = self.trajectory.eval(self.traj_step)

        # update controller's trajectory
        self.robots[0].controller.traj_pos = self.traj_pt

        # update velocity running mean (simple moving average)
        self.vel_running_mean += ((np.linalg.norm(self.robots[0]._hand_vel) - self.vel_running_mean) / self.timestep)

        # update contact force running mean (exponential moving average)
        self.z_contact_force_running_mean = self.alpha * self.sim.data.cfrc_ext[self.probe_id][-1] + (1 - self.alpha) * self.z_contact_force_running_mean

        # check for early termination
        if self.early_termination:
            done = done or self._check_terminated()

        # collect data
        if self.save_data:
            # simulation data
            self.data_ee_pos[self.timestep - 1] = self._eef_xpos
            self.data_ee_goal_pos[self.timestep - 1] = self.traj_pt
            self.data_ee_vel[self.timestep - 1] = self.robots[0]._hand_vel
            self.data_ee_goal_vel[self.timestep - 1] = self.goal_velocity
            self.data_ee_running_mean_vel[self.timestep -1] = self.vel_running_mean
            self.data_ee_quat[self.timestep - 1] = self._eef_xquat
            self.data_ee_desired_quat[self.timestep - 1] = self.goal_quat
            self.data_ee_z_contact_force[self.timestep - 1] = self.sim.data.cfrc_ext[self.probe_id][-1]
            self.data_ee_z_desired_contact_force[self.timestep - 1] = self.goal_contact_z_force
            self.data_ee_z_running_mean_contact_force[self.timestep - 1] = self.z_contact_force_running_mean
            self.data_is_contact[self.timestep - 1] = self._check_probe_contact_with_torso()
            self.data_q_pos[self.timestep - 1] = self.robots[0]._joint_positions
            self.data_q_torques[self.timestep - 1] = self.robots[0].torques
            self.data_time[self.timestep - 1] = (self.timestep - 1) / self.horizon * 100                         # percentage of completed episode

            # reward data
            self.data_pos_reward[self.timestep - 1] = self.pos_reward
            self.data_ori_reward[self.timestep - 1] = self.ori_reward
            self.data_vel_reward[self.timestep - 1] = self.vel_reward
            self.data_force_reward[self.timestep - 1] = self.force_reward

            # policy/controller data
            self.data_action[self.timestep - 1] = action
        
        # save data
        if done and self.save_data:
            # simulation data
            sim_data_fldr = "simulation_data"
            self._save_data(self.data_ee_pos, sim_data_fldr, "ee_pos")
            self._save_data(self.data_ee_goal_pos, sim_data_fldr, "ee_goal_pos")
            self._save_data(self.data_ee_vel, sim_data_fldr, "ee_vel")
            self._save_data(self.data_ee_goal_vel, sim_data_fldr, "ee_goal_vel")
            self._save_data(self.data_ee_running_mean_vel, sim_data_fldr, "ee_running_mean_vel")
            self._save_data(self.data_ee_quat, sim_data_fldr, "ee_quat")
            self._save_data(self.data_ee_desired_quat, sim_data_fldr, "ee_desired_quat")
            self._save_data(self.data_ee_z_contact_force, sim_data_fldr, "ee_z_contact_force")
            self._save_data(self.data_ee_z_desired_contact_force, sim_data_fldr, "ee_z_desired_contact_force")
            self._save_data(self.data_ee_z_running_mean_contact_force, sim_data_fldr, "ee_z_running_mean_contact_force")
            self._save_data(self.data_is_contact, sim_data_fldr, "is_contact")
            self._save_data(self.data_q_pos, sim_data_fldr, "q_pos")
            self._save_data(self.data_q_torques, sim_data_fldr, "q_torques")
            self._save_data(self.data_time, sim_data_fldr, "time")

            # reward data
            reward_data_fdlr = "reward_data"
            self._save_data(self.data_pos_reward, reward_data_fdlr, "pos")
            self._save_data(self.data_ori_reward, reward_data_fdlr, "ori")
            self._save_data(self.data_vel_reward, reward_data_fdlr, "vel")
            self._save_data(self.data_force_reward, reward_data_fdlr, "force")

            # policy/controller data
            self._save_data(self.data_action, "policy_data", "action")


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
        return False


    def _check_terminated(self):
        """
        Check if the task has completed one way or another. The following conditions lead to termination:
            - Collision with table
            - Joint Limit reached
            - Deviates from trajectory position
            - Deviates from desired orientation when in contact with torso
            - Loses contact with torso

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
            terminated = True

        # Prematurely terminate if probe deviates away from trajectory (represented by a low position reward)
        if np.linalg.norm(self.pos_error) > self.pos_error_threshold:
            print(40 * '-' + " DEVIATES FROM TRAJECTORY " + 40 * '-')
            terminated = True

        # Prematurely terminate if probe deviates from desired orientation when touching probe
        if self._check_probe_contact_with_torso() and self.ori_error > self.ori_error_threshold:
            print(40 * '-' + " (TOUCHING BODY) PROBE DEVIATES FROM DESIRED ORIENTATION " + 40 * '-')
            terminated = True

        # Prematurely terminate if probe loses contact with torso
        if self.has_touched_torso and not self._check_probe_contact_with_torso():
            print(40 * '-' + " LOST CONTACT WITH TORSO " + 40 * '-')
            terminated = True

        return terminated


    def _get_contacts_objects(self, model):
        """
        Checks for any contacts with @model (as defined by @model's contact_geoms) and returns the set of
        contact objects currently in contact with that model (excluding the geoms that are part of the model itself).

        Args:
            model (MujocoModel): Model to check contacts for.

        Returns:
            set: Unique contact objects containing information about contacts with this model.

        Raises:
            AssertionError: [Invalid input type]
        """
        # Make sure model is MujocoModel type
        assert isinstance(model, MujocoModel), \
            "Inputted model must be of type MujocoModel; got type {} instead!".format(type(model))
        contact_set = set()
        for contact in self.sim.data.contact[: self.sim.data.ncon]:
            # check contact geom in geoms; add to contact set if match is found
            g1, g2 = self.sim.model.geom_id2name(contact.geom1), self.sim.model.geom_id2name(contact.geom2)
            if g1 in model.contact_geoms or g2 in model.contact_geoms:
                contact_set.add(contact)

        return contact_set


    def _check_probe_contact_with_upper_part_torso(self):
        """
        Check if the probe is in contact with the upper/top part of torso. Touching the torso on the sides should not count as contact.

        Returns:
            bool: True if probe both is in contact with upper part of torso and inside distance threshold from the torso center.
        """     
        # check for contact only if probe is in contact with upper part and close to torso center
        if  self._eef_xpos[-1] >= self._torso_xpos[-1] and np.linalg.norm(self._eef_xpos[:2] - self._torso_xpos[:2]) < 0.14:
            return self._check_probe_contact_with_torso()

        return False


    def _check_probe_contact_with_torso(self):
        """
        Check if the probe is in contact with the torso.

        NOTE This method utilizes the autogenerated geom names for MuJoCo-native composite objects
        
        Returns:
            bool: True if probe is in contact with torso
        """     
        gripper_contacts = self._get_contacts_objects(self.robots[0].gripper)
        reg_ex = "[G]\d+[_]\d+[_]\d+$"

        # check contact with torso geoms based on autogenerated names
        for contact in gripper_contacts:
            g1, g2 = self.sim.model.geom_id2name(contact.geom1), self.sim.model.geom_id2name(contact.geom2) 
            match1 = re.search(reg_ex, g1)
            match2 = re.search(reg_ex, g2)
            if match1 != None or match2 != None:
                contact_normal_axis = contact.frame[:3]
                self.has_touched_torso = True
                return True
    
        return False

    
    def _check_probe_contact_with_table(self):
        """
        Check if the probe is in contact with the tabletop.

        Returns:
            bool: True if probe is in contact with table
        """
        return self.check_contact(self.robots[0].gripper, "table_collision")
    

    def get_trajectory(self):
        """
        Calculates a trajectory between two waypoints on the torso. The waypoints are extracted from a grid on the torso.
        The first waypoint is given at time t=0, and the second waypoint is given at t=1.

        Args:

        Returns:
            (klampt.model.trajectory Object):  trajectory
        """
        grid = self._get_torso_grid()

        start_point = self._get_waypoint(grid)
        end_point = self._get_waypoint(grid)

        milestones = np.array([start_point, end_point])
        self.num_waypoints = np.size(milestones, 0)

        return trajectory.Trajectory(milestones=milestones)


    def _get_torso_grid(self):
        """
        Creates a 2D grid in the xy-plane on the top of the torso.

        Args:

        Returns:
            (numpy.array):  grid. First row contains x-coordinates and the second row contains y-coordinates.
        """
        x = np.linspace(-self.x_range + self._torso_xpos[0] + 0.03, self.x_range + self._torso_xpos[0], num=self.grid_pts)  # add offset in negative range due to weird robot angles close to robot base
        y = np.linspace(-self.y_range + self._torso_xpos[1], self.y_range + self._torso_xpos[1], num=self.grid_pts)

        x = np.array([x])
        y = np.array([y])

        return np.concatenate((x, y))

    
    def _get_waypoint(self, grid):
        """
        Extracts a random waypoint from the grid.

        Args:

        Returns:
            (numpy.array):  waypoint
        """
        x_pos = np.random.choice(grid[0])
        y_pos = np.random.choice(grid[1])
        z_pos = self._torso_xpos[-1] + self.top_torso_offset

        return np.array([x_pos, y_pos, z_pos])
        
    
    def _get_initial_qpos(self):
        """
        Calculates the initial joint position for the robot based on the initial desired pose (self.traj_pt, self.goal_quat).

        Args:

        Returns:
            (np.array): n joint positions 
        """
        pos = self._convert_robosuite_to_toolbox_xpos(self.traj_pt)
        ori_euler = mat2euler(quat2mat(self.goal_quat))

        # desired pose
        T = SE3(pos) * SE3.RPY(ori_euler)

        # find initial joint positions
        if self.robots[0].name == "UR5e":
            robot = rtb.models.DH.UR5()
            sol = robot.ikine_min(T, q0=self.robots[0].init_qpos)

            # flip last joint around (pi)
            sol.q[-1] -= np.pi
            return sol.q

        elif self.robots[0].name == "Panda":
            robot = rtb.models.DH.Panda()
            sol = robot.ikine_min(T, q0=self.robots[0].init_qpos)
            return sol.q


    def _convert_robosuite_to_toolbox_xpos(self, pos):
        """
        Converts origin used in robosuite to origin used for robotics toolbox. Also transforms robosuite world frame (vectors x, y, z) to
        to correspond to world frame used in toolbox.

        Args:
            pos (np.array): position (x,y,z) given in robosuite coordinates and frame 

        Returns:
            (np.array):  position (x,y,z) given in robotics toolbox coordinates and frame
        """
        xpos_offset = self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0])[0]
        zpos_offset = self.robots[0].robot_model.top_offset[-1] - 0.016

        # the numeric offset values have been found empirically, where they are chosen so that 
        # self._eef_xpos matches the desired position.
        if self.robots[0].name == "UR5e":
            return np.array([-pos[0] + xpos_offset + 0.08, -pos[1] + 0.025, pos[2] - zpos_offset + 0.15]) 

        if self.robots[0].name == "Panda":
            return np.array([pos[0] - xpos_offset - 0.06, pos[1], pos[2] - zpos_offset + 0.105])


    def _add_noise_to_qpos(self, qpos, mu, sigma):
        """
        Adds Gaussian noise (variance) to the joint positions.

        Args:
            qpos (np.array): joint positions 
            mu (float): mean (“centre”) of the distribution
            sigma (float): standard deviation (spread or “width”) of the distribution. Must be non-negative

        Returns:
            (np.array):  joint positions with added noise
        """
        noise = np.random.normal(mu, sigma, qpos.size)
        return qpos + noise


    def _save_data(self, data, fldr, filename):
        """
        Saves data to desired path.

        Args:
            data (np.array): Data to be saved 
            fldr (string): Name of destination folder
            filename (string): Name of file

        Returns:
        """
        os.makedirs(fldr, exist_ok=True)

        idx = 1
        path = os.path.join(fldr, filename + "_" + str(idx) + ".csv")

        while os.path.exists(path):
            idx += 1
            path = os.path.join(fldr, filename + "_" + str(idx) + ".csv")

        pd.DataFrame(data).to_csv(path, header=None, index=None)


    @property
    def _torso_xpos(self):
        """
        Grabs torso center position

        Returns:
            np.array: torso pos (x,y,z)
        """
        return np.array(self.sim.data.body_xpos[self.torso_body_id]) 