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
from robosuite.models.arenas import EmptyArena

import robosuite.utils.transform_utils as T

from my_models.objects import BoxObject

class HMFC(SingleArmEnv):
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
        reward_scale=1.0,
        reward_shaping=False,
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
            "Tried to specify gripper other than UltrasoundProbeGripper in HMFC environment!"

        assert robots == "Panda", \
            "Robot must be Panda!"

        assert "HMFC" in controller_configs["type"], \
            "The robot controller must be of type HMFC"
        

        # misc settings
        self.early_termination = early_termination
        self.save_data = save_data

        self.goal_quat = np.array([-0.69192486,  0.72186726, -0.00514253, -0.01100909]) # Upright probe orientation found from experimenting (x,y,z,w)


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


        return reward


    def _load_model(self):
        """
        Loads an xml model, puts it in self.model
        """
        super()._load_model()

        # load model for table top workspace
        mujoco_arena = EmptyArena()

        # Arena always gets set to zero origin
        mujoco_arena.set_origin([0, 0, 0])

        # initialize objects of interest
        self.cube = BoxObject(name="cube")

        # task includes arena, robot, and objects of interest
        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots], 
            mujoco_objects=[self.cube]
        )


    def _setup_references(self):
        """
        Sets up references to important components. A reference is typically an
        index or a list of indices that point to the corresponding elements
        in a flatten array, which is how MuJoCo stores physical simulation data.
        """
        super()._setup_references()

        # additional object references from this env
        self.probe_id = self.sim.model.body_name2id(self.robots[0].gripper.root_body)
        

    def _setup_observables(self):
        """
        Sets up observables to be used for this environment. Creates object-based observables if enabled

        Returns:
            OrderedDict: Dictionary mapping observable names to its corresponding Observable object
        """
        observables = super()._setup_observables()

        return observables


    def _reset_internal(self):
        """
        Resets simulation internal configurations.
        """
        super()._reset_internal()

        # initial position of end-effector
        self.ee_initial_pos = self._eef_xpos

        # create trajectory
        self.trajectory = self.get_trajectory()
        
        # initialize trajectory step
        self.initial_traj_step = 0#np.random.default_rng().uniform(low=0, high=self.num_waypoints - 1)
        self.traj_step = self.initial_traj_step                                    # step at which to evaluate trajectory. Must be in interval [0, num_waypoints - 1]
        
        # set first trajectory point
        self.traj_pt = self.trajectory.eval(self.traj_step)
        self.traj_pt_vel = self.trajectory.deriv(self.traj_step)

        # give controller access to robot (and its measurements)
        if self.robots[0].controller.name == "HMFC":
            self.robots[0].controller.set_robot(self.robots[0])

        # initialize controller's trajectory
        self.robots[0].controller.traj_pos = self.traj_pt
        self.robots[0].controller.traj_ori = T.quat2axisangle(self.goal_quat)

        # get initial joint positions for robot
        init_qpos = self._get_initial_qpos()

        # override initial robot joint positions
        self.robots[0].set_robot_joint_positions(init_qpos)

        # update controller with new initial joints
        self.robots[0].controller.update_initial_joints(init_qpos)

        # initialize data collection
        if self.save_data:
            # simulation data
            self.data_ee_pos = np.array(np.zeros((self.horizon, 2)))
            self.data_ee_goal_pos = np.array(np.zeros((self.horizon, 2)))
            self.data_ee_force = np.array(np.zeros(self.horizon))
            self.data_ee_force_mean = np.array(np.zeros(self.horizon))
            self.data_ee_goal_force = np.array(np.zeros(self.horizon))
            self.data_time = np.array(np.zeros(self.horizon))


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

        # check for early termination
        if self.early_termination:
            done = done or self._check_terminated()

        # collect data
        if self.save_data:
            controller = self.robots[0].controller

            # simulation data
            self.data_ee_pos[self.timestep - 1] = self._eef_xpos[:-1]
            self.data_ee_goal_pos[self.timestep - 1] = controller.p_d
            self.data_ee_force[self.timestep - 1] = controller.z_force
            self.data_ee_force_mean[self.timestep - 1] = controller.z_force_running_mean
            self.data_ee_goal_force[self.timestep - 1] = controller.f_d
            self.data_time[self.timestep - 1] = (self.timestep - 1) / self.horizon * 100                         # percentage of completed episode

        
        # save data
        if done and self.save_data:
            # simulation data
            sim_data_fldr = "hmfc_test_data"
            self._save_data(self.data_ee_pos, sim_data_fldr, "ee_pos")
            self._save_data(self.data_ee_goal_pos, sim_data_fldr, "ee_goal_pos")
            self._save_data(self.data_ee_force, sim_data_fldr, "ee_force")
            self._save_data(self.data_ee_force_mean, sim_data_fldr, "ee_force_mean")
            self._save_data(self.data_ee_goal_force, sim_data_fldr, "ee_goal_force")
            self._save_data(self.data_time, sim_data_fldr, "time")



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
        Returns:
            bool: True if episode is terminated
        """

        terminated = False

        return terminated


    def get_trajectory(self):
        """
        Calculates a trajectory between two waypoints on the torso. The waypoints are extracted from a grid on the torso.
        The first waypoint is given at time t=0, and the second waypoint is given at t=1.

        Args:

        Returns:
            (klampt.model.trajectory Object):  trajectory
        """
        start_point =  [0.3, 0, 0.299]
        end_point =  [0.5, -0.2, 0.299] 

        milestones = np.array([start_point, end_point])
        self.num_waypoints = np.size(milestones, 0)

        return trajectory.Trajectory(milestones=milestones)
        
    
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

        # the numeric offset values have been found empirically, where they are chosen so that 
        # self._eef_xpos matches the desired position.
        if self.robots[0].name == "UR5e":
            return np.array([-pos[0] + 0.08, -pos[1] + 0.025, pos[2] + 0.15]) 

        if self.robots[0].name == "Panda":
            return np.array([pos[0] - 0.06, pos[1], pos[2] + 0.03])


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
