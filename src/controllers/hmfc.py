from robosuite.controllers.base_controller import Controller
import robosuite.utils.transform_utils as T
import quaternion

import numpy as np

class HybridMotionForceController(Controller):


    def __init__(self,
                 sim,
                 eef_name,
                 joint_indexes,
                 actuator_range,
                 input_max=1,
                 input_min=-1,
                 output_max=(0.05, 0.05, 0.05, 0.5, 0.5, 0.5),
                 output_min=(-0.05, -0.05, -0.05, -0.5, -0.5, -0.5),
                 policy_freq=20,
                 **kwargs # does nothing; used so no error raised when dict is passed with extra terms used previously
                 ):

        super().__init__(
            sim,
            eef_name,
            joint_indexes,
            actuator_range,
        )

        # Control dimension
        self.control_dim = 0    # x and y position, and force in z-direction

        # input and output max and min (allow for either explicit lists or single numbers)
        self.input_max = self.nums2array(input_max, self.control_dim)
        self.input_min = self.nums2array(input_min, self.control_dim)
        self.output_max = self.nums2array(output_max, self.control_dim)
        self.output_min = self.nums2array(output_min, self.control_dim)

        # control frequency
        self.control_freq = policy_freq

        # Subspace
        self.S_f = np.array([[0, 0, 1, 0, 0, 0]]).reshape([6,1])     # force-control-subspace (only doing force control in z)

        self.S_v = np.array([[1, 0, 0, 0, 0],
                        [0, 1, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                        [0, 0, 1, 0, 0],
                        [0, 0, 0, 1, 0],
                        [0, 0, 0, 0, 1]]).reshape([6,5])             # motion-control-subspace (x, y, ori_x, ori_y, ori_z)

        # Stiffness of the interaction [should be estimated (this one is chosen at random)]
        self.K = np.array([[1, 0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0, 0],
                    [0, 0, 100, 0, 0, 0],
                    [0, 0, 0, 5, 0, 0],
                    [0, 0, 0, 0, 5, 0],
                    [0, 0, 0, 0, 0, 1]]).reshape([6,6])

        self.C = np.linalg.inv(self.K)

        # Force control dynamics
        self.K_Plambda = 1                       # force gain
        self.K_Dlambda = self.K_Plambda*0.001     # force damping

        # Position control dynamics
        self.Pp = 1                         # x and y pos gain 
        self.Dp = self.Pp*0.1*0.5*0.5        # x and y pos damping

        # Orientation control dynamics
        self.Po = 1                                   # orientation gain
        self.Do = 1                                    # orientation damping

        self.K_Pr = np.array([[self.Pp, 0, 0, 0, 0],    # Stiffness matrix
                              [0, self.Pp, 0, 0, 0],
                              [0, 0, self.Po, 0, 0],
                              [0, 0, 0, self.Po, 0],
                              [0, 0, 0, 0, self.Po]])

        self.K_Dr = np.array([[self.Dp, 0, 0, 0, 0],    # Damping matrix
                              [0, self.Dp, 0, 0, 0],
                              [0, 0, self.Do, 0, 0],
                              [0, 0, 0, self.Do, 0],
                              [0, 0, 0, 0, self.Do]])

        # Initialize robot
        self.robot = None
        self.probe_id = None

        # Initialize goals 
        self.r_d_ddot = np.zeros(5)         # position trajectories [placeholder]
        self.r_d_dot = np.zeros(5)     
        self.p_d = np.zeros(2)

        self.f_d_ddot = 0                   # force trajectories [placeholder]
        self.f_d_dot = 0
        self.f_d = 0

        self.goal_ori = T.convert_quat(np.array([-0.69192486,  0.72186726, -0.00514253, -0.01100909]), to="wxyz")    # (w, x, y, z) quaternion

        # Initialize measurements
        self.z_force = 0                    # contact force in z-direction
        self.prev_z_force = 0
        self.v = np.zeros(5)                # angular and linear velocity



    def _initialize_measurements(self):
        self.probe_id = self.sim.model.body_name2id(self.robot.gripper.root_body)
        self.z_force = self.sim.data.cfrc_ext[self.probe_id][-1]
        self.prev_z_force = self.z_force
        self.v = self.get_eef_velocity().reshape([5, 1])


    # Must be called in environment's reset function
    def set_robot(self, robot):
        self.robot = robot
        self._initialize_measurements()
    

    # ------------ Helper functions -------------------------------- # 

    def quatdiff_in_euler_radians(self, quat_curr, quat_des):
        curr_mat = T.quat2mat(T.convert_quat(quat_curr, to="xyzw"))
        des_mat = T.quat2mat(T.convert_quat(quat_des, to="xyzw"))
        rel_mat = des_mat.T.dot(curr_mat)
        rel_quat = quaternion.from_rotation_matrix(rel_mat)
        vec = quaternion.as_float_array(rel_quat)[1:]
        if rel_quat.w < 0.0:
            vec = -vec
        return -des_mat.dot(vec)


    # Fetch linear (excluding z) and angular velocity of eef
    def get_eef_velocity(self):
        lin_v = self.robot._hand_vel[:-1]
        ang_v = self.robot._hand_ang_vel
        return np.concatenate((lin_v, ang_v))


    # Fetch the derivative of the force
    def get_lambda_dot(self):
        return (self.z_force - self.prev_z_force) / ( 1 / self.control_freq)


    # Fetch the psudoinverse of S_f/S_v as in equation (9.34) in chapter 9.3 of The Handbook of Robotics
    def get_S_inv(self, S, C):
        a = np.linalg.inv(np.linalg.multi_dot([S.T,C,S]))
        return np.array(np.linalg.multi_dot([a,S.T,C]))


    # Fetch K' as in equation (9.49) in chapter 9.3 of The Handbook of Robotics
    def get_K_dot(self, S_f, S_f_inv, C):
        return np.array(np.linalg.multi_dot([S_f,S_f_inv,np.linalg.inv(C)])).reshape([6,6])


    # Calculate the error in position and orientation (in the subspace subject to motion control)
    def get_delta_r(self, ori, goal_ori, p, p_d, two_dim = True):
        delta_pos = p_d - p[:2]
        delta_ori = self.quatdiff_in_euler_radians(ori, goal_ori)    
        if two_dim == True:
            return np.array([np.append(delta_pos,delta_ori)]).reshape([5,1])

        else:
            return np.append(delta_pos,delta_ori)


    # ------------  Calculation of torque -----------------

    # Calculate f_lambda (part of equation 9.62) as in equation (9.65) in chapter 9.3 of The Handbook of Robotics
    def calculate_f_lambda(self, f_d_ddot, f_d_dot, f_d, S_f, C, K_Dlambda, K_Plambda, z_force):
        S_f_inv = self.get_S_inv(S_f,C)
        K_dot = self.get_K_dot(S_f,S_f_inv,C)

        #lambda_dot = np.linalg.multi_dot([S_f_inv, K_dot, self.J_full, self.joint_vel])

        lambda_dot = self.get_lambda_dot()         
        lambda_a = f_d_ddot 
        lambda_b = np.array(np.dot(K_Dlambda,(f_d_dot-lambda_dot)))
        lambda_c = np.dot(K_Plambda,(f_d-z_force))
 
        return max(lambda_a + lambda_b + lambda_c,0)


    # Get the subproducts of f_lambda (for plotting/troubleshooting)
    def get_f_lambda_subproducts(self, f_d_ddot, f_d_dot, f_d, i,time_per_iteration, S_f, C, K_Dlambda, K_Plambda, z_force, h_e_hist, jacobian, joint_names, sim):
        S_f_inv = self.get_S_inv(S_f, C)
        K_dot = self.get_K_dot(S_f, S_f_inv, C)
        if sim: 
            lambda_dot = self.get_lambda_dot()
        else: 
            lambda_dot = (np.linalg.multi_dot([S_f_inv,K_dot,jacobian, self.joint_vel])) # At least not correct for interaction tasks in simulation (due to incorrect readings of joint velocity)
        lambda_a = f_d_ddot
        lambda_b = np.array(np.dot(K_Dlambda,(f_d_dot-lambda_dot)))
        lambda_c = np.dot(K_Plambda,(f_d-z_force))
        return lambda_dot, lambda_a, lambda_b, lambda_c, (lambda_a + lambda_b + lambda_c)


    # Calculate alpha_v (part of equation 9.62) as on page 213 in chapter 9.3 of The Handbook of Robotics
    def calculate_alpha_v(self, ori, r_d_ddot, r_d_dot, p, p_d, v):
        return (r_d_ddot.reshape([5, 1]) + np.array(np.dot(self.K_Dr, r_d_dot.reshape([5, 1]) - v)).reshape([5, 1])+ np.array(np.dot(self.K_Pr, self.get_delta_r(ori, self.goal_ori, p, p_d))).reshape([5, 1]))


    # Calculate alpha (part of equation 9.16) as in equation (9.62) in chapter 9.3 of The Handbook of Robotics
    def calculate_alpha(self, S_v, alpha_v,C,S_f,f_lambda):
        S_v_inv = self.get_S_inv(S_v,C)
        P_v = np.array(np.dot(S_v,S_v_inv))
        C_dot = np.array(np.dot((np.identity(6)-(P_v).reshape([6,6])),C)).reshape([6,6])
        return np.array(np.dot(S_v, alpha_v)).reshape([6,1]) + f_lambda*np.array(np.dot(C_dot,S_f)).reshape([6,1])


    def set_goal(self, action):
        
        self.r_d_ddot = np.zeros(5)         # position trajectories [placeholder]
        self.r_d_dot = np.zeros(5)     
        self.p_d = np.zeros(2)

        self.f_d_ddot = 0                   # force trajectories [placeholder]
        self.f_d_dot = 0
        self.f_d = 0
         

    def run_controller(self):

        # eef measurements
        self.z_force = self.sim.data.cfrc_ext[self.probe_id][-1]
        self.v = self.get_eef_velocity().reshape([5, 1])
        
        pos = self.ee_pos
        ori = T.convert_quat(T.mat2quat(self.ee_ori_mat), to="wxyz")    # (w, x, y, z) quaternion

        h_e = np.array([0, 0, self.z_force, 0, 0, 0])

        # control law
        alpha_v = self.calculate_alpha_v(ori, self.r_d_ddot, self.r_d_dot, pos, self.p_d, self.v) 
        f_lambda = self.calculate_f_lambda(self.f_d_ddot, self.f_d_dot, self.f_d, self.S_f ,self.C, self.K_Dlambda, self.K_Plambda, self.z_force)

        alpha = self.calculate_alpha(self.S_v, alpha_v, self.C, self.S_f, -f_lambda)
        cartesian_inertia = np.linalg.inv(np.linalg.multi_dot([self.J_full, np.linalg.inv(self.mass_matrix), self.J_full.T]))

        # torque computations
        external_torque = np.dot(self.J_full.T, h_e).reshape([7,1])
        torque = np.array(np.linalg.multi_dot([self.J_full.T ,cartesian_inertia, alpha])) + external_torque # NOTE MODIFIED FROM DEFAULT. Removed external forces and such
        torque = torque.flatten()

        # update measurements
        self.prev_z_force = self.z_force

        # Always run superclass call for any cleanups at the end
        super().run_controller()

        return torque


    def update_initial_joints(self, initial_joints):
        # First, update from the superclass method
        super().update_initial_joints(initial_joints)

        # We also need to reset the goal in case the old goals were set to the initial confguration
        self.reset_goal()


    def reset_goal(self):
        """
        Resets the goal to the current state of the robot
        """
        self.goal_ori = T.convert_quat(T.mat2quat(self.ee_ori_mat), to="wxyz")
        self.p_d = np.array(self.ee_pos)[:-1]


    @property
    def control_limits(self):
        """
        Returns the limits over this controller's action space, overrides the superclass property

            2-tuple:
                - (np.array) minimum action values
                - (np.array) maximum action values
        """
        low, high = self.input_min, self.input_max
        return low, high


    @property
    def name(self):
        return "HMFC"
