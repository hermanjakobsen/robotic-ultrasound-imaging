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

        # input and output max and min (allow for either explicit lists or single numbers)
        self.input_max = self.nums2array(input_max, self.control_dim)
        self.input_min = self.nums2array(input_min, self.control_dim)
        self.output_max = self.nums2array(output_max, self.control_dim)
        self.output_min = self.nums2array(output_min, self.control_dim)

        # Control dimension
        self.control_dim = 3    # x and y position, and force in z-direction

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
        self.K_Plambda = 45                       # force gain
        self.K_Dlambda = self.K_Plambda*0.001     # force damping

        # Position control dynamics
        self.Pp = 60                         # x and y pos gain 
        self.Dp = self.Pp*0.1*0.5*0.5        # x and y pos damping

        # Orientation control dynamics
        self.Po = 120                                   # orientation gain
        self.Do = 40                                    # orientation damping

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

        # Initialize goals based on initial pos / ori
        self.goal_ori = T.convert_quat(T.mat2quat(np.array(self.initial_ee_ori_mat)), to="wxyz")    # (w, x, y, z) quaternion
        self.goal_pos = np.array(self.initial_ee_pos)

        # Initialize force measurements
        self.prev_z_force = 0
        self.z_force = 0


    def _initialize_measurements(self):
        self.probe_id = self.sim.model.body_name2id(self.robot.gripper.root_body)
        self.prev_z_force = self.sim.data.cfrc_ext[self.probe_id][-3:]
        self.z_force = self.prev_z_force


    # Must be called in environment's reset function
    def set_robot(self, robot):
        self.robot = robot
        self._initialize_measurements()
    

    # ------------ Helper functions -------------------------------- # 

    def quatdiff_in_euler_radians(self, quat_curr, quat_des):
        curr_mat = quaternion.as_rotation_matrix(quat_curr)
        des_mat = quaternion.as_rotation_matrix(quat_des)
        rel_mat = des_mat.T.dot(curr_mat)
        rel_quat = quaternion.from_rotation_matrix(rel_mat)
        vec = quaternion.as_float_array(rel_quat)[1:]
        if rel_quat.w < 0.0:
            vec = -vec
        return -des_mat.dot(vec)

        
    # Return the position and (relative) orientation 
    def get_x(self, p, ori, goal_ori):
        pos_x = p[:2]
        rel_ori = self.quatdiff_in_euler_radians(goal_ori, np.asarray(ori))
        return np.append(pos_x,rel_ori)


    # Fetch the estimated external forces and torques (h_e / F_ext)
    def construct_h_e(self, Fz):
        return np.array([0,0,Fz,0,0,0])


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
        delta_ori = self.quatdiff_in_euler_radians(np.asarray(ori), goal_ori)    
        if two_dim == True:
            return np.array([np.append(delta_pos,delta_ori)]).reshape([5,1])

        else:
            return np.append(delta_pos,delta_ori)


    # ------------  Calculation of torque -----------------

    # Calculate f_lambda (part of equation 9.62) as in equation (9.65) in chapter 9.3 of The Handbook of Robotics
    def calculate_f_lambda(self, f_d_ddot, f_d_dot, f_d, S_f, C, K_Dlambda, K_Plambda, z_force):
        S_f_inv = self.get_S_inv(S_f,C)
        K_dot = self.get_K_dot(S_f,S_f_inv,C)

        lambda_dot = np.linalg.multi_dot([S_f_inv, K_dot, self.J_full, self.joint_vel])
        #lambda_dot = self.get_lambda_dot()
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


    def run_controller(self):
        
        # position trajectories [placeholder]
        r_d_ddot = np.zeros(5)
        r_d_dot = np.zeros(5)     
        p_d = np.zeros(2)

        # force trajectories [placeholder]
        f_d_ddot = 0
        f_d_dot = 0
        f_d = 0

        # linear and angular velocity [placeholder]
        v   = np.zeros(5)

        # eef measurements
        pos = self.ee_pos
        ori = T.convert_quat(T.mat2quat(self.ee_ori_mat), to="wxyz")    # (w, x, y, z) quaternion
        self.z_force = self.sim.data.cfrc_ext[self.probe_id][-3:]

        # torque computations
        alpha_v = self.calculate_alpha_v(ori, r_d_ddot, r_d_dot, pos, p_d, v) 
        f_lambda = self.calculate_f_lambda(f_d_ddot, f_d_dot, f_d, self.S_f ,self.C, self.K_Dlambda, self.K_Plambda, self.z_force)
        alpha = self.calculate_alpha(self.S_v, alpha_v, self.C, self.S_f, -f_lambda)
        cartesian_inertia = np.linalg.inv(np.linalg.multi_dot([self.J_full, np.linalg.inv(self.mass_matrix), self.J_full.T]))

         # NOTE MODIFIED FROM DEFAULT. Removed external forces and such
        torque = np.array(np.linalg.multi_dot([self.J_full.T ,cartesian_inertia, alpha])).reshape([7,1])


        super().run_controller()

        # update measurements
        self.prev_z_force = self.z_force
        
        return torque


    @property
    def name(self):
        return "HMFC"