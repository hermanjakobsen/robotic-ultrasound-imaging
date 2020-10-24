import robosuite as suite
from robosuite.environments.base import register_env
from robosuite import load_controller_config

from my_environments import Ultrasound
from my_models.grippers import UltrasoundProbeGripper
from helper import register_gripper, plot_joint_pos
from demos import robosuite_simulation_controller_test, robosuite_simulation_contact_btw_probe_and_body, mujoco_py_simulation, body_softness_test
                    

register_env(Ultrasound)
register_gripper(UltrasoundProbeGripper)

controller_config = load_controller_config(default_controller='JOINT_POSITION')

env = suite.make(
            'Ultrasound',
            robots='UR5e',
            controller_configs=None,
            gripper_types='UltrasoundProbeGripper',
            has_renderer = True,
            has_offscreen_renderer= False,
            use_camera_obs=False,
            use_object_obs=False,
            control_freq = 50,
            render_camera = None,
            horizon=800      
        )

#robosuite_simulation_controller_test(env, env.horizon, 'UR5e')
robosuite_simulation_contact_btw_probe_and_body(env, env.horizon, 'UR5e')
#mujoco_py_simulation(env, env.horizon)
#body_softness_test()

#plot_joint_pos('data/UR5e_joint_pos_controller_test.csv')