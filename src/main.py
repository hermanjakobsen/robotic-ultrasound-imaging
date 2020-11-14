import robosuite as suite
from robosuite.environments.base import register_env
from robosuite import load_controller_config

from my_environments import Ultrasound
from my_models.grippers import UltrasoundProbeGripper
from helper import register_gripper, plot_joint_pos, plot_forces_and_contact
from demos import robosuite_simulation_controller_test, \
    robosuite_simulation_contact_btw_probe_and_body, \
    mujoco_py_simulation, \
    body_softness_test, \
    change_parameters_of_soft_body_demo

               
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

#robosuite_simulation_controller_test(env, 'UR5e', save_data=False)
#robosuite_simulation_contact_btw_probe_and_body('main_test', save_data=False)
#mujoco_py_simulation(env)
#body_softness_test()
change_parameters_of_soft_body_demo(5)

#plot_joint_pos('data/UR5e_joint_pos_controller_test.csv')
#plot_forces_and_contact('data/panda_gripper_ee_forces_contact_btw_probe_and_body.csv', 'data/panda_gripper_contact_contact_btw_probe_and_body.csv')
