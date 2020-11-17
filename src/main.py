import robosuite as suite
from robosuite.environments.base import register_env
from robosuite import load_controller_config

from my_environments import Ultrasound, FetchPush
from my_models.grippers import UltrasoundProbeGripper
from helper import register_gripper, plot_joint_pos, plot_forces_and_contact
from demos import controller_demo, \
    contact_btw_probe_and_body_demo, \
    standard_mujoco_py_demo, \
    drop_cube_on_body_demo, \
    change_parameters_of_soft_body_demo, \
    fetch_push_gym_demo


register_env(Ultrasound)
register_env(FetchPush)
register_gripper(UltrasoundProbeGripper)

#controller_demo('UR5e', save_data=False)
#contact_btw_probe_and_body_demo(2, 'main_test', save_data=False)
#standard_mujoco_py_demo()
#drop_cube_on_body_demo()
#change_parameters_of_soft_body_demo(3)
fetch_push_gym_demo()

#plot_joint_pos('data/UR5e_joint_pos_controller_test.csv')
#plot_forces_and_contact('data/panda_gripper_ee_forces_contact_btw_probe_and_body.csv', 'data/panda_gripper_contact_contact_btw_probe_and_body.csv')
