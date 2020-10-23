import robosuite as suite
from robosuite.environments.base import register_env

from my_environments import Ultrasound
from my_models.grippers import UltrasoundProbeGripper
from helper import register_gripper
from demos import robosuite_simulation_controller_test, mujoco_py_simulation, body_softness_test
 
register_env(Ultrasound)
register_gripper(UltrasoundProbeGripper)

env = suite.make(
            'Ultrasound',
            robots='UR5e',
            gripper_types='UltrasoundProbeGripper',
            has_renderer = True,
            has_offscreen_renderer= False,
            use_camera_obs=False,
            use_object_obs=False,
            control_freq = 50,
            render_camera = None,
            horizon=5000      
        )


robosuite_simulation_controller_test(env, env.horizon)
#mujoco_py_simulation(env, env.horizon)
#body_softness_test()