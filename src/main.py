import robosuite as suite
import gym
import os.path as osp
import numpy as np

from robosuite.environments.base import register_env
from robosuite import load_controller_config
from robosuite.wrappers import GymWrapper

from stable_baselines3 import PPO
from stable_baselines3.common.save_util import save_to_zip_file, load_from_zip_file
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from my_environments import Ultrasound, FetchPush
from my_models.grippers import UltrasoundProbeGripper
from helper import register_gripper, plot_joint_pos, plot_forces_and_contact, plot_gripper_position
from demos import controller_demo, \
    contact_btw_probe_and_body_demo, \
    standard_mujoco_py_demo, \
    drop_cube_on_body_demo, \
    change_parameters_of_soft_body_demo, \
    fetch_push_gym_demo


register_env(Ultrasound)
register_env(FetchPush)
register_gripper(UltrasoundProbeGripper)

## Simulations
#controller_demo('UR5e', save_data=False)
#contact_btw_probe_and_body_demo(1, 'main', save_data=False)
#standard_mujoco_py_demo()
#drop_cube_on_body_demo()
#change_parameters_of_soft_body_demo(3)
#fetch_push_gym_demo()

#plot_joint_pos('data/UR5e_joint_pos_controller_test.csv')
#plot_forces_and_contact('data/main_ee_forces_contact_btw_probe_and_body.csv', 'data/main_contact_contact_btw_probe_and_body.csv')
#plot_gripper_position('data/main_gripper_pos_contact_btw_probe_and_body.csv')

## RL

def wrap_env(env):
    wrapped_env = Monitor(env)                          # Needed for extracting eprewmean and eplenmean
    wrapped_env = DummyVecEnv([lambda : wrapped_env])   # Needed for all environments (e.g. used for mulit-processing)
    wrapped_env = VecNormalize(wrapped_env)             # Needed for improving training?
    return wrapped_env

controller_config = load_controller_config(default_controller='JOINT_POSITION')

env = GymWrapper(
        suite.make(
            'FetchPush',
            robots='UR5e',
            controller_configs=controller_config,
            gripper_types='UltrasoundProbeGripper',
            has_renderer = False,
            has_offscreen_renderer= False,
            use_camera_obs=False,
            use_object_obs=True,
            control_freq = 50,
            render_camera = None,
            horizon = 1000,
            reward_shaping = True,
        )
    )

env = wrap_env(env)
    
model = PPO('MlpPolicy', env, verbose=1, tensorboard_log='./ppo_fetchpush_tensorboard/')
model.learn(total_timesteps=2e6, tb_log_name='2M')

model.save('trained_models/2M')
env.save('trained_models/vec_normalize_2M.pkl')     # Save VecNormalize statistics

env_robo = GymWrapper(               # Should this environment also be vecnormalized when used for rendering?
        suite.make(
            'FetchPush',
            robots='UR5e',
            controller_configs=controller_config,
            gripper_types='UltrasoundProbeGripper',
            has_renderer = True,
            has_offscreen_renderer= False,
            use_camera_obs=False,
            use_object_obs=True,
            control_freq = 50,
            render_camera = None,
            horizon = 1000,
            reward_shaping = True
        )
    )

env = wrap_env(env_robo)

obs = env.reset()

low, high = env_robo.action_spec
eprew = 0
while True:
    action, _states = model.predict(obs, deterministic=True)
    #action = np.array([-np.pi/2, 0, 0, 0, 0, 0])#np.random.uniform(low, high)
    obs, reward, done, info = env.step(action)

    eprew += reward
    env_robo.render()
    if done:
        print(f'eprew: {eprew}')
        obs = env.reset()
        eprew = 0

env.close()