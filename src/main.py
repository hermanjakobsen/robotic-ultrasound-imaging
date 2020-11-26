import robosuite as suite
import gym
import os.path as osp
import tensorflow as tf
import numpy as np

# Used to remove Cudlas status error
gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)

from robosuite.environments.base import register_env
from robosuite import load_controller_config
from robosuite.wrappers import GymWrapper

from baselines.ppo2.ppo2 import learn
from baselines.common.vec_env import DummyVecEnv, VecEnv, VecNormalize
from baselines.bench import Monitor

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
#contact_btw_probe_and_body_demo('main', save_data=True)
#standard_mujoco_py_demo()
#drop_cube_on_body_demo()
#change_parameters_of_soft_body_demo(3)
#fetch_push_gym_demo()

#plot_joint_pos('data/UR5e_joint_pos_controller_test.csv')
#plot_forces_and_contact('data/main_ee_forces_contact_btw_probe_and_body.csv', 'data/main_contact_contact_btw_probe_and_body.csv')
#plot_gripper_position('data/main_gripper_pos_contact_btw_probe_and_body.csv')

## RL
controller_config = load_controller_config(default_controller='JOINT_POSITION')
network='mlp'
seed = None

# Exclude renderer while training
env_robo = GymWrapper(
    suite.make(
        'FetchPush',
        robots='UR5e',
        controller_configs=controller_config,
        gripper_types='UltrasoundProbeGripper',
        has_renderer = False,
        has_offscreen_renderer= False,
        use_camera_obs=False,
        use_object_obs=True,
        render_camera = None,
        reward_shaping = True,
        control_freq = 50,
    )
)

env = Monitor(env_robo, None, allow_early_resets=True)
env = DummyVecEnv([lambda: env])
env = VecNormalize(env)

# Training
model = learn(network=network, env=env, seed=seed, total_timesteps=2e5)

save_path = osp.expanduser("trained_models/test/")
ckpt = tf.train.Checkpoint(model=model)
manager = tf.train.CheckpointManager(ckpt, save_path, max_to_keep=None)
manager.save()

# Create identical environment with renderer
env_robo = GymWrapper(
    suite.make(
        'FetchPush',
        robots='UR5e',
        controller_configs=controller_config,
        gripper_types='UltrasoundProbeGripper',
        has_renderer = True,
        has_offscreen_renderer= False,
        use_camera_obs=False,
        use_object_obs=True,
        render_camera = None,
        reward_shaping = True,
        control_freq = 50,
    )
)
env = Monitor(env_robo, None, allow_early_resets=True)
env = DummyVecEnv([lambda: env])
env = VecNormalize(env)

# Train loaded model for zero timesteps (i.e. load trained model)
model = learn(network=network, env=env, seed=seed, total_timesteps=0, load_path="trained_models/test/")

print("Running trained model")
obs = env.reset()

state = model.initial_state if hasattr(model, 'initial_state') else None
dones = np.zeros((1,))

episode_rew = np.zeros(env.num_envs) if isinstance(env, VecEnv) else np.zeros(1)
while True:
    
    if state is not None:
        actions, _, state, _ = model.step(obs,S=state, M=dones)
    else:
        actions, _, _, _ = model.step(obs)


    obs, rew, done, _ = env.step(actions)
    episode_rew += rew
    env_robo.render()
    done_any = done.any() if isinstance(done, np.ndarray) else done
    if done_any:
        for i in np.nonzero(done)[0]:
            print('episode_rew={}'.format(episode_rew[i]))
            episode_rew[i] = 0

    env.close()
