import robosuite as suite
import os

from robosuite.wrappers import GymWrapper
from robosuite import load_controller_config
from robosuite.environments.base import register_env

from stable_baselines3 import PPO
from stable_baselines3.common.save_util import save_to_zip_file, load_from_zip_file
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.monitor import Monitor

from my_models.grippers import UltrasoundProbeGripper
from my_environments import Ultrasound, FetchPush
from helper import register_gripper

def make_training_env(env_id, options, rank, seed=0):
    """
    Utility function for multiprocessed env.

    :param env_id: (str) the environment ID
    :param options: (dict) additional arguments to pass to the specific environment class initializer
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    """
    def _init():
        env = GymWrapper(suite.make(env_id, **options))
        env = Monitor(env)
        env.seed(seed + rank)
        return env
    set_random_seed(seed)
    return _init

if __name__ == '__main__':
    register_env(FetchPush)
    register_gripper(UltrasoundProbeGripper)    # Not able to register

    # Environment specifications
    env_id = 'FetchPush'
    options = {}
    options['robots'] = 'UR5e'
    options['controller_configs'] = load_controller_config(default_controller='OSC_POSE')
    options['gripper_types'] = None
    options['has_renderer'] = False
    options['has_offscreen_renderer'] = False
    options['use_camera_obs'] = False
    options['use_object_obs'] = True
    options['control_freq'] = 50
    options['render_camera'] = None
    options['horizon'] = 1500
    options['reward_shaping'] = True

    # Settings
    training = True
    training_timesteps = 2e6
    num_cpu = 4
    tb_log_folder = 'ppo_fetchpush_tensorboard'
    tb_log_name = 'test'
    load_model_for_training_path = None
    save_model_folder = 'trained_models'
    save_model_filename = 'test'
    load_model_folder = save_model_folder
    load_model_filename = save_model_filename

    save_model_path = os.path.join(save_model_folder, save_model_filename)
    save_vecnormalize_path = os.path.join(save_model_folder, 'vec_normalize_' + save_model_filename + '.pkl')
    load_model_path = os.path.join(load_model_folder, load_model_filename)
    load_vecnormalize_path = os.path.join(load_model_folder, 'vec_normalize_' + load_model_filename + '.pkl')

    if training:
        env = SubprocVecEnv([make_training_env(env_id, options, i) for i in range(num_cpu)])
        env = VecNormalize(env)

        if isinstance(load_model_for_training_path, str):
            model = PPO.load(load_model_for_training_path)
            model.set_env(env)
        else:
            model = PPO('MlpPolicy', env, verbose=1, tensorboard_log=tb_log_folder)
        
        model.learn(total_timesteps=training_timesteps, tb_log_name=tb_log_name)

        model.save(save_model_path)
        env.save(save_vecnormalize_path)
    
    else:
        options['has_renderer'] = True
        env_gym = GymWrapper(suite.make(env_id, **options))
        env = DummyVecEnv([lambda : env_gym])

        model = PPO.load(load_model_path)
        env = VecNormalize.load(load_vecnormalize_path, env)

        env.training = False
        env.norm_reward = False

        obs = env.reset()
        eprew = 0

        while True:
            action, _states = model.predict(obs, deterministic=True)
            # action = env.action_space.sample()
            obs, reward, done, info = env.step(action)
            print(f'reward: {reward}')
            eprew += reward
            env_gym.render()

            if done:
                print(f'eprew: {eprew}')
                obs = env.reset()
                eprew = 0

        env.close()


