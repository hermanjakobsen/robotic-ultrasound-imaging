import robosuite as suite
import os
import yaml

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
from stable_baselines3.common.callbacks import EvalCallback

from my_models.grippers import UltrasoundProbeGripper
from my_environments import Ultrasound, FetchPush
from utils.common import register_gripper

def make_training_env(env_id, options, rank, seed=0):
    """
    Utility function for multiprocessed env.

    :param env_id: (str) the environment ID
    :param options: (dict) additional arguments to pass to the specific environment class initializer
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    """
    def _init():
        register_gripper(UltrasoundProbeGripper)
        env = GymWrapper(suite.make(env_id, **options))
        env = Monitor(env)
        env.seed(seed + rank)
        return env
    set_random_seed(seed)
    return _init


if __name__ == '__main__':
    register_env(FetchPush)

    with open("rl_config.yaml", 'r') as stream:
        config = yaml.safe_load(stream)

    # Environment specifications
    env_options = config["robosuite"]
    env_id = env_options.pop("env_id")

    # Settings for stable-baselines RL algorithm
    sb_config = config["sb_config"]
    training_timesteps = sb_config["total_timesteps"]
    num_cpu = sb_config["num_cpu"]

    # Settings for stable-baselines policy
    policy_kwargs = config["sb_policy"]
    policy_type = policy_kwargs.pop("type")

    # Settings used for file handling and logging (save/load destination etc)
    file_handling = config["file_handling"]

    tb_log_folder = file_handling["tb_log_folder"]
    tb_log_name = file_handling["tb_log_name"]

    save_model_folder = file_handling["save_model_folder"]
    save_model_filename = file_handling["save_model_filename"]
    load_model_folder = file_handling["load_model_folder"]
    load_model_filename = file_handling["load_model_filename"]

    load_model_for_training_path = file_handling["load_model_for_training_path"]
    load_vecnormalize_for_training_path = file_handling["load_vecnormalize_for_training_path"]

    # Join paths
    save_model_path = os.path.join(save_model_folder, save_model_filename)
    save_vecnormalize_path = os.path.join(save_model_folder, 'vec_normalize_' + save_model_filename + '.pkl')
    load_model_path = os.path.join(load_model_folder, load_model_filename)
    load_vecnormalize_path = os.path.join(load_model_folder, 'vec_normalize_' + load_model_filename + '.pkl')

    # Settings for pipeline
    training = config["training"]
    seed = config["seed"]

    # RL pipeline
    if training:
        env = SubprocVecEnv([make_training_env(env_id, env_options, i, seed) for i in range(num_cpu)])
        env = VecNormalize(env)

        # Check if should continue training on a model
        if isinstance(load_model_for_training_path, str):
            env = VecNormalize.load(load_vecnormalize_for_training_path, env)
            model = PPO.load(load_model_for_training_path, env=env)
        else:
            model = PPO(policy_type, env, policy_kwargs=policy_kwargs, tensorboard_log=tb_log_folder, verbose=1)


        # Evaluation during training (save best model)
        eval_env_func = make_training_env(env_id, env_options, num_cpu, seed)
        eval_env = DummyVecEnv([eval_env_func])
        eval_env = VecNormalize(eval_env)

        eval_callback = EvalCallback(eval_env, best_model_save_path='./best_models/',
                             log_path='./logs_best_model/',
                             deterministic=True, render=False, n_eval_episodes=10)

        # Training
        model.learn(total_timesteps=training_timesteps, tb_log_name=tb_log_name, callback=eval_callback)

        # Save trained model
        model.save(save_model_path)
        env.save(save_vecnormalize_path)

    else:
        env_options['has_renderer'] = True
        register_gripper(UltrasoundProbeGripper)
        env_gym = GymWrapper(suite.make(env_id, **env_options))
        env = DummyVecEnv([lambda : env_gym])

        model = PPO.load(load_model_path)
        env = VecNormalize.load(load_vecnormalize_path, env)

        env.training = False
        env.norm_reward = False

        obs = env.reset()
        eprew = 0
        while True:
            action, _states = model.predict(obs, deterministic=True)
            #action = env.action_space.sample()
            obs, reward, done, info = env.step(action)
            #print(action)
            print(f'reward: {reward}')
            eprew += reward
            env_gym.render()
            if done:
                print(f'eprew: {eprew}')
                obs = env.reset()
                eprew = 0

        env.close()


