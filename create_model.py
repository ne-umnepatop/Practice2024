import gymnasium as gym
from stable_baselines3 import PPO
import numpy as np


def reward_fn(obs, action, next_obs, reward, done, info):
    forward_vel = info['forward_velocity']
    action_norm = np.linalg.norm(action)
    body_orientation = info['body_orientation']
    reward = forward_vel - 0.1 * action_norm
    if body_orientation < 0: reward -= 10
    return reward

def create_model(environment, iterations, name_of_model):
    """
    Обучение и сохранение модели
    """
    env = gym.make(environment)
    model = PPO('MlpPolicy', env, verbose=1, reward_function=reward_fn)
    model.learn(total_timesteps=iterations)
    model.save(name_of_model)
    env.close()


if __name__=='__main__':

    create_model('HalfCheetah-v4', 10**4//4, 'half_cheetah_model')
