import gymnasium as gym
from stable_baselines3 import PPO
import numpy as np


def reward_fn(obs, action, next_obs, reward, done, info):
    forward_vel = info['forward_velocity']
    action_norm = np.linalg.norm(action)
    vel = info['x_velocity']
    reward = forward_vel - 0.8 * action_norm
    if vel < 0: reward -= 200
    # elif vel > 0: reward += 10
    return reward


class MyPPO(PPO):
    def compute_reward(self, obs, action, next_obs, reward, done, info):
        return reward_fn(obs, action, next_obs, reward, done, info)


def create_model(environment, iterations, name_of_model):
    """
    Обучение и сохранение модели
    """
    env = gym.make(environment)
    model = MyPPO('MlpPolicy', env, verbose=1)
    model.learn(total_timesteps=iterations)
    model.save(name_of_model)
    env.close()


if __name__=='__main__':

    create_model('HalfCheetah-v4', 10**5, 'half_cheetah_model')
