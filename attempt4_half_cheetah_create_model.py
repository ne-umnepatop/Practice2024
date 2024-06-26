import gymnasium as gym
from stable_baselines3 import PPO


def learn(ITERATIONS, name_of_model):
    """
    Обучение и сохранение модели
    """
    env = gym.make('HalfCheetah-v4')
    model = PPO('MlpPolicy', env, verbose=1)
    model.learn(total_timesteps=ITERATIONS)
    model.save(name_of_model)
    env.close()


if __name__=='__main__':

    learn(10**4//4, 'half_cheetah_model')
