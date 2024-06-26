import gymnasium as gym
from stable_baselines3 import PPO


def create_model(environment, iterations, name_of_model):
    """
    Обучение и сохранение модели
    """
    env = gym.make(environment)
    model = PPO('MlpPolicy', env, verbose=1)
    model.learn(total_timesteps=iterations)
    model.save(name_of_model)
    env.close()


if __name__=='__main__':

    create_model('HalfCheetah-v4', 10**4//4, 'half_cheetah_model')
