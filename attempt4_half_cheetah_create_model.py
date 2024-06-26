import gymnasium as gym
from stable_baselines3 import PPO


def create_model(ENVIRONMENT, ITERATIONS, NAME_OF_MODEL):
    """
    Обучение и сохранение модели
    """
    env = gym.make(ENVIRONMENT)
    model = PPO('MlpPolicy', env, verbose=1)
    model.learn(total_timesteps=ITERATIONS)
    model.save(NAME_OF_MODEL)
    env.close()


if __name__=='__main__':

    create_model('HalfCheetah-v4', 10**4//4, 'half_cheetah_model')
