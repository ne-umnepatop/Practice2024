import gymnasium as gym
from stable_baselines3 import PPO
from attempt4_half_cheetah import test_agent

def demo_model():
    """
    Демонстрация повдения обученной модели
    """
    env = gym.make('HalfCheetah-v4', render_mode='human')
    model = PPO.load('half_cheetah_model_PC', env=env)
    print("Testing PPO Agent")
    test_agent(model, env)
    env.close()

if __name__=='__main__':
    demo_model()
