import gymnasium as gym
from stable_baselines3 import PPO


def test_agent(agent, env, episodes=5):
    """
    Получение баллов за прохождение моделью среды
    """
    for episode in range(episodes):
        obs, _ = env.reset()
        total_reward = 0
        terminated = False
        truncated = False
        while  not(terminated or truncated):
            if agent:
                action, _ = agent.predict(obs)
            else:
                action = env.action_space.sample()
            result = env.step(action)
            obs, reward, terminated, truncated, _ = result
            total_reward += reward
            env.render()
        print(f"Episode {episode + 1}: Total Reward = {total_reward}")

        
def demo_model(environment, name_of_model, episodes = 5):
    """
    Демонстрация повдения обученной модели
    """
    env = gym.make(environment, render_mode='human')
    model = PPO.load(name_of_model, env=env)
    print("Testing PPO Agent")
    test_agent(model, env, episodes)
    env.close()

if __name__=='__main__':

    demo_model('HalfCheetah-v4', 'half_cheetah_model_PC')
