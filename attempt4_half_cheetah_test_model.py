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
            # print(result)
            obs, reward, terminated, truncated, info = result
            total_reward += reward
            env.render()
        print(f"Episode {episode + 1}: Total Reward = {total_reward}")

def demo_model(ENVIRONMENT, NAME_OF_MODEL, EPISODES = 5):
    """
    Демонстрация повдения обученной модели
    """
    env = gym.make(ENVIRONMENT, render_mode='human')
    model = PPO.load(NAME_OF_MODEL, env=env)
    print("Testing PPO Agent")
    test_agent(model, env, EPISODES)
    env.close()

if __name__=='__main__':

    demo_model('HalfCheetah-v4', 'half_cheetah_model_PC')
