
import gymnasium as gym
from stable_baselines3 import PPO
import numpy as np

ITERATIONS = 10**3
env = gym.make('CartPole-v1')
model = PPO('MlpPolicy', env, verbose=1)
model.learn(total_timesteps=ITERATIONS)
model.save("ppo_cartpole")


def test_agent(agent, env, episodes=5):
    for episode in range(episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0
        while not done:
            if agent:
                action, _ = agent.predict(obs)
            else:
                action = env.action_space.sample()
            result = env.step(action)
            # print(result)
            obs, reward, done, info = result[:4]
            total_reward += reward
            env.render()
        print(f"Episode {episode + 1}: Total Reward = {total_reward}")

        
env.close()

env = gym.make('CartPole-v1', render_mode='human')
# Тестирование случайного агента
print("Testing Random Agent")
test_agent(None, env)
env.close()

env = gym.make('CartPole-v1', render_mode='human')
# Тестирование обученного агента
print("Testing PPO Agent")
test_agent(model, env)

env.close()
