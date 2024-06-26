import gymnasium as gym
from stable_baselines3 import PPO
# from stable_baselines3.common.env_util import make_vec_env


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


# ITERATIONS = int(1e6)
ITERATIONS = 10**3

env = gym.make('HalfCheetah-v4')
model = PPO('MlpPolicy', env, verbose=1)
model.learn(total_timesteps=ITERATIONS)
model.save('half_cheetah_model')
# model = PPO.load('half_cheetah_model', env=env)        
env.close()

# env = gym.make('HalfCheetah-v4', render_mode='human')
# # Тестирование случайного агента
# print("Testing Random Agent")
# test_agent(None, env)
# env.close()

env = gym.make('HalfCheetah-v4', render_mode='human')
# Тестирование обученного агента
print("Testing PPO Agent")
test_agent(model, env)
env.close()
