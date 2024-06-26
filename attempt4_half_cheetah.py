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


if __name__=='__main__':

    ITERATIONS = 10**6//4

    env = gym.make('HalfCheetah-v4')
    model = PPO('MlpPolicy', env, verbose=1)
    model.learn(total_timesteps=ITERATIONS)
    model.save('half_cheetah_model')
    env.close()

    env = gym.make('HalfCheetah-v4', render_mode='human')
    model = PPO.load('half_cheetah_model', env=env)
    print("Testing PPO Agent")
    test_agent(model, env)
    env.close()
