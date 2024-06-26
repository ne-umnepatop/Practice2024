import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from attempt4_half_cheetah import test_agent


if __name__=='__main__':

    ITERATIONS = 10**3
    DEMO_EPISODES = 5  # Number of demonstration episodes to collect

    env = gym.make('HalfCheetah-v4')
    model = PPO.load('half_cheetah_model', env=env)

    demo_states = []
    demo_actions = []
    for _ in range(DEMO_EPISODES):
        obs, _ = env.reset()
        # reward_tot = 0
        terminated = False
        truncated = False
        while  not(terminated or truncated):
            action, _ = model.predict(obs)
            res = env.step(action)
            # print(res)
            next_obs, reward, terminated, truncated, info = res
            # print(info)
            demo_states.append(obs)
            # print(obs)
            # print(action)
            # reward_tot += reward
            demo_actions.append(action)
            obs = next_obs
    env.close()

    # print(demo_states)
    # print(demo_actions)
    out = np.transpose(np.column_stack((demo_states, demo_actions)))
    np.savetxt('HalfCheetah.csv', out, delimiter=',')
