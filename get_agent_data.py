import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO


if __name__=='__main__':

    ITERATIONS = 10**4
    DEMO_EPISODES = 5

    env = gym.make('HalfCheetah-v4')
    model = PPO.load('half_cheetah_model', env=env)

    demo_states = []
    demo_actions = []
    for _ in range(DEMO_EPISODES):
        obs, _ = env.reset()
        terminated, truncated = False, False
        while  not(terminated or truncated):
            action, _ = model.predict(obs)
            res = env.step(action)
            next_obs, reward, terminated, truncated, info = res
            demo_states.append(obs)
            demo_actions.append(action)
            obs = next_obs
    env.close()

    out = np.transpose(np.column_stack((demo_states, demo_actions)))
    np.savetxt('HalfCheetah.csv', out, delimiter=',')
