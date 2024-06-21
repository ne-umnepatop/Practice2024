import gymnasium as gym
import numpy as np
from control import lqr, pid

env = gym.make('InvertedPendulum-v4')


def collect_expert_data(env, episodes=100, steps=1000):
    states = []
    actions = []

    for _ in range(episodes):
        state, _ = env.reset()
        done = False
        episode_states = []
        episode_actions = []

        for _ in range(steps):
            # action = env.action_space.sample()
            # action = pid(state)
            action = [0]
            next_state, _, done, _, _ = env.step(action)
            episode_states.append(state)
            episode_actions.append(action)
            state = next_state

            if done:
                break

        states.extend(episode_states)
        actions.extend(episode_actions)

    return np.array(states), np.array(actions)

out = np.transpose(np.column_stack(collect_expert_data(env, episodes=100, steps=1000)))
np.savetxt('inverted_pendulum.csv', out, delimiter=',')
