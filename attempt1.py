import gymnasium as gym
import numpy as np

# Создание среды
env = gym.make('InvertedPendulum-v4')

# Функция для сбора данных
def collect_expert_data(env, num_episodes=100, max_steps=1000):
    states = []
    actions = []

    for episode in range(num_episodes):
        state, _ = env.reset()
        done = False
        episode_states = []
        episode_actions = []

        for step in range(max_steps):
            action = env.action_space.sample()  # Здесь вместо random action можно использовать обученного агента
            next_state, reward, done, _, _ = env.step(action)
            episode_states.append(state)
            episode_actions.append(action)
            state = next_state

            if done:
                break

        states.extend(episode_states)
        actions.extend(episode_actions)

    return np.array(states), np.array(actions)

# Сбор данных
states, actions = collect_expert_data(env, num_episodes=100, max_steps=1000)

# Сохранение данных
np.save('inverted_pendulum_states.npy', states)
np.save('inverted_pendulum_actions.npy', actions)
