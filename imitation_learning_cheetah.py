import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO

# from attempt4_half_cheetah import test_agent


ITERATIONS = 10**3

env = gym.make('HalfCheetah-v4', render_mode='ansi')
model = PPO.load('half_cheetah_model', env=env)

demo_states = []
demo_actions = []
for _ in range(DEMO_EPISODES):
    obs, _ = env.reset()
    reward_tot=0
    while reward_tot<500:
        action, _ = model.predict(obs)
        next_obs, reward, done, info = env.step(action)
        print(info)
        demo_states.append(obs)
        print(obs)
        print(action)
        reward_tot+=reward[0]
        demo_actions.append(action)
        obs = next_obs
env.close()

print(demo_states)
print(demo_actions)
out = np.transpose(np.column_stack(demo_states, demo_actions))
np.savetxt('HalfCheetah.csv', out, delimiter=',')

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
# bc_model = BCModel('MlpPolicy', env, demonstration_states=demo_states, demonstration_actions=demo_actions, verbose=1)
# bc_model.learn(total_timesteps=ITERATIONS)
# bc_model.save('half_cheetah_imitation_model')

# env = gym.make('HalfCheetah-v4', render_mode='human')
# print("Testing Behavioral Cloning Agent")
# test_agent(bc_model, env)
env.close()

