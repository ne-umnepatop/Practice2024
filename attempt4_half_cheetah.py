import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

# Iterations = int(1e6)
Iterations = 10**3

env = make_vec_env('HalfCheetah-v4', n_envs=1)
model = PPO('MlpPolicy', env, verbose=1)
model.learn(total_timesteps=Iterations)
model.save('half_cheetah_model')
env = make_vec_env('HalfCheetah-v4', n_envs=1)

model = PPO.load('half_cheetah_model', env=env)
obs = env.reset()
while True:
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, dones, info = env.step(action)
    env.render()
    if dones[0]:
        break
