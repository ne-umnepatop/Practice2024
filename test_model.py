import gymnasium as gym
import torch
from stable_baselines3 import PPO
from imitation_learning import MLP, OBS_SIZE, ACTION_SIZE, NERONS


def test_agent(agent, env, episodes=5, mod='ppo'):
    """
    Получение баллов за прохождение моделью среды
    mod = ppo или imit для подкрепления или имитации
    """
    for episode in range(episodes):
        obs, _ = env.reset()
        total_reward = 0
        terminated = False
        truncated = False
        while  not(terminated or truncated):
            if agent:
                if mod=='ppo':
                    action, _ = agent.predict(obs)
                elif mod=='imit':
                    obs = torch.tensor(obs, dtype=torch.float32)
                    action = agent(obs).detach().numpy()
                else:
                    print('Нет такой модели')
                    action = env.action_space.sample()
            else:
                action = env.action_space.sample()
            result = env.step(action)
            obs, reward, terminated, truncated, info = result
            total_reward += reward
            env.render()
        print(f"Episode {episode + 1}: Total Reward = {total_reward}")


def demo_model(environment, name_of_model, episodes = 5, mod='ppo'):
    """
    Демонстрация повдения обученной модели
    """
    env = gym.make(environment, render_mode='human')
    if mod=='ppo':
        model = PPO.load(name_of_model, env=env)
        print("Testing PPO Agent")
    elif mod=='imit':
        model = MLP(OBS_SIZE, ACTION_SIZE, NERONS)
        model.load_state_dict(torch.load(name_of_model))
        model.eval()
        print("Testing Imitation Model")
    else: model = None
    test_agent(model, env, episodes, mod)
    env.close()

if __name__=='__main__':

    demo_model('HalfCheetah-v4', 'half_cheetah_model', mod='ppo')
    # demo_model('HalfCheetah-v4', 'half_cheetah_model_PC', mod='ppo')
    # demo_model('HalfCheetah-v4', 'half_cheetah_im.pth', mod='imit')
