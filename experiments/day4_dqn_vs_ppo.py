import gymnasium as gym
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.evaluation import evaluate_policy

STEPS = 50_000

for name, Algo in [("PPO", PPO), ("DQN", DQN)]:
    env = gym.make("CartPole-v1")
    model = Algo("MlpPolicy", env, verbose=0, seed=0)
    model.learn(total_timesteps=STEPS)

    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=20)
    print(f"{name}: mean total reward = {mean_reward:.1f} (± {std_reward:.1f})")
    env.close()
    
    #Do not decide "which is better" from this. Results change with the seed, the number of steps, and the settings. 
    # DQN with default settings is often weaker on CartPole in short training. 
    # This is a lesson: an algorithm's result depends a lot on hyperparameters. In our project we will test properly with several runs.