import gymnasium as gym
from stable_baselines3 import PPO

env = gym.make("CartPole-v1")

# TRAIN: the agent plays thousands of steps and learns from the rewards
model = PPO("MlpPolicy", env, verbose=0)
model.learn(total_timesteps=50_000)

# TEST: play 20 episodes using what it learned
scores = []
for episode in range(20):
    state, info = env.reset()
    total, done = 0, False
    while not done:
        action, _ = model.predict(state, deterministic=True)   # the brain decides
        state, reward, terminated, truncated, info = env.step(int(action))
        total += reward
        done = terminated or truncated
    scores.append(total)

print("Average total reward of the TRAINED agent:", sum(scores) / len(scores))
env.close()