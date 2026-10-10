import gymnasium as gym

GAMMA = 0.99

env = gym.make("CartPole-v1")

print("=== The MDP of CartPole ===")
print("State  (what it sees):", env.observation_space)
print("Action (what it does):", env.action_space)
print("Gamma  (future care) :", GAMMA)

state, info = env.reset(seed=1)
rewards = []
done = False

while not done:
    action = env.action_space.sample()
    state, reward, terminated, truncated, info = env.step(action)
    rewards.append(reward)
    done = terminated or truncated

# Total reward: simple sum (this is what we plot in TensorBoard)
total_reward = sum(rewards)

# Discounted return: each step in the future is multiplied by gamma again
discounted_return = sum((GAMMA ** t) * r for t, r in enumerate(rewards))

print("\nEpisode length        :", len(rewards))
print("Total reward          :", total_reward)
print("Discounted return     :", round(discounted_return, 3))
env.close()