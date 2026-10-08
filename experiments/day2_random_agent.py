import gymnasium as gym

# 1. Create the environment (the world)
env = gym.make("CartPole-v1")

# 2. Look at the "shape" of the world
print("Action space     :", env.action_space)        # what the agent can do
print("Observation space:", env.observation_space)   # what the agent sees

# 3. Start a new episode: reset() gives the first state
state, info = env.reset(seed=0)
print("\nFirst state:", state)

# 4. Play ONE episode with a RANDOM agent
total_reward = 0
steps = 0
done = False

while not done:
    action = env.action_space.sample()      # random choice: 0 = left, 1 = right
    state, reward, terminated, truncated, info = env.step(action)

    total_reward += reward
    steps += 1
    done = terminated or truncated          # the episode is over

    if steps <= 5:                          # print only the first 5 steps
        print(f"Step {steps}: action={action}, reward={reward}, state={state}")

print(f"\nEpisode finished after {steps} steps. Total reward = {total_reward}")

# 5. Play 20 episodes and see the average
scores = []
for episode in range(20):
    state, info = env.reset()
    total, done = 0, False
    while not done:
        state, reward, terminated, truncated, info = env.step(env.action_space.sample())
        total += reward
        done = terminated or truncated
    scores.append(total)

print("Average total reward of the random agent:", sum(scores) / len(scores))
env.close()