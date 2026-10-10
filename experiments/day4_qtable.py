import gymnasium as gym
import numpy as np

env = gym.make("FrozenLake-v1", is_slippery=False)

n_states = env.observation_space.n      # 16 squares
n_actions = env.action_space.n          # 4 moves

# The brain: a table with one row per state and one column per action
Q = np.zeros((n_states, n_actions))

alpha = 0.1       # learning speed
gamma = 0.99      # care about the future
epsilon = 1.0     # start by exploring everything
rng = np.random.default_rng(0)

for episode in range(3000):
    state, _ = env.reset()
    done = False
    while not done:
        # Explore or exploit?
        if rng.random() < epsilon:
            action = env.action_space.sample()       # explore
        else:
            action = int(np.argmax(Q[state]))        # exploit

        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        # THE LEARNING RULE
        target = reward + gamma * np.max(Q[next_state]) * (not terminated)
        Q[state, action] += alpha * (target - Q[state, action])

        state = next_state

    epsilon = max(0.05, epsilon * 0.999)             # explore a bit less each episode

# ---- Look at what the agent learned ----
arrows = {0: "←", 1: "↓", 2: "→", 3: "↑"}
print("Learned policy (best action in each square):")
for row in range(4):
    print(" ".join(arrows[int(np.argmax(Q[row * 4 + col]))] for col in range(4)))

# ---- Test it ----
wins = 0
for _ in range(100):
    state, _ = env.reset()
    done = False
    while not done:
        state, reward, terminated, truncated, _ = env.step(int(np.argmax(Q[state])))
        done = terminated or truncated
    wins += reward
print("\nSuccess rate over 100 games:", wins, "%")