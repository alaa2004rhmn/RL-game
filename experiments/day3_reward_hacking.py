import gymnasium as gym
from stable_baselines3 import PPO


class PunishEveryStep(gym.RewardWrapper):
    """Changes the reward: -1 at every step (instead of +1)."""
    def reward(self, reward):
        return -1.0


def train_and_test(env, name):
    model = PPO("MlpPolicy", env, verbose=0)
    model.learn(total_timesteps=30_000)

    lengths = []
    for _ in range(20):
        state, info = env.reset()
        steps, done = 0, False
        while not done:
            action, _ = model.predict(state, deterministic=True)
            state, reward, terminated, truncated, info = env.step(int(action))
            steps += 1
            done = terminated or truncated
        lengths.append(steps)

    print(f"{name}: average episode length = {sum(lengths) / len(lengths):.1f} steps")


# Agent 1: the good reward (+1 per step)
train_and_test(gym.make("CartPole-v1"), "GOOD reward (+1 per step)")

# Agent 2: the bad reward (-1 per step)
train_and_test(PunishEveryStep(gym.make("CartPole-v1")), "BAD reward (-1 per step)")