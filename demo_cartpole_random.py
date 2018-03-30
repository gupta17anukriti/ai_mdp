import gym
import random
import numpy as np

def random_action(obs, reward, info):
    return random.randint(0, 1)

def basic_policy(obs, reward, info):
    angle = obs[2]
    return 0 if angle < 0 else 1

if __name__ == "__main__":
    num_iter = 100000
    totals = []
    for i in range(num_iter):
        env = gym.make("CartPole-v0")
        obs = env.reset()
        action = basic_policy(obs, 0, None)
        done = False
        episode_rewards = 0.0
        while not done:
            obs, reward, done, info = env.step(action)
            episode_rewards += reward
            if i % 10000 == 0:
                env.render()
            if not done:
                action = basic_policy(obs, reward, info)
        totals.append(episode_rewards)
    print(np.mean(totals), np.std(totals), np.min(totals), np.max(totals))
    # 42   8 24  72