import gym
import random
from mdp import *
from mdp_aprox_egreedy import *
import numpy as np

class carpole_egreedy_t(mdp_t):
    def __init__(self):
        self.env = gym.make("CartPole-v0")
        self.current = None
        self.iter = 0
        self.episode_reward = None
        self.episode_rewards = []

    def create_state(self, obs, done):
        # x - position, velocity, angle, angular velocity
        return (round(obs[0], 1), round(obs[1], 2), round(obs[2], 2), round(obs[3], 2)), done

    def start_state(self):
        if self.episode_reward is not None:
            self.episode_rewards.append(self.episode_reward)
            print("Total reward: " + str(self.episode_reward))
        self.episode_reward = 0.0
        self.iter += 1
        self.env.reset()
        self.current = self.create_state(self.env.reset(), False)
        return self.current

    def actions(self, s):
        return [0, 1]

    def is_end(self, s):
        return s[1]

    def transition(self, s, a):
        assert(s == self.current)
        obs, reward, done, info = self.env.step(a)
        if self.iter % 100 == 0:
            self.env.render()
        self.current = self.create_state(obs, done)
        self.episode_reward += reward
        return reward, self.current

    def discount(self):
        return 1


if __name__ == "__main__":
    num_iter = 1000
    m = carpole_egreedy_t()
    qs = qval_qlearn_epsilon_greedy(m, 100000)
    totals = m.episode_rewards
    print((np.mean(totals), np.std(totals), np.min(totals), np.max(totals)))
    # for 100000: 70, 46  8 200

