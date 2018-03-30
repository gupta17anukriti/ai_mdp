import random, numpy, math, gym
from linear_predict import *

class brain_t:
    def __init__(self):
        self.eta = 0.01
        self.FEATURE_SIZE = 13
        self.ACTION_SIZE = 2
        self.samples = []
        self.W = [None] * self.FEATURE_SIZE
        for i in range(len(self.W)):
            self.W[i] = np.random.uniform(-0.0001, 0.0001, size = self.FEATURE_SIZE)

    def train(self, x, y):
        self.eta *= 0.999
        for i in range(self.ACTION_SIZE):
            points = [(self.phi(x[j]), y[j][i]) for j in range(len(x))]
            self.W[i] = stochastic_gradient_descent(sG, points, self.FEATURE_SIZE, self.W[i], self.eta, 100)

    def phi(self, px):
        s, ds, a, da = px
        return np.array([s, ds, a, da, s * a, ds * da, s * da, ds * a, s**2, ds**2, a**2, da**2, 1.0])

    def predict(self, states):
        return [self.predict_single(s) for s in states]

    def predict_single(self, s):
        W, phi = self.W, self.phi
        return [W[j].dot(phi(s)) for j in range(self.ACTION_SIZE)]

    # stored as ( s, a, r, s_ )
    def add_cache(self, sample):
        self.samples.append(sample)

        if len(self.samples) > 10000: # keep 10000 max states in cache
            self.samples.pop(0)

    # get array of ( s, a, r, s_ )
    def sample_cache(self, n):
        n = min(n, len(self.samples))
        return random.sample(self.samples, n)


class agent_t:
    steps = 0
    epsilon = 1.0

    def __init__(self):
        self.brain = brain_t()
        self.STATE_SIZE = 4
        self.last_sample = None

    def act(self, s):
        if random.random() < self.epsilon:
            return random.randint(0, 1)
        else:
            return numpy.argmax(self.brain.predict_single(s))

    def observe(self, sample):
        self.brain.add_cache(sample)
        self.steps += 1
        # slowly decrease Epsilon based on our eperience
        self.epsilon = 0.01 + (1 - 0.01) * math.exp(-0.001 * self.steps)

    def replay(self):
        brain = self.brain
        batch = self.brain.sample_cache(128)       # get random samples from cache (up to 128)
        batchLen = len(batch)                      # get number of random samples

        no_state = numpy.zeros(self.STATE_SIZE)    # number of states

        states = numpy.array([o[0] for o in batch])
        states_ = numpy.array([no_state if o[3] is None else o[3] for o in batch])

        p = brain.predict(states)
        p_ = brain.predict(states_)

        x = numpy.zeros((batchLen, self.STATE_SIZE))
        y = numpy.zeros((batchLen, brain.ACTION_SIZE))

        for i in range(batchLen):
            s, a, r, s_ = batch[i]

            t = p[i]
            if s_ is None:
                t[a] = r
            else:
                t[a] = r + 0.001 * numpy.amax(p_[i])

            x[i] = s
            y[i] = t

        self.brain.train(x, y)


# -------------------- ENVIRONMENT ---------------------
class Environment:
    def __init__(self, problem):
        self.problem = problem
        self.env = gym.make(problem)

    def run(self, agent):
        s = self.env.reset()
        R = 0

        while True:
            self.env.render()

            a = agent.act(s)

            s_, r, done, info = self.env.step(a)

            if done:  # terminal state
                s_ = None

            agent.observe((s, a, r, s_))
            agent.replay()

            s = s_
            R += r

            if done:
                break

        print("Total reward:", R)
        if R > 150:
            print(agent.brain.W)


# -------------------- MAIN ----------------------------
PROBLEM = 'CartPole-v0'
env = Environment(PROBLEM)

stateCnt = env.env.observation_space.shape[0]
actionCnt = env.env.action_space.n

agent = agent_t()

try:
    while True:
        env.run(agent)
finally:
    pass
    # agent.brain.model.save("cartpole-basic.h5")