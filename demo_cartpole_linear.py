import random, numpy, math, gym
from linear_predict import *

MEMORY_CAPACITY = 10000
BATCH_SIZE      = 256
GAMMA           = 0.999
MAX_EPSILON     = 1
MIN_EPSILON     = 0.01
LAMBDA          = 0.002  # speed of decay
ETA             = 0.01
FEATURE_SIZE    = 13
ACTION_SIZE     = 2
STATE_SIZE      = 4

class brain_t:
    def __init__(self):
        self.samples = []
        self.W = [None] * FEATURE_SIZE
        for i in range(len(self.W)):
            self.W[i] = np.random.uniform(-0.0001, 0.0001, size=FEATURE_SIZE)

    def train(self, x, y):
        for i in range(ACTION_SIZE):
            points = [(self.phi(x[j]), y[j][i]) for j in range(len(x))]
            self.W[i] = stochastic_gradient_descent(G, points, FEATURE_SIZE, self.W[i], ETA, 100)

    def phi(self, px):
        s, ds, a, da = px
        return np.array([s, ds, a, da, s * a, ds * da, s * da, ds * a, s**2, ds**2, a**2, da**2, 1.0])

    def predict(self, states):
        return [self.predict_single(s) for s in states]

    def predict_single(self, s):
        W, phi = self.W, self.phi
        return [W[j].dot(phi(s)) for j in range(ACTION_SIZE)]

    # stored as ( s, a, r, s_ )
    def add_cache(self, sample):
        self.samples.append(sample)
        if len(self.samples) > MEMORY_CAPACITY:
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

    def stop_training(self):
        self.epsilon = 0

    def act(self, s):
        if random.random() < self.epsilon:
            return random.randint(0, 1)
        else:
            return numpy.argmax(self.brain.predict_single(s))

    def observe(self, sample):
        if self.epsilon == 0:
            return
        self.brain.add_cache(sample)
        self.steps += 1
        self.epsilon = MIN_EPSILON + (MAX_EPSILON - MIN_EPSILON) * math.exp(-LAMBDA * self.steps)

        brain = self.brain
        batch = self.brain.sample_cache(BATCH_SIZE) # get random samples from cache (up to 128)
        batch_len = len(batch)                       # get number of random samples

        no_state = numpy.zeros(STATE_SIZE)
        states = numpy.array([o[0] for o in batch])
        states_ = numpy.array([no_state if o[3] is None else o[3] for o in batch])

        p = brain.predict(states)
        p_ = brain.predict(states_)

        x = numpy.zeros((batch_len, STATE_SIZE))
        y = numpy.zeros((batch_len, ACTION_SIZE))

        for i in range(batch_len):
            # s - current state, a - action we take, r - reward we get, s_ is next state
            s, a, r, s_ = batch[i]
            # this is array of Qopt(s, a), we have just 2 actions so this array consists of [Qopt(s, a0), Qopt(s, a1)]
            # so t is target that we initialize with w.phi(state)
            target = p[i]
            # now, for action 'a' we update 'target' as: Q(s, a) = Reward(s, a, s') + Discount * Vopt(s')
            if s_ is None:
                target[a] = r
            else:
                Vopt_ = numpy.amax(p_[i])
                target[a] = r + GAMMA * Vopt_
            # x are states, but we will replace it with phi(x) in brain.train(x, y)
            x[i] = s
            # y are targets
            y[i] = target

        self.brain.train(x, y)


# -------------------- ENVIRONMENT ---------------------
class Environment:
    def __init__(self, problem):
        self.problem = problem
        self.env = gym.make(problem)
        self.win_count = 0

    def run(self, agent):
        s = self.env.reset()
        R = 0

        while True:
            self.env.render()
            a = agent.act(s)
            s_, r, done, info = self.env.step(a)

            if done:  # end state
                s_ = None

            agent.observe((s, a, r, s_))

            s = s_
            R += r

            if done:
                break

        print("Total reward:", R)
        if R == 200:
            agent.stop_training()


if __name__ == '__main__':
    env = Environment('CartPole-v0')
    agent = agent_t()
    while True:
        env.run(agent)
