import mdp
import random
from collections import *

class montecarlo_mdp_t(object):
    def __init__(self, start, states, discount):
        self.states = states
        self.start = start
        self.state_actions = {}
        self.disc = discount
        self.state_index = [None] * len(states)
        for s, so in states.items():
            self.state_actions[so.id] = [k for k in so.actions.keys()]
            self.state_index[so.index] = s
    def start_state(self):
        return self.start
    def actions(self, s):
        return self.state_actions[s]
    def is_end(self, s):
        return self.states[s].is_end
    def transitions(self, s, a):
        return self.states[s].actions[a]
    def discount(self):
        return self.disc
    def size(self):
        return len(self.states)
    def get_state(self, id):
        return self.state_index[id]

class mdp_node_t(object):
    def __init__(self, id, index, isend):
        self.id = id
        self.index = index
        self.is_end = isend
        self.actions = defaultdict() # {action: [(probability, reward, next_state)]}

# estimate rewards and probabilities based on provided episodes and return 'estimated' mdp
# this is type of monte-carlo model approximation
# episode: s1,a1,r1,s2;a2,r2,s3;a3,r3,s4;
def approximate_model(episodes, start, discount):
    sa = defaultdict(int)    # {(state, action): count}
    sas = defaultdict()      # {(state, action): set of next states}
    san = defaultdict(int)   # {(state, action, next_state): count}
    rew = defaultdict(float) # {state, action, next_state): reward}}
    p = defaultdict(float)   # {state, action, next_state): probability}}
    es = set()               # set of end states
    for e in episodes:
        es.add(e[len(e) - 1])
        for i in range(1, len(e), 3):
            s, a, r, n = e[i - 1], e[i], e[i + 1], e[i + 2]
            rew[(s, a, n)] += r
            sa[(s, a)] += 1
            san[(s, a, n)] += 1
            if not (s, a) in sas:
                sas[(s, a)] = set()
            sas[(s, a)].add(n)
    # calculate probability and reward for each (s, a, n) tuple
    for k, r in rew.items():
        rew[k] = r / san[k]
        p[k] = san[k] / sa[(k[0], k[1])]
    # create list of all states and their actions
    states, i = defaultdict(), 0
    for k, na in sas.items(): # 'k' is (state, action) 'na' is set of next_states for 'k'
        s, a = k
        if not s in states:
            states[s] = mdp_node_t(s, i, s in es)
            i += 1
        nd = states[s]
        for n in na:
            if not n in states:
                states[n] = mdp_node_t(n, i, n in es)
                i += 1
            if not a in nd.actions:
                nd.actions[a] = []
            nd.actions[a].append((p[(s, a, n)], rew[(s, a, n)], n))
    m = montecarlo_mdp_t(start, states, discount)
    return m

# approximate Q-values by ising on-policy model-free Monte Carlo
# returns dictionary of {(state, action): Qp(state, action)}
def qvalues_monte_carlo(episodes, discount, eta = None):
    Q = defaultdict(float)
    N = defaultdict(int)
    for e in episodes:
        u = 0.0 # current utility
        for i in range(len(e) - 4, 0, -3):
            s, a, r, sn = e[i], e[i + 1], e[i + 2], e[i + 3]
            prediction = Q[(s, a)]  # prediction
            target = r + discount * u
            u = target
            _eta = eta
            if _eta is None:
                _eta = 1.0 / (1.0 + N[(s, a)])
                N[(s, a)] += 1
            Q[(s, a)] = (1 - _eta) * prediction + _eta * target
    return Q

def qvalues_sarsa(episodes, discount, eta = None):
    Q, N = defaultdict(float), defaultdict(int)
    def Qp(s, a):
        return 0 if a is None or (s, a) not in Q else Q[(s, a)]
    for e in episodes:
        an = None
        for i in range(len(e) - 4, 0, -3):
            s, a, r, sn = e[i], e[i + 1], e[i + 2], e[i + 3]
            _eta = eta
            if _eta is None:
                _eta = 1.0 / (1.0 + N[(s, a)])
                N[(s, a)] += 1
            prediction = Qp(s, a)
            target = r + discount * Qp(sn, an)
            Q[(s, a)] = (1 - _eta) * prediction + _eta * target
    return Q

def qvalues_qlearning_ex(episodes, discount, eta = None):
    Qopt = defaultdict(float)
    Vopt = defaultdict(float)
    Popt = defaultdict()
    N    = defaultdict(int)
    for e in episodes:
        for i in range(len(e) - 4, 0, -3):
            s, a, r, sn = e[i], e[i + 1], e[i + 2], e[i + 3]
            _eta = eta
            if _eta is None:
                _eta = 1.0 / (1.0 + N[(s, a)])
                N[(s, a)] += 1
            prediction = Qopt[(s, a)]
            target = r + discount * Vopt[sn]
            Qopt[(s, a)] = (1.0 - _eta) * prediction + _eta * target
            if s not in Vopt or Qopt[s, a] > Vopt[s]:
                Vopt[s] = Qopt[s, a]
                Popt[s] = a
    return Qopt, Vopt, Popt

def qvalues_qlearning(episodes, discount, eta = None):
    return qvalues_qlearning_ex(episodes, discount, eta)[0]

def q_values_opt_policy(qs):
    SA = defaultdict()        # optimal actions for each state
    SAQ = defaultdict(float)  # Q values for optimal actions for each state
    for sa, q in qs.items():
        s, a = sa
        if s not in SA or q > SAQ[s]:
            SA[s] = a
            SAQ[s] = q
    return SA
