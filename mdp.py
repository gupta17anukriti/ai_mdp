import sys
from random import *

class mdp_t(object):
    def start_state(self):
        return self.get_state(0)
    def actions(self, s): pass              # return array of possible actions
    def is_end(self, s): pass               # is this end state?
    def transitions(self, s, a): pass       # return array of (probability, reward, next state) tuples
    def transition(self, s, a):
        ta = self.transitions(s, a)
        p, r, t = 0.0, uniform(0, 1), None
        for i in range(len(ta)):
            t, p = ta[i], p + ta[i][0]
            if r <= p:
                break
        return t[1], t[2]                   # returns (reward, next_state)
    def discount(self): return 1.0
    # size and get_state methods are used only for policy iterator and value iterator algorithms (which are not episode based)
    def size(self): return -1               # return number of states in MDP (returning -1 for unknown). this is used for
    def get_state(self, id): return id      # return state for passed state index

class policy_t(object):                 # abstract policy (i.e. what action we select for each state of MDP)
    def __init__(self, mdp):
        self.mdp = mdp
    def action(self, s):                # returns policy for passed state
        pass
    def create_episode(self, s):        # generate episode [state, action, reward, state, .... ], 3n+1 states (n >= 0)
        e, m = [s], self.mdp
        while not m.is_end(s):
            a = self.action(s)
            t = m.transition(s, a)
            e += [a, t[0], t[1]]
            s = t[1]
        return e

class random_policy_t(policy_t):
    def action(self, s):
        aa = self.mdp.actions(s)
        return aa[randint(0, len(aa) - 1)]

class dict_policy_t(policy_t):         # policy from dictionary {state: action}
    def __init__(self, mdp, d):
        self.mdp = mdp
        self.d = d
    def action(self, s):
        return self.d[s]

def episode_utility(m, e):             # return total reward (utility) of episode array
    r, d = 0, 1.0
    for i in range(1, len(e) - 1, 3):
        r += e[i + 1] * d
        d *= m.discount()
    return r
