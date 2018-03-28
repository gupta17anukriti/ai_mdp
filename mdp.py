import sys
from random import *

# abstract class for Markov Decision Process (MDP)
class mdp_t(object):
    def start_state(self):
        return self.get_state(0)
    def actions(self, s):
        pass
    def is_end(self, s):
        pass
    # transition to next state for (state, action) pair, returns (reward, next_state)
    def transition(self, s, a):
        ta = self.transitions(s, a) # get list of all possible transitions for (state, action)
        t = None                    # random state which we select based on probability distribution in 'ta'
        p, r = 0.0, uniform(0, 1)   # probability threshold for current state
        for i in range(len(ta)):
            t, p = ta[i], p + ta[i][0]
            if r < p:
                break
        return t[1], t[2]
    # this should return array of (probability, reward, next state) tuples
    def transitions(self, s, a):
        pass
    # discount must be between 0 and 1 (inclusive)
    def discount(self):
        return 1.0
    # return number of states in MDP (returning -1 for unknown)
    def size(self):
        return -1
    # returns state for given state id
    def get_state(self, id):
        return id

# abstract policy (i.e. what action we select for each state of MDP)
class policy_t(object):
    def __init__(self, mdp):
        self.mdp = mdp
    # returns policy for passed state
    def action(self, s):
        pass
    # generate episode from passed state, episode is array of [state, action, reward, state, .... ]
    # this array has 3n+1 states (n >= 0), it starts with passed state and ends with end
    # state (is_end(s) == True).
    def create_episode(self, s):
        e, m = [s], self.mdp
        while not m.is_end(s):
            a = self.action(s)      # get action per current policy
            t = m.transition(s, a)  # generate 'per distribution' transition
            e += [a, t[0], t[1]]
            s = t[1]                # advance to next state
        return e
    def update_parameters(self, s, a, sn, r):
        pass
    def end_update_parameters(self):
        pass

# random policy (for each state we select uniformly random action)
class random_policy_t(policy_t):
    def action(self, s):
        aa = self.mdp.actions(s)
        return aa[randint(0, len(aa) - 1)]

# policy from dictionary {state: action}
class dict_policy_t(policy_t):
    def __init__(self, mdp, d):
        self.mdp = mdp
        self.d = d
    def action(self, s):
        return self.d[s]

# return total reward (utility) of episode array
def episode_utility(m, e):
    r, d = 0, 1.0
    for i in range(1, len(e) - 1, 3):
        r += e[i + 1] * d
        d *= m.discount()
    return r

# convert episode array to string (for debugging and output)
def episode_to_str(m, e, rounding = 2, add_utility = True):
    s = str(e[0]) + ";"
    for i in range(1, len(e) - 1, 3):
        # action, reward, state
        s += str(e[i]) + "," + str(round(e[i + 1], rounding)) + "," + str(e[i + 2]) + ";";
    if add_utility:
        s += " ({})".format(round(episode_utility(m, e), rounding))
    return s
