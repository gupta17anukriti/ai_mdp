from mdp import *
from mdp_utils import *

class mdp_reinforcement_policy_t(object):
    # m - instance of MDP (mdp_t derivative), p is policy
    def __init__(self, m: mdp_t, p: policy_t):
        self.m = m
        self.p = p

    def train(self, iter = 1000, s = None):
        if s is None:
           s = self.m.start_state()
        for i in range(iter):
            if self.m.is_end(s):
                break
            a = self.p.action(s)
            r, sn = self.m.transition(s, a)
            self.p.update_parameters(s, a, sn, r)
        self.p.end_update_parameters()
