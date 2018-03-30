from mdp_iter import *
from mdp_aprox import *
from mdp_utils import *
import math

# probs is array of floats [0..1], sum of all number should be 1,
# returns random id weighted by prob[id] value
def ETA(N, sa, eta):
    _eta = eta
    if _eta is None:
        _eta = 1.0 / (1.0 + N[sa])  # _eta = 1.0 / math.sqrt(1.0 + N[(s, a)])
        N[sa] += 1
    return _eta

# 's' - state, 'actions' - possible actions for state,
#  Q - Qopt dictionary {(s, a): value}
def epsilon_greedy(s, actions, Q, epsilon):
    # with probability epsilon we select random action
    if uniform(0.0, 1.0) < epsilon:
        return random_choice(actions)
    else:
        # select optimal value found so far
        vopt, aopt = None, None
        for a in actions:
            q = Q[(s, a)] if ((s, a) in Q) else None
            if q is not None and q > 0 and (vopt is None or q > vopt):
                vopt, aopt = Q[(s, a)], a
        return random_choice(actions) if aopt is None else aopt

def Vopt(Q, s, actions, is_end):
    if is_end:
        return 0
    v = None
    for a in actions:
        if v is None or Q[(s, a)] > v:
            v = Q[(s, a)]
    return 0 if v is None else v

def qval_qlearn_epsilon_greedy(m: mdp_t, num_episodes, eta = None):
    Qopt, N, d = defaultdict(float), defaultdict(int), m.discount() * UEPSILON
    for i in range(num_episodes):
        epsilon = 1.0 - (i + 1.0) / num_episodes
        s = m.start_state()
        # run single episode
        while not m.is_end(s):
            a            = epsilon_greedy(s, m.actions(s), Qopt, epsilon)
            r, sn        = m.transition(s, a)
            _eta         = ETA(N, (s,a), eta)
            prediction   = Qopt[(s, a)]
            target       = r + d * Vopt(Qopt, sn, m.actions(sn), m.is_end(sn))
            Qopt[(s, a)] = (1.0 - _eta) * prediction + _eta * target
            s = sn
    # return optimal policy and and its Qs and Vs
    return Qopt

def qval_qlearn_epsilon_greedy_ex(m: mdp_t, num_episodes, eta = None):
    qs = qval_qlearn_epsilon_greedy(m, num_episodes, eta)
    return qs, q_values_opt_policy(qs)