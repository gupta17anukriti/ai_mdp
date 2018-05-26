from mdp import *
import numpy as np
from collections import *

def to_matrix(m: mdp_t, p: policy_t):
    N = m.size()  # number of states
    A = np.identity(N)
    b = np.zeros(N)

    def S(i):
        return m.get_state(i)

    # create dictionary (state: state_index)
    si = defaultdict()
    for j in range(N):
        si[S(j)] = j

    for j in range(N):
        s = S(j)
        if not m.is_end(s):
            # get list of transitions [(probability, reward, next state)]
            trans = m.transitions(s, p.action(s))
            for t, r, sn in trans:
                b[j] += t * r
                A[j][si[sn]] -= t * m.discount()

    return A, b

def policy_evaluation(m: mdp_t, p: policy_t):
    A, b = to_matrix(m, p)
    return np.linalg.solve(A, b)
