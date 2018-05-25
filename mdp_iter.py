from mdp import *
from collections import *

EPSILON = 10e-6
UEPSILON = 1.0

# given MDP and policy, calculate value (V) of each state. this is, in essence aproximate solution of
# system of linear equations (which we could also do using elimination, but it is slower)
def policy_evaluation(m: mdp_t, p: policy_t, niter = 1000, epsilon = EPSILON):
    V0, N = defaultdict(float), m.size()
    def S(i):
        return m.get_state(i)
    for i in range(niter):                              # for each iteration
        V1 = defaultdict(float)
        for j in range(N):                              # for each state
            s = S(j)
            if not m.is_end(s):
                ta = m.transitions(s, p.action(s))       # get list of actions for a state
                if len(ta) > 0:
                    V1[s] = sum(p * (r + m.discount() * V0[sn]) for p, r, sn in ta)
        if epsilon > 0.0 and max(abs(V0[S(k)] - V1[S(k)]) for k in range(N)) < epsilon:
            break
        V0 = V1
    return V0

# given MDP calculate optimal policy (dictionary {state: action})
def value_iterator(m: mdp_t, niter = 1000, epsilon = EPSILON):
    V0, P, N = defaultdict(float), defaultdict(), m.size()  # optimal state values, optimal policy
    def Vopt(s):                                            # current state value for optimal policy
        return V0[s]
    def Qopt(s, a):                                         # optimal Q-value for state-action pair
        discount = m.discount() * UEPSILON
        return sum(p * (r + (discount * Vopt(sn))) for p, r, sn in m.transitions(s, a))
    def S(i):
        return m.get_state(i)
    for i in range(niter):
        V1 = defaultdict(float)
        for j in range(N):
            s = S(j)
            if not m.is_end(s):
                aa = m.actions(s)
                if len(aa) > 0:
                    V1[s], P[s] = max((Qopt(s, a), a) for a in aa)
        if epsilon > 0.0 and max(abs(V0[S(k)] - V1[S(k)]) for k in range(N)) < epsilon:
           break
        V0 = V1
    return P
