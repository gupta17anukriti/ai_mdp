from mdp import *

EPSILON = 10e-6

# given MDP and policy, calculate value (V) of each state. this is, in essence aproximate solution of
# system of linear equations (which we could also do using elimination, but it is slower)
def policy_evaluation(m: mdp_t, p: policy_t, niter = 1000, epsilon = EPSILON):
    V0 = [0] * m.size()
    for i in range(niter):                              # for each iteration
        V1 = [0] * m.size()
        for j in range(m.size()):                       # for each state
            s = m.get_state(j)
            if not m.is_end(s):
                ta = m.transitions(s, p.action(s))       # get list of actions for a state
                V1[s] = sum(t[0] * (t[1] + m.discount() * V0[t[2]]) for t in ta)
            if epsilon > 0.0 and max(abs(V0[k] - V1[k]) for k in range(m.size())) < epsilon:
                break
        V0 = V1
    return V0

# given MDP calculate optimal policy (array of actions for each state index)
def value_iterator(m: mdp_t, niter = 1000, epsilon = EPSILON):
    V0, P = [0] * m.size(), [None] * m.size()   # optimal state values, optimal policy
    def Vopt(s):                                # current state value for optimal policy
        return V0[s]
    def Qopt(s, a):                             # optimal Q-value for state-action pair
        return sum(t[0] * (t[1] + m.discount() * Vopt(t[2])) for t in m.transitions(s, a))
    for i in range(niter):
        V1 = [0.0] * m.size()
        for j in range(m.size()):
            s = m.get_state(j)
            if not m.is_end(s):
                V1[s], P[s] = max((Qopt(s, a), a) for a in m.actions(s))
        if epsilon > 0 and max(abs(V0[k] - V1[k]) for k in range(m.size())) < epsilon:
            break
        V0 = V1
    return P
