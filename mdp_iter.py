from mdp import *
from collections import *
EPSILON = 10e-6

# return true if algorithm converged to specified epsilon value
def converged(epsilon, m, V0, V1):
    N = m.size()
    def S(i):
        return m.get_state(i)
    return epsilon > 0.0 and max(abs(V0[S(k)] - V1[S(k)]) for k in range(N)) < epsilon


# given MDP and policy, calculate value (V) of each state. this is, in essence aproximate solution of
# system of linear equations (which we could also do using elimination, but it is slower)
def policy_evaluation(m: mdp_t, p: policy_t, niter = 1000, epsilon = EPSILON):
    V0 = defaultdict(float)  # Vopt(s, a) values, initially all 0
    N = m.size()  # number of states

    # return state by its index
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
        if converged(epsilon, m, V0, V1):
            break
        V0 = V1
    return V0


# given MDP calculate optimal policy (dictionary {state: action}), this is similar to policy evaluation, but
# at each iteration we recalculate policy based on current approximation of V values
def value_iterator(m: mdp_t, niter = 1000, epsilon = EPSILON):
    V0 = defaultdict(float) # Vopt(s, a) values, initially all 0
    P = defaultdict()       # Popt(s) - optimal policy for each state
    N = m.size()            # number of states

    # current state value for optimal policy
    def Vopt(s):
        return V0[s]

    # Qopt(s, a) -value for state-action pair
    def Qopt(s, a):
        discount = m.discount()
        return sum(p * (r + (discount * Vopt(sn))) for p, r, sn in m.transitions(s, a))

    # return state by its index
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
        if converged(epsilon, m, V0, V1):
            break
        V0 = V1
    return P
