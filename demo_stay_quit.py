import mdp
import random
import time
import mdp_iter as imdp
import mdp_utils as mu
import mdp_aprox as mc
import mdp_gauss as mg

#
# Simple game: at  initial  state  (s0)  we  have  2  actions  'stay' and 'quit', if we
# 'quit' we  get  $10  and  game  ends  (s1), if we 'stay' than with probability 2/3 we go to
# state s0 with reward $4, or with probability 1/3 we go to end state (s1) with also reward $4.
# For this game discount is default 1.0.
#
# Expectation for <stay> policy is $12 from E = 2/3($4 + E) + 1/3($4)
# Expectation for <quit> policy is obviously $10
# Expectation for random policy is $10.5 which is found from system of equations:
#   E  = 1/2($10) + 1/2(E')
#   E' = 2/3($4 + E) + 1/3($4)
#
class stay_quit_t(mdp.mdp_t):
    def actions(self, s):
        # we return same actions for all states, since s1 is end state and we have just 2 states
        return ['stay', 'quit']
    def start_state(self):
        return 'in'
    def is_end(self, s):
        return s == 'end'
    def transitions(self, s, a):
        if s == 'in' :
            return [(2.0/3.0, 4.0, 'in'), (1.0 - 2.0/3.0, 4.0, 'end')] if a == 'stay' else [(1.0, 10.0, 'end')]
        return [] # for s1 there are no transitions
    def size(self):
        return 2
    def get_state(self, id):
        return 'in' if id == 0 else 'end'

class stay_policy_t(mdp.policy_t):
    def action(self, s):
        return 'stay'

class quit_policy_t(mdp.policy_t):
    def action(self, s):
        return 'quit'

def run(m: mdp.mdp_t, p: quit_policy_t, msg):
    return mu.run(m, p, msg, 10000)

def title(msg):
    mu.separator(msg)

if __name__ == '__main__':
    random.seed(time.time())
    m = stay_quit_t()
    stay_policy = stay_policy_t(m)
    quit_policy = quit_policy_t(m)
    rand_policy = mdp.random_policy_t(m)

    title("Run many episodes for a policy and find average utility")
    run(m, rand_policy, 'random policy (should be close to 10.5)')
    run(m, stay_policy, '<stay> policy (should be close to 12)')
    run(m, quit_policy, '<quit> policy (must be exactly 10)')

    title("Calculate utility for policy using policy evaluation algorithm")
    # use policy evaluation to estimate utility of each policy
    print("\n<stay> policy evaluation (must be very close to 12):")
    print(imdp.policy_evaluation(m, stay_policy))
    print("\n<quit> policy evaluation (must be very close to 10):")
    print(imdp.policy_evaluation(m, quit_policy))

    title("Find optimal policy using value iterator algorithm")
    # calculate optimal policy
    print("\noptimal policy for each state:")
    print(imdp.value_iterator(m))

    title("Approximate transitions and rewards based on random policy")
    # calculate optimal policy
    print("\napproximate transitions/rewards and then do optimal policy evaluation")
    ea = run(m, rand_policy, '')
    me = mc.approximate_model(ea, 'in', 1.0)
    pme = imdp.value_iterator(me)
    print("\ncalculated policy:")
    print(pme)
    pe = mdp.dict_policy_t(me, pme)
    print("\npolicy evaluation:")
    print(imdp.policy_evaluation(me, pe))

    title("Policy evaluation using Gaussian elimination")
    print("\nOptimal policy (stay) :")
    print(mg.policy_evaluation(m, stay_policy))
    print("\nSuboptimal policy (quit) :")
    print(mg.policy_evaluation(m, quit_policy))
