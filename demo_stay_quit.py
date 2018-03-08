import mdp
import random
import time
import mdp_iterator as imdp
import mdp_utils

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
    def is_end(self, s):
        return s == 1
    def transitions(self, s, a):
        if s == 0 :
            return [(2.0/3.0, 4.0, 0), (1.0 - 2.0/3.0, 4.0, 1)] if a == 'stay' else [(1.0, 10.0, 1)]
        return [] # for s1 there are no transitions
    def size(self):
        return 2

class stay_policy_t(mdp.policy_t):
    def action(self, s):
        return 'stay'

class quit_policy_t(mdp.policy_t):
    def action(self, s):
        return 'quit'

if __name__ == '__main__':
    random.seed(time.time())
    m = stay_quit_t()
    mdp_utils.run_with_policy(m, mdp.random_policy_t(m), 'random policy (should be close to 10.5)')
    mdp_utils.run_with_policy(m, stay_policy_t(m), '<stay> policy (should be close to 12)')
    mdp_utils.run_with_policy(m, quit_policy_t(m), '<quit> policy (must be exactly 10)')

    # use policy evaluation to estimate utility of each policy
    print("<stay> policy evaluation (must be very close to 12):")
    print(imdp.policy_evaluation(m, stay_policy_t(m)))
    print("<quit> policy evaluation (must be very close to 10):")
    print(imdp.policy_evaluation(m, quit_policy_t(m)))
    print("random policy evaluation (must be very close to 10.5):")
    print(imdp.policy_evaluation(m, mdp.random_policy_t(m), 1000, 0.0)) # something is wrong here, I'm not getting 10.5 :(

    # calculate optimal policy
    print("optimal policy for each state:")
    print(imdp.value_iterator(m))
