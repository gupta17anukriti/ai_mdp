from demo_volcano_crossing import *
from mdp_aprox_egreedy import *

if __name__ == '__main__':
    seed(time.time())
    eta = 0.5

    m = volcano_crossing_t()
    policy1 = mdpi.value_iterator(m)
    print("optimal policy through value iterator:")
    print(policy1)
    v = mdpi.policy_evaluation(m, dict_policy_t(m, policy1), 1000)
    print("optimal policy evaluation:")
    print(v)

    print("\n\nApproximate q-value using Q-learning with epsilon-greedy policy")
    m = volcano_crossing_t()
    qs = qval_qlearn_epsilon_greedy_ex(m, 10000)
    failed = mdp_utils.compare_dictionaries(policy1, qs[1])
    if failed:
        print("Policy calculation FAILED")
    else:
        print("Policy calculation succeeded")
    print("policy evaluation:")
    print(qs[1])

