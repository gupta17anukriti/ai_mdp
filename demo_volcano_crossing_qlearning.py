from demo_volcano_crossing import *

if __name__ == '__main__':
    random.seed(time.time())

    print("approximate q-value using Q-learning")
    m = volcano_crossing_t()
    policy1 = mdpi.value_iterator(m)
    print("optimal policy through value iterator:")
    print(policy1)
    po = mdp.dict_policy_t(m, policy1)
    ea = mdp_utils.run_with_policy(m, po, '', 10000)
    qs = mdpa.approximate_q_values_qlearning(ea, m.discount())
    failed = mdp_utils.compare_dictionaries(policy1, qs[2])
    if failed:
        print("Estimation FAILED")
    else:
        print("Estimation succeeded")

    print("\n\napproximate q-value using Q-learning, but now user random policy")
    m = volcano_crossing_t()
    po = mdp.random_policy_t(m)
    ea = mdp_utils.run_with_policy(m, po, '', 10000)
    qs = mdpa.approximate_q_values_qlearning(ea, m.discount())
    failed = mdp_utils.compare_dictionaries(policy1, qs[2])
    if failed:
        print("Estimation FAILED")
    else:
        print("Estimation succeeded")

