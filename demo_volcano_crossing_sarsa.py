from demo_volcano_crossing import *

if __name__ == '__main__':
    random.seed(time.time())

    print("approximate q-value using SARSA")
    m = volcano_crossing_t()
    policy1 = mdpi.value_iterator(m)
    print("optimal policy through value iterator:")
    print(policy1)
    po = mdp.dict_policy_t(m, policy1)
    ea = mdp_utils.run_with_policy(m, po, '', 10000)
    qs = mdpa.approximate_q_values_sarsa(ea, m.discount())
    ## print(qs)
    print("optimal policy through model free monte carlo:")
    policy2 = mdpa.q_values_opt_policy(qs)
    print(policy2)

    failed = mdp_utils.compare_dictionaries(policy1, policy2)
    if failed:
        print("Estimation FAILED")
    else:
        print("Estimation succeeded")

    print("\n\napproximate q-value using model free SARSA, but now user random policy")
    m = volcano_crossing_t()
    po = mdp.random_policy_t(m)
    ea = mdp_utils.run_with_policy(m, po, '', 10000)
    qs = mdpa.approximate_q_values_sarsa(ea, m.discount())
    print("optimal policy through model free monte carlo:")
    policy3 = mdpa.q_values_opt_policy(qs)
    print(policy3)

    failed = mdp_utils.compare_dictionaries(policy1, policy3)
    if failed:
        print("Estimation FAILED")
    else:
        print("Estimation succeeded")

