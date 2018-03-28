import mdp

def run_with_policy(m, policy, descr, cnt = 1000, print_episode = False):
    print("\nrunning " + str(cnt) + " episode(s) with " + descr + ":")
    u = 0.0
    ea = []
    for i in range(0, cnt):
        e = policy.create_episode(m.start_state())
        if print_episode:
            print(mdp.episode_to_str(m, e))
        u += mdp.episode_utility(m, e)
        ea.append(e)
    print("average episode utility: " + str(u / float(cnt)))
    return ea

def compare_dictionaries(pd, pe):
    failed = False
    if len(pd) == len(pe):
        for k, v in pd.items():
            if pe[k] != v:
                print("failed for " + str(k) + " - " + str(v) + " vs. " + str(pe[k]))
                failed = True
    return failed
