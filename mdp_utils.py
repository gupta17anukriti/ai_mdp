import mdp

def run_with_policy(m, policy, descr, cnt = 10000, print_episode = False):
    print("running " + str(cnt) + " episode(s) with " + descr + ":")
    u = 0.0
    for i in range(0, cnt):
        e = policy.create_episode(m.start_state())
        if print_episode:
            print(mdp.episode_to_str(m, e))
        u += mdp.episode_utility(m, e)
    print("average episode utility: " + str(u / float(cnt)))

def print_episode(m, policy, descr):
    run_with_policy(m, policy, descr, 10, True)
