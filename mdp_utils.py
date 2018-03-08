import mdp

def run_with_policy(m, policy, descr, cnt = 10000):
    print("running several episodes with " + descr + ":")
    u = 0.0
    for i in range(0, cnt):
        e = policy.create_episode(0)
        # print(mdp.episode_to_str(e))
        u += mdp.episode_utility(m, e)
    print("average episode utility: " + str(u / float(cnt)))
