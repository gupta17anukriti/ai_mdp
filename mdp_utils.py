import mdp
import random

# convert episode array to string (for debugging and output)
def episode_to_str(m, e, rounding = 2, add_utility = True):
    s = str(e[0]) + ";"
    for i in range(1, len(e) - 1, 3):
        # action, reward, state
        s += str(e[i]) + "," + str(round(e[i + 1], rounding)) + "," + str(e[i + 2]) + ";";
    if add_utility:
        s += " ({})".format(round(mdp.episode_utility(m, e), rounding))
    return s

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
    failed = True
    if len(pd) == len(pe):
        failed = False
        for k, v in pd.items():
            if pe[k] != v:
                print("failed for " + str(k) + " - " + str(v) + " vs. " + str(pe[k]))
                failed = True
    else:
        print("mismatch of number of elements")
    return failed


def multi_choise(probs):
    p, v = random.uniform(0.0, 1.0), 0.0
    for i in range(len(probs)):
        if v > p: return i
        v += probs[i]
    return len(probs) - 1

# returns random value from array (uniformly distributed)
def random_choice(arr):
    return arr[random.randint(0, len(arr) - 1)]
