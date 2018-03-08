import mdp
import random
import mdp_utils
import time
import mdp_aprox as mdpa
import mdp_iter as mdpi

class volcano_crossing_t(mdp.mdp_t):
    def __init__(self, slip = 0.1, disc = 1.0):
        rows, cols = 3, 4
        rs = [(2, 0, 2, True), (0, 2, -50, True), (1, 2, -50, True), (0, 3, 20, True)] # [(row, col, reward, is_end)]
        map = [None] * rows
        for i in range(rows):
            map[i] = [(0.0, False)] * cols # (reward, is it end state)
        for r in rs:
            map[r[0]][r[1]] = (r[2], r[3])
        self.map = map
        self.shape, self.slip, self.disc, self.start = (rows, cols), slip, disc, (rows - 2, 0)
        self._action_moves = {'S': (1, 0), 'N':(-1, 0), 'W':(0, -1), 'E':(0, 1)}
        self._actions = [k for k in self._action_moves.keys()]

    def start_state(self):
        return self.start

    def actions(self, s):
        return self._actions

    def is_end(self, s):
        return self.map[s[0]][s[1]][1];
        return self._end(s[0], s[1])

    def _r(self, row, col):
        return self.map[row][col][0];

    def _invalid(self, s):
        return s[0] < 0 or s[1] < 0 or s[0] >= self.shape[0] or s[1] >= self.shape[1]

    def _next(self, s, a):
        inc = self._action_moves[a]
        n = (s[0] + inc[0],s[1] + inc[1])
        return s if self._invalid(n) else n

    # this should return array of (probability, reward, next state) tuples
    def transitions(self, s, a):
        assert(not self.is_end(s))
        td = []
        for v in self.actions(s):
            n = self._next(s, v)
            td.append(((1.0 - self.slip if v == a else 0) + self.slip / 4, self._r(n[0], n[1]), n));
        return td

    def discount(self):
        return self.disc

    def get_state(self, id):
        return (int(id / self.shape[1]), id % self.shape[1])

    def size(self):
        return self.shape[0] * self.shape[1]

if __name__ == '__main__':
    random.seed(time.time())
    m = volcano_crossing_t()
    policy = mdp.random_policy_t(m)
    print("---")
    print("my policy1:")
    mp = {(1, 0): 'E', (0, 0): 'E', (0, 1): 'S', (1, 1): 'S', (2, 1): 'E', (2, 2): 'E', (2, 3): 'N', (1, 3): 'N'}
    print(mp)
    print("my policy 1 evaluation:")
    print(mdpi.policy_evaluation(m, mdp.dict_policy_t(m, mp)))
    print("---")
    print("my policy2:")
    mp = {(1, 0): 'S', (0, 0): 'S', (1, 1): 'S', (0, 1): 'S', (2, 1): 'E', (2, 2): 'E', (2, 3): 'N', (1, 3): 'N'}
    print(mp)
    print("my policy 2 evaluation:")
    print(mdpi.policy_evaluation(m, mdp.dict_policy_t(m, mp)))
    print("---")
    mdp_utils.run_with_policy(m, policy, 'random policy')
    print("---")
    print("monte carlo evaluation")
    ea = mdp_utils.run_with_policy(m, policy, '')
    me = mdpa.approximate_model(ea, m.start_state(), m.discount())
    print("estimated optimal policy:")
    pe = mdpi.value_iterator(me)
    print(pe)
    print("policy evaluation:")
    print(mdpi.policy_evaluation(me, mdp.dict_policy_t(me, pe)))
