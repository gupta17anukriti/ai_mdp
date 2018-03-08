import mdp
import random
import collections
import mdp_utils
import time

class volcano_crossing_t(mdp.mdp_t):
    def __init__(self, rows = 3, cols = 4, rs = ((2, 0, 2, True), (0, 2, -50, True), (0, 1, -50, True), (0, 3, 20, True))):
        map = [None] * rows
        for i in range(rows):
            map[i] = [(0.0, False)] * cols # (reward, is it end state)
        for r in rs:
            map[r[0]][r[1]] = (r[2], r[3])
        self.map = map
        self.shape, self.slip, self.disc, self.start = (rows, cols), 0.1, 1.0, (rows - 2, 0)

    def start_state(self):
        return self.start

    def actions(self, s):
        return ['S', 'N', 'W', 'E']

    def is_end(self, s):
        return self._end(s[0], s[1])

    def _r(self, row, col):
        return self.map[row][col][0];

    def _end(self, row, col):
        return self.map[row][col][1];

    def _invalid(self, s):
        return s[0] < 0 or s[1] < 0 or s[0] >= self.shape[0] or s[1] >= self.shape[1]

    def _next(self, s, a):
        if a == 'N':
            n = (s[0] - 1, s[1])
        elif a == 'S':
            n = (s[0] + 1, s[1])
        elif a == 'W':
            n = (s[0], s[1] - 1)
        else:
            n = (s[0], s[1] + 1)
        return s if self._invalid(n) else n

    # this should return array of (probability, reward, next state) tuples
    def transitions(self, s, a):
        if self.is_end(s):
            return []
        td = collections.defaultdict(float)
        for v in self.actions(s):
            td[self._next(s, v)] += (1.0 - self.slip if v == a else 0) + self.slip / 4
        return [(p, self._r(n[0], n[1]), n) for n, p in td.items()]

    def discount(self):
        return self.disc;

    def get_state(self, id):
        return (int(id / self.shape[0]), id % self.shape[1])

    def size(self):
        return self.shape[0] * self.shape[1]

if __name__ == '__main__':
    random.seed(time.time())
    m = volcano_crossing_t()
    mdp_utils.run_with_policy(m, mdp.random_policy_t(m), 'random policy')
    mdp_utils.print_episode(m, mdp.random_policy_t(m), 'example episode for random policy')

