import random
import numpy as np
from collections import defaultdict


class QLearningAgent:
    def __init__(self, n_actions=3, alpha=0.1, gamma=0.9, epsilon=1.0, eps_min=0.01, eps_decay=0.995):
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_min = eps_min
        self.eps_decay = eps_decay
        self.q_table = defaultdict(lambda: np.zeros(self.n_actions))

    def get_action(self, state):
        s = tuple(state)
        if random.random() < self.epsilon:
            return random.randint(0, self.n_actions - 1)
        return int(np.argmax(self.q_table[s]))

    def update(self, state, action, reward, next_state):
        s, ns = tuple(state), tuple(next_state)
        current = self.q_table[s][action]
        best_next = np.max(self.q_table[ns])
        target = reward + self.gamma * best_next
        self.q_table[s][action] = current + self.alpha * (target - current)
        # decay exploration
        self.epsilon = max(self.eps_min, self.epsilon * self.eps_decay)

    def save_q(self, path="q_table.npy"):
        # basic save (converts dict to lists)
        keys = list(map(list, self.q_table.keys()))
        vals = list(self.q_table.values())
        np.save(path, {'keys': keys, 'vals': vals})

    def load_q(self, path="q_table.npy"):
        data = np.load(path, allow_pickle=True).item()
        keys, vals = data['keys'], data['vals']
        for k, v in zip(keys, vals):
            self.q_table[tuple(k)] = v
