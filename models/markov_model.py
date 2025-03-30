import numpy as np
from collections import Counter

class MarkovChainModel:
    def __init__(self, num_states=3):
        self.num_states = num_states
        self.transition_matrix = np.zeros((num_states, num_states))

    def fit(self, state_series):
        counts = Counter()
        for (i, j) in zip(state_series[:-1], state_series[1:]):
            counts[(i, j)] += 1

        for i in range(self.num_states):
            row_total = sum(counts[(i, k)] for k in range(self.num_states))
            for j in range(self.num_states):
                self.transition_matrix[i][j] = counts[(i, j)] / row_total if row_total > 0 else 0

    def predict_next_state(self, current_state):
        probs = self.transition_matrix[current_state]
        return np.random.choice(self.num_states, p=probs)
