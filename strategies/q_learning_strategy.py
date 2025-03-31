# 📁 File: strategies/q_learning_strategy.py

import numpy as np
import pandas as pd
from strategies.base_strategy import BaseStrategy
import random
import collections.abc

class QTable:
    def __init__(self, actions, learning_rate=0.1, discount=0.9, epsilon=0.1):
        from collections import defaultdict
        self.actions = actions
        self.lr = learning_rate
        self.gamma = discount
        self.epsilon = epsilon

        self.q = defaultdict(lambda: [0.0] * len(self.actions))  # 👈 KILLS None once and for all

    def _sanitize_state(self, state):
        return tuple([0 if pd.isna(x) else int(x) for x in state])

    def get(self, state):
        state = self._sanitize_state(state)
        val = self.q[state]

        # Double check corruption
        if not isinstance(val, list) or len(val) != len(self.actions) or any(
            not isinstance(v, (int, float)) for v in val
        ):
            print(f"[FORCE REBUILD Q] Q[{state}] was corrupted → {val}")
            self.q[state] = [0.0] * len(self.actions)

        return self.q[state]

    def update(self, state, action_idx, reward, next_state):
        state = self._sanitize_state(state)
        next_state = self._sanitize_state(next_state)

        q_current = self.get(state)
        q_next = self.get(next_state)

        max_next = max(q_next)
        q_current[action_idx] += self.lr * (reward + self.gamma * max_next - q_current[action_idx])

    def select_action(self, state):
        state = self._sanitize_state(state)
        q_vals = self.get(state)
        if random.random() < self.epsilon:
            return random.randint(0, len(self.actions) - 1)
        return int(np.argmax(q_vals))

    def get_confidence(self, state):
        state = self._sanitize_state(state)
        q_vals = self.get(state)
        return max(q_vals) / (sum(abs(v) for v in q_vals) + 1e-6)


class QLearningStrategy(BaseStrategy):
    def __init__(self, data: pd.DataFrame, bins=10):
        self.data = data
        self.actions = [-1, 0, 1]  # sell, hold, buy
        self.qtable = QTable(self.actions)
        self.bins = bins

    def _discretize(self, window=5):
        pct = self.data['kalman'].pct_change().fillna(0)
        ma = self.data['kalman'].rolling(window).mean().bfill()
        deviation = (self.data['kalman'] - ma) / self.data['kalman']
        vol = pct.rolling(window).std().bfill()
        cycle = self.data['fourier'] - self.data['kalman']

        d1 = pd.cut(pct, self.bins, labels=False)
        d2 = pd.cut(deviation, self.bins, labels=False)
        d3 = pd.cut(vol, self.bins, labels=False)
        d4 = pd.cut(cycle, self.bins, labels=False)

        return list(zip(d1, d2, d3, d4))

    def train(self, episodes=10):
        states = self._discretize()
        print(f"Training Q-learning on {len(states)} states for {episodes} episodes")
        for _ in range(episodes):
            for t in range(len(states) - 1):
                state = tuple([0 if pd.isna(x) else int(x) for x in states[t]])
                next_state = tuple([0 if pd.isna(x) else int(x) for x in states[t + 1]])

                action_idx = self.qtable.select_action(state)
                action = self.actions[action_idx]

                price_now = self.data['close'].iloc[t]
                price_next = self.data['close'].iloc[t + 1]

                reward = 0.0
                if action == 1:
                    reward = float((price_next - price_now).item()) if hasattr(price_next - price_now, 'item') else float(price_next - price_now)
                elif action == -1:
                    reward = float((price_now - price_next).item()) if hasattr(price_now - price_next, 'item') else float(price_now - price_next)

                self.qtable.update(state, action_idx, reward, next_state)

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        signals = []
        states = self._discretize()
        for t in range(len(states)):
            state = tuple([0 if pd.isna(x) else int(x) for x in states[t]])
            action_idx = self.qtable.select_action(state)
            confidence = self.qtable.get_confidence(state)

            base_signal = self.actions[action_idx]
            weighted_signal = int(np.sign(base_signal) * confidence * 3)

            signals.append(weighted_signal)
        return pd.Series(signals, index=data.index)
