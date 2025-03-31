import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from strategies.base_strategy import BaseStrategy

class OrderNMarkovModel:
    def __init__(self, order=2, num_states=3):
        self.order = order
        self.num_states = num_states
        self.transition_probs = {}

    def fit(self, state_series):
        for i in range(len(state_series) - self.order):
            history = tuple(state_series[i:i+self.order])
            next_state = state_series[i + self.order]
            if history not in self.transition_probs:
                self.transition_probs[history] = [0] * self.num_states
            self.transition_probs[history][next_state] += 1

        for hist, counts in self.transition_probs.items():
            total = sum(counts)
            self.transition_probs[hist] = [c / total for c in counts]

    def predict_next_state(self, current_history):
        history = tuple(current_history[-self.order:])
        probs = self.transition_probs.get(history)
        if probs:
            return np.random.choice(self.num_states, p=probs), max(probs)
        return 1, 0.0  # default to 'flat' if unseen

    def plot_transition_heatmap(self):
        matrix = {}
        for hist, probs in self.transition_probs.items():
            key = '→'.join(map(str, hist))
            matrix[key] = probs

        df = pd.DataFrame.from_dict(matrix, orient='index', columns=['Down', 'Flat', 'Up'])
        plt.figure(figsize=(10, 6))
        sns.heatmap(df, annot=True, cmap='viridis', fmt=".2f")
        plt.title('Transition Probability Heatmap (Order-N)')
        plt.ylabel('History')
        plt.xlabel('Next State')
        plt.tight_layout()
        plt.show()

class MarkovStrategy(BaseStrategy):
    def __init__(self, markov_model, state_series: pd.Series, data: pd.DataFrame, 
                 prob_threshold: float = 0.6, capital: float = 10000):
        self.model = markov_model
        self.state_series = state_series
        self.data = data
        self.prob_threshold = prob_threshold
        self.capital = capital
        self.model.fit(state_series.tolist())

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        signals = []

        for i in range(self.model.order, len(self.state_series) - 1):
            current_history = self.state_series.iloc[i - self.model.order:i].tolist()
            predicted_state, confidence = self.model.predict_next_state(current_history)
            current_state = current_history[-1]

            if current_state == 0 and predicted_state == 2 and confidence > self.prob_threshold:
                signals.append(1)
            elif current_state == 2 and predicted_state == 0 and confidence > self.prob_threshold:
                signals.append(-1)
            else:
                signals.append(0)

        signals = [0] * self.model.order + signals + [0]
        signals = pd.Series(signals[:len(data)], index=data.index)
        signals = self.trailing_stop_filter(signals)
        return signals

    def trailing_stop_filter(self, signals: pd.Series, trailing_stop_pct: float = 0.03) -> pd.Series:
        pos = 0
        entry_price = 0
        new_signals = []

        for i, signal in enumerate(signals):
            price = self.data['close'].iloc[i]

            if signal == 1:
                entry_price = price
                pos = 1
                new_signals.append(1)
            elif signal == -1:
                entry_price = price
                pos = -1
                new_signals.append(-1)
            elif pos == 1:
                if price < entry_price * (1 - trailing_stop_pct):
                    new_signals.append(-1)
                    pos = 0
                else:
                    new_signals.append(0)
            elif pos == -1:
                if price > entry_price * (1 + trailing_stop_pct):
                    new_signals.append(1)
                    pos = 0
                else:
                    new_signals.append(0)
            else:
                new_signals.append(0)

        return pd.Series(new_signals, index=signals.index)

    def plot_signals(self, signals: pd.Series):
        plt.figure(figsize=(14, 6))
        plt.plot(self.data['close'], label='Close Price', alpha=0.7)
        buy_signals = self.data['close'][signals == 1]
        sell_signals = self.data['close'][signals == -1]
        plt.scatter(buy_signals.index, buy_signals, label='Buy Signal', marker='^', color='green')
        plt.scatter(sell_signals.index, sell_signals, label='Sell Signal', marker='v', color='red')
        plt.legend()
        plt.title('Price and Markov Strategy Signals')
        plt.grid(True)
        plt.tight_layout()
        plt.show()
