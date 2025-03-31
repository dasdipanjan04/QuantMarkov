import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from strategies.base_strategy import BaseStrategy

class MetaStrategy(BaseStrategy):
    def __init__(self, strategies: list, weights: list = None):
        self.strategies = strategies
        self.weights = weights if weights else [1 / len(strategies)] * len(strategies)

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        signal_df = pd.DataFrame(index=data.index)
        for i, strategy in enumerate(self.strategies):
            strat_signal = strategy.generate_signals(data)
            signal_df[f'strategy_{i}'] = strat_signal * self.weights[i]
        return signal_df.sum(axis=1).apply(np.sign)


    def plot_signals(self, signals: pd.Series, data: pd.DataFrame):
        plt.figure(figsize=(14, 6))
        plt.plot(data['close'], label='Price', alpha=0.7)

        buy_signals = signals[signals > 0]
        sell_signals = signals[signals < 0]

        plt.scatter(buy_signals.index, data.loc[buy_signals.index, 'close'], label='Buy', marker='^', color='green')
        plt.scatter(sell_signals.index, data.loc[sell_signals.index, 'close'], label='Sell', marker='v', color='red')

        plt.title("Buy & Sell Signals")
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
