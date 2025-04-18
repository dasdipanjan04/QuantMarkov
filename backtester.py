import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from strategies.markov_strategy import BaseStrategy

class Backtester:
    def __init__(self, data: pd.DataFrame, strategy: BaseStrategy, initial_capital: float = 10000.0):
        self.data = data
        self.strategy = strategy
        self.initial_capital = initial_capital
        self.results = None

    def run(self):
        close = self.data['close']
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]  # use first column if multiple

        signals = self.strategy.generate_signals(self.data)
        positions = signals.replace(-1, 0).cumsum()
        portfolio = pd.DataFrame(index=self.data.index)
        portfolio['positions'] = positions
        portfolio['holdings'] = close * portfolio['positions']
        portfolio['cash'] = self.initial_capital - (signals * close).cumsum()
        portfolio['total'] = portfolio['cash'] + portfolio['holdings']
        portfolio['returns'] = portfolio['total'].pct_change().fillna(0)
        self.results = portfolio
        return portfolio

    def plot(self):
        if self.results is not None:
            self.results['total'].plot(title='Portfolio Value Over Time', figsize=(12, 6))
            plt.ylabel('Portfolio Value')
            plt.grid(True)
            plt.show()
        else:
            print("Run the backtest first.")


def compute_metrics(portfolio: pd.DataFrame):
    total_return = (portfolio['total'].iloc[-1] / portfolio['total'].iloc[0]) - 1
    sharpe_ratio = (portfolio['returns'].mean() / portfolio['returns'].std()) * np.sqrt(252)
    drawdown = portfolio['total'] / portfolio['total'].cummax() - 1
    max_drawdown = drawdown.min()
    return {
        'Total Return': total_return,
        'Sharpe Ratio': sharpe_ratio,
        'Max Drawdown': max_drawdown
    }