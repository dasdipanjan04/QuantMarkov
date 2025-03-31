from utils.state_encoder import encode_states
from strategies.markov_strategy import MarkovStrategy, OrderNMarkovModel
from backtester import Backtester, compute_metrics
from data_loader.data_loader import DataLoader
from strategies.kalman_strategy import KalmanTrendStrategy
from strategies.fourier_strategy import FourierCycleStrategy
from strategies.q_learning_strategy import QLearningStrategy
from strategies.meta_strategy import MetaStrategy
from models.kalman_filter import KalmanFilter
from models.fourier_filter import FourierFilter

data = DataLoader.fetch_yahoo("AAPL", start="2023-01-01", end="2023-01-31")


kf = KalmanFilter()
ff = FourierFilter(keep_ratio=0.03)
data['kalman'] = kf.apply(data['close'])
data['fourier'] = ff.apply(data['close'].squeeze())


# Create individual strategies
kalman_strat = KalmanTrendStrategy(data)
fourier_strat = FourierCycleStrategy(data)
q_strat = QLearningStrategy(data)
q_strat.train(episodes=50)

# Combine with meta-strategy
meta = MetaStrategy(strategies=[kalman_strat, fourier_strat, q_strat], weights=[0.3, 0.3, 0.4])

# Backtest
backtester = Backtester(data, meta)
portfolio = backtester.run()
backtester.plot()

metrics = compute_metrics(portfolio)
print("\n📊 MetaStrategy Performance:")
for k, v in metrics.items():
    print(f"{k}: {v:.2%}")

# Reuse the final strategy used in backtester
signals = meta.generate_signals(data)
meta.plot_signals(signals, data)

