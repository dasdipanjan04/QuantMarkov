from utils.state_encoder import encode_states
from strategies.markov_strategy import MarkovStrategy, OrderNMarkovModel
from backtester import Backtester, compute_metrics
from data_loader.data_loader import DataLoader

# Load data
data = DataLoader.fetch_yahoo("AAPL", start="2023-01-01", end="2023-12-31")

# Encode price into discrete states
state_series = encode_states(data)

# Initialize Order-N Markov Model
markov_model = OrderNMarkovModel(order=2, num_states=3)

# Initialize Strategy
strategy = MarkovStrategy(
    markov_model=markov_model,
    state_series=state_series,
    data=data,
    prob_threshold=0.6,
    capital=10000
)

# Run Backtest
backtester = Backtester(data, strategy)
portfolio = backtester.run()
backtester.plot()

# Compute Metrics
metrics = compute_metrics(portfolio)
print("\n📊 Performance Metrics:")
for key, value in metrics.items():
    print(f"{key}: {value:.2%}")

# Get generated signals
signals = strategy.generate_signals(data)

# Plot Heatmap of transition probabilities
markov_model.plot_transition_heatmap()

# Plot signals on price
strategy.plot_signals(signals)
