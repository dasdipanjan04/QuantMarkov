from utils.state_encoder import encode_states
from models.markov_model import MarkovChainModel
from strategies.markov_strategy import MarkovStrategy
from backtester import Backtester, compute_metrics
from data_loader.data_loader import DataLoader

data = DataLoader.fetch_yahoo("AAPL", start="2023-01-01", end="2023-12-31")

state_series = encode_states(data)

markov_model = MarkovChainModel()
strategy = MarkovStrategy(markov_model, state_series)

backtester = Backtester(data, strategy)
portfolio = backtester.run()
backtester.plot()

metrics = compute_metrics(portfolio)
print("\nPerformance Metrics:")
for key, value in metrics.items():
    print(f"{key}: {value:.2%}")
