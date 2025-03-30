import pandas as pd

class BaseStrategy:
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        raise NotImplementedError("Must implement generate_signals method")

class MarkovStrategy(BaseStrategy):
    def __init__(self, markov_model, state_series: pd.Series):
        self.model = markov_model
        self.state_series = state_series
        self.model.fit(state_series)

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        signals = []
        for i in range(len(self.state_series) - 1):
            current_state = self.state_series.iloc[i]
            next_state = self.model.predict_next_state(current_state)

            if next_state == 2:   # Predicting up
                signals.append(1)
            elif next_state == 0: # Predicting down
                signals.append(-1)
            else:
                signals.append(0)
        signals.append(0)
        return pd.Series(signals, index=data.index)
