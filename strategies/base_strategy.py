import pandas as pd

class BaseStrategy:
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        raise NotImplementedError("Must implement generate_signals method")