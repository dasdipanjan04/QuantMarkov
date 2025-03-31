# 📁 File: strategies/fourier_strategy.py

import pandas as pd
import numpy as np
from strategies.base_strategy import BaseStrategy

class FourierCycleStrategy(BaseStrategy):
    def __init__(self, data: pd.DataFrame, threshold: float = 0.01):
        self.data = data
        self.threshold = threshold

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        try:
            close = self.data['close']
            if isinstance(close, pd.DataFrame):
                close = close.squeeze()

            if not isinstance(close, pd.Series):
                raise TypeError(f"'close' must be a Series, got {type(close)}")

            close = pd.to_numeric(close, errors='coerce')

            fourier = pd.to_numeric(self.data['fourier'], errors='coerce')
        except Exception as e:
            print("[ERROR] FourierStrategy input conversion failed:", e)
            raise

        residual = fourier - close
        residual = residual.fillna(0)

        signals = []
        for r in residual:
            if isinstance(r, (pd.Timestamp, pd.Timedelta)):
                signals.append(0)
            elif r > self.threshold:
                signals.append(-1)  # Price above cycle → sell
            elif r < -self.threshold:
                signals.append(1)   # Price below cycle → buy
            else:
                signals.append(0)

        return pd.Series(signals, index=self.data.index)
