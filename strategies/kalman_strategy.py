import pandas as pd
import numpy as np
from strategies.base_strategy import BaseStrategy

class KalmanTrendStrategy(BaseStrategy):
    def __init__(self, data: pd.DataFrame, slope_threshold: float = 0.001):
        self.data = data
        self.slope_threshold = slope_threshold

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        if 'kalman' not in self.data.columns:
            raise ValueError("Data must include 'kalman' column")

        trend = self.data['kalman']
        slope = trend.diff().fillna(0)

        signals = []
        for s in slope:
            if s > self.slope_threshold:
                signals.append(1)
            elif s < -self.slope_threshold:
                signals.append(-1)
            else:
                signals.append(0)

        return pd.Series(signals, index=self.data.index)
