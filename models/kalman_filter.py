import numpy as np
import pandas as pd

class KalmanFilter:
    def __init__(self, R=0.01, Q=1e-5):
        self.R = R
        self.Q = Q
        self.A = 1
        self.H = 1

    def apply(self, series: pd.Series) -> pd.Series:
        n = len(series)
        xhat = np.zeros(n)
        P = np.zeros(n)
        xhatminus = np.zeros(n)
        Pminus = np.zeros(n)
        K = np.zeros(n)

        xhat[0] = float(series.iloc[0].item()) if hasattr(series.iloc[0], 'item') else float(series.iloc[0])
        P[0] = 1.0

        for k in range(1, n):
            xhatminus[k] = self.A * xhat[k - 1]
            Pminus[k] = self.A * P[k - 1] * self.A + self.Q
            innovation = series.iloc[k] - self.H * xhatminus[k]
            if hasattr(innovation, 'item'):
                innovation = float(innovation.item())
            K[k] = Pminus[k] * self.H / (self.H * Pminus[k] * self.H + self.R)
            xhat[k] = xhatminus[k] + K[k] * innovation
            P[k] = (1 - K[k] * self.H) * Pminus[k]

        return pd.Series(xhat, index=series.index)
