import numpy as np
import pandas as pd

class FourierFilter:
    def __init__(self, keep_ratio=0.05):
        self.keep_ratio = keep_ratio

    def apply(self, series: pd.Series) -> pd.Series:
        series = series.squeeze()
        n = len(series)
        fft = np.fft.fft(series.values)
        freqs = np.fft.fftfreq(n)
        fft_filtered = np.copy(fft)
        cutoff = int(n * self.keep_ratio)
        indices = np.argsort(np.abs(freqs))
        mask = np.zeros(n, dtype=bool)
        mask[indices[:cutoff]] = True
        fft_filtered[~mask] = 0
        smoothed = np.fft.ifft(fft_filtered).real
        return pd.Series(smoothed, index=series.index)
