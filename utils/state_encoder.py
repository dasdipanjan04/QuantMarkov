import pandas as pd

def encode_states(data: pd.DataFrame, threshold=0.002) -> pd.Series:
    returns = data['close'].pct_change().fillna(0)
    states = returns.apply(lambda r: 2 if r > threshold else (0 if r < -threshold else 1))
    return states
