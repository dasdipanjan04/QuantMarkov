import pandas as pd
import re

def encode_states(data: pd.DataFrame, threshold=0.002) -> pd.Series:
    # Flatten MultiIndex if needed
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = ['_'.join(col).strip().lower() for col in data.columns.values]
    else:
        data.columns = data.columns.str.strip().str.lower()

    # Dynamically strip ticker suffixes like _aapl, _msft, etc.
    data.columns = [re.sub(r'_(\w+)$', '', col) for col in data.columns]

    # Confirm 'close' exists now
    assert 'close' in data.columns, f"'close' not found in: {data.columns}"

    # Calculate returns and map to states
    returns = data['close'].pct_change().fillna(0)
    states = returns.apply(lambda r: 2 if r > threshold else (0 if r < -threshold else 1))

    return states

