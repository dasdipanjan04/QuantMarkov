import pandas as pd
import yfinance as yf

class DataLoader:
    def __init__(self, filepath=None):
        self.filepath = filepath

    def load(self):
        if self.filepath:
            df = pd.read_csv(self.filepath, parse_dates=['timestamp'])
            df.set_index('timestamp', inplace=True)
            df = df[['open', 'high', 'low', 'close', 'volume']].dropna()
            return df
        else:
            raise ValueError("No file path provided")

    @staticmethod
    def fetch_yahoo(symbol: str, start: str, end: str, interval: str = "1d"):
        try:
            data = yf.download(symbol, start=start, end=end, interval=interval)
            data.reset_index(inplace=True)
            data.rename(columns={
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume',
                'Date': 'timestamp'
            }, inplace=True)
            data = data[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
            data.set_index('timestamp', inplace=True)
            return data
        except Exception as e:
            print(f"Error fetching data for {symbol}: {e}")
            return None
