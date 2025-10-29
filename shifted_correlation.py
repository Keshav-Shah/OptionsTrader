import pickle
import pandas as pd
import numpy as np

class ShiftedCorrelation:
    def __init__(self, data_path="stock_data_dict.pkl"):
        """
        Initialize with path to saved Yahoo Finance data dictionary (.pkl).
        """
        with open(data_path, "rb") as f:
            self.data_dict = pickle.load(f)

    def get_close(self, ticker):
        """
        Return the closing price time series for the given ticker.
        """
        if ticker not in self.data_dict:
            raise ValueError(f"Ticker '{ticker}' not found in saved data.")
        df = self.data_dict[ticker]
        if "Close" not in df.columns:
            raise ValueError(f"No 'Close' column found for {ticker}.")
        return df["Close"]

    def compute(self, ticker1, ticker2, shift_days):
        """
        Compute correlation between ticker1 and ticker2,
        after shifting ticker2 by +shift_days (forward in time).
        """
        s1 = self.get_close(ticker1)
        s2 = self.get_close(ticker2)

        # Align on overlapping dates
        df = pd.concat([s1, s2], axis=1, join="inner")
        df.columns = [ticker1, ticker2]

        # Shift second stock
        df[ticker2] = df[ticker2].shift(shift_days)

        # Drop NaN values introduced by shift
        df = df.dropna()

        if len(df) < 5:
            print("⚠️ Not enough overlapping data after shift.")
            return np.nan

        # Compute Pearson correlation
        corr = df[ticker1].corr(df[ticker2])
        return corr


if __name__ == "__main__":
    # Example usage:
    corr_tool = ShiftedCorrelation("stock_data_dict.pkl")

    ticker1 = "AAPL"
    ticker2 = "MSFT"
    shift_days = 0

    corr = corr_tool.compute(ticker1, ticker2, shift_days)
    print(f"📈 Correlation between {ticker1} and {ticker2} with {shift_days}-day shift: {corr:.4f}")
