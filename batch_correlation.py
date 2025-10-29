import itertools
import pandas as pd
from shifted_correlation import ShiftedCorrelation

def compute_all_shifted_correlations(tickers, shift_days=3, data_path="stock_data_dict.pkl"):
    """
    Compute pairwise shifted correlations for all ticker pairs.
    Returns a DataFrame with columns [Ticker1, Ticker2, ShiftDays, Correlation].
    """
    corr_tool = ShiftedCorrelation(data_path)
    results = []

    for t1, t2 in itertools.combinations(tickers, 2):
        try:
            corr = corr_tool.compute(t1, t2, shift_days)
            results.append({
                "Ticker1": t1,
                "Ticker2": t2,
                "ShiftDays": shift_days,
                "Correlation": corr
            })
            print(f"{t1}-{t2}: {corr:.4f}")
        except Exception as e:
            print(f"⚠️ Error computing {t1}-{t2}: {e}")

    df = pd.DataFrame(results)
    return df


if __name__ == "__main__":
    # You can reuse the same tickers from fetch_yahoo_data.py
    tickers = [
        "AAPL","MSFT","GOOGL","AMZN","NVDA","META","TSLA","JPM","UNH","V","HD",
        "PG","MA","LLY","AVGO","XOM","COST","PEP","KO","PFE","MRK","ORCL","ABBV",
        "DIS","CVX","ADBE","MCD","ACN","DHR","CSCO","NFLX","CRM","LIN","ABT",
        "TXN","NKE","TMO","INTC","WMT","UPS","HON","NEE","PM","AMGN","AMD",
        "QCOM","LOW","CAT","IBM","MS","AMT","INTU","DE","GS","SPGI","RTX",
        "BLK","ISRG","PLD","NOW","MDT","GE","ADI","SYK","ELV","BKNG","GILD",
        "C","CVS","LRCX","ADP","REGN","MO","MDLZ","TMUS","T","CB","SCHW",
        "SO","PGR","AXP","ZTS","CL","DUK","EOG","USB","TGT","PNC","MMC",
        "FIS","BDX","CME","ITW","GM","CSX","NSC","AON","EMR","COF","FDX",
        "ETN","PSA"
    ][:100]

    shift_days = 3  # example shift
    df = compute_all_shifted_correlations(tickers, shift_days=shift_days)

    # Save results
    df.to_csv(f"shifted_correlations_{shift_days}d.csv", index=False)
    df.to_pickle(f"shifted_correlations_{shift_days}d.pkl")

    print(f"\n✅ Saved correlations for {len(df)} pairs to shifted_correlations_{shift_days}d.csv")
