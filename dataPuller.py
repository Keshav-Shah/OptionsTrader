import yfinance as yf
import datetime
import pickle

def fetch_stock_data(tickers, period_years=3):
    """
    Fetch daily Yahoo Finance data for the given tickers over the past `period_years` years.
    Returns a dictionary: {ticker: pandas.DataFrame}.
    """
    end = datetime.datetime.today()
    start = end - datetime.timedelta(days=365 * period_years)
    data_dict = {}

    for ticker in tickers:
        try:
            print(f"Fetching {ticker}...")
            df = yf.download(ticker, start=start, end=end, interval="1d", progress=False)
            if not df.empty:
                data_dict[ticker] = df
            else:
                print(f"⚠️ No data found for {ticker}")
        except Exception as e:
            print(f"❌ Error fetching {ticker}: {e}")
    
    return data_dict


if __name__ == "__main__":
    # Example: first 100 S&P 500 tickers
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

    data = fetch_stock_data(tickers, period_years=1)

    # Save as pickle
    with open("stock_data_dict.pkl", "wb") as f:
        pickle.dump(data, f)
    
    print(f"✅ Saved data for {len(data)} tickers to stock_data_dict.pkl")
