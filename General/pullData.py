

import yfinance as yf
import datetime
import pickle

def fetch_stock_data(tickers, period_years=5):
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
    # Original tech giants
    "AAPL","MSFT","GOOGL","AMZN","NVDA","META","TSLA","AVGO","ORCL","ADBE",
    "CRM","NFLX","AMD","INTC","QCOM","CSCO","IBM","INTU","NOW","ADI",
    
    # AI & Advanced Tech
    "PLTR","AI","SNOW","PATH","DDOG","NET","CRWD","ZS","OKTA","MDB",
    "PANW","FTNT","S","DOCU","TWLO","U","RBLX","COIN","SQ","PYPL",
    "ARM","SMCI","MRVL","MU","LRCX","KLAC","AMAT","ASML","TSM",
    
    # Utilities (including ITRI)
    "ITRI",  # Itron - your requested company
    "NEE","DUK","SO","D","AEP","SRE","EXC","XEL","ED","PEG",
    "WEC","ES","AWK","DTE","ETR","FE","ATO","CMS","CNP","NI",
    "EVRG","LNT","AES","PPL","NRG","VST","UGI","OGE","PNW",
    
    # More Tech/Software
    "SHOP","UBER","ABNB","DASH","SPOT","ROKU","ZM","TEAM","DBX",
    "PINS","SNAP","LYFT","ETSY","GDDY","WIX","FVRR","UPWK",
    
    # Semiconductor/Hardware
    "AVGO","QCOM","NVDA","AMD","INTC","TXN","ADI","MCHP","XLNX",
    "SWKS","QRVO","MPWR","ON","NXPI","STM","MXIM","LSCC",
    
    # Enterprise Software/Cloud
    "MSFT","ORCL","SAP","ADBE","CRM","NOW","WDAY","VEEV","SPLK",
    "HUBS","ZEN","BILL","GTLB","ESTC","SUMO","FROG","NCNO",
    
    # Cybersecurity
    "CRWD","PANW","FTNT","ZS","OKTA","PING","TENB","FEYE","RPD",
    
    # Electric/Clean Energy
    "TSLA","RIVN","LCID","NIO","XPEV","LI","FSR","PLUG","FCEL",
    "BLDP","BE","ENPH","SEDG","RUN","NOVA","SPWR","CSIQ"
    ]

    # Remove duplicates while preserving order
    tickers = list(dict.fromkeys(tickers))

    data = fetch_stock_data(tickers, period_years=5)

    # Save as pickle
    with open("stock_data_dict.pkl", "wb") as f:
        pickle.dump(data, f)
    
    print(f"✅ Saved data for {len(data)} tickers to stock_data_dict.pkl")


