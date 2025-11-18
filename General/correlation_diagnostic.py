import pickle
import pandas as pd
import numpy as np

def diagnose_stock_data(pickle_file='stock_data_dict.pkl'):
    """Run comprehensive diagnostics on the stock data"""
    
    print("🔍 STOCK DATA DIAGNOSTICS")
    print("=" * 60)
    
    # Load the data
    try:
        with open(pickle_file, 'rb') as f:
            data_dict = pickle.load(f)
        print(f"✅ Successfully loaded {pickle_file}")
    except Exception as e:
        print(f"❌ Error loading file: {e}")
        return
    
    print(f"\n📊 BASIC INFO:")
    print(f"   - Number of stocks: {len(data_dict)}")
    print(f"   - Type of container: {type(data_dict)}")
    
    if len(data_dict) == 0:
        print("❌ The dictionary is empty!")
        return
    
    print(f"\n📈 EXAMINING EACH STOCK:")
    print("-" * 60)
    
    issues = []
    good_stocks = []
    
    for i, (ticker, df) in enumerate(data_dict.items()):
        print(f"\n{i+1}. {ticker}:")
        print(f"   - Type: {type(df)}")
        
        if not isinstance(df, pd.DataFrame):
            print(f"   ❌ Not a DataFrame!")
            issues.append(f"{ticker}: Not a DataFrame")
            continue
            
        print(f"   - Shape: {df.shape}")
        print(f"   - Date range: {df.index[0]} to {df.index[-1]}" if len(df) > 0 else "   - EMPTY DATAFRAME")
        print(f"   - Columns: {list(df.columns)}")
        
        # Handle multi-level columns
        if isinstance(df.columns, pd.MultiIndex):
            print(f"   ⚠️  Multi-level columns detected!")
            # Try to get Close column
            close_prices = None
            if ('Close', ticker) in df.columns:
                close_prices = df[('Close', ticker)]
                close_col = ('Close', ticker)
            elif 'Close' in [col[0] for col in df.columns]:
                # Find any Close column
                close_cols = [col for col in df.columns if col[0] == 'Close']
                if close_cols:
                    close_prices = df[close_cols[0]]
                    close_col = close_cols[0]
        else:
            # Single-level columns
            has_close = 'Close' in df.columns
            has_adj_close = 'Adj Close' in df.columns
            
            if has_close:
                close_prices = df['Close']
                close_col = 'Close'
            elif has_adj_close:
                close_prices = df['Adj Close']
                close_col = 'Adj Close'
            else:
                close_prices = None
                close_col = None
        
        if close_prices is None:
            print(f"   ❌ Cannot find Close/Adj Close column!")
            issues.append(f"{ticker}: No accessible Close column")
            continue
        
        print(f"   - Using column: {close_col}")
        print(f"   - Non-null values: {close_prices.notna().sum()}/{len(close_prices)}")
        print(f"   - Null values: {close_prices.isna().sum()}")
        
        if close_prices.notna().sum() == 0:
            print(f"   ❌ All values are NaN!")
            issues.append(f"{ticker}: All NaN values")
            continue
            
        # Show sample values
        valid_prices = close_prices.dropna()
        if len(valid_prices) > 0:
            print(f"   - Price range: ${valid_prices.min():.2f} - ${valid_prices.max():.2f}")
            print(f"   - Mean price: ${valid_prices.mean():.2f}")
            print(f"   - First few prices: {valid_prices.head(3).values}")
            good_stocks.append(ticker)
        
        # Check data type
        print(f"   - Data type: {close_prices.dtype}")
        
        # Check index
        print(f"   - Index type: {type(df.index)}")
        print(f"   - Index name: {df.index.name}")
        
        # Show first few rows
        if len(df) > 0 and i < 3:  # Show details for first 3 stocks only
            print(f"   - First 2 rows:")
            print(df.head(2).to_string(max_cols=6))
    
    print("\n" + "=" * 60)
    print("📋 SUMMARY:")
    print(f"   - Good stocks with data: {len(good_stocks)}")
    print(f"   - Problematic stocks: {len(issues)}")
    
    if issues:
        print(f"\n❌ ISSUES FOUND:")
        for issue in issues[:10]:  # Show first 10 issues
            print(f"   - {issue}")
        if len(issues) > 10:
            print(f"   ... and {len(issues)-10} more issues")
    
    if good_stocks:
        print(f"\n✅ STOCKS WITH VALID DATA (first 20):")
        print(f"   {', '.join(good_stocks[:20])}")
    
    # Test correlation calculation on first two good stocks
    if len(good_stocks) >= 2:
        print(f"\n🧪 TEST CORRELATION CALCULATION:")
        print(f"   Testing {good_stocks[0]} vs {good_stocks[1]}")
        
        try:
            df1 = data_dict[good_stocks[0]]
            df2 = data_dict[good_stocks[1]]
            
            # Extract Close prices handling multi-level columns
            if isinstance(df1.columns, pd.MultiIndex):
                s1 = df1[('Close', good_stocks[0])].dropna() if ('Close', good_stocks[0]) in df1.columns else df1.iloc[:, 0].dropna()
            else:
                s1 = df1['Close' if 'Close' in df1.columns else 'Adj Close'].dropna()
                
            if isinstance(df2.columns, pd.MultiIndex):
                s2 = df2[('Close', good_stocks[1])].dropna() if ('Close', good_stocks[1]) in df2.columns else df2.iloc[:, 0].dropna()
            else:
                s2 = df2['Close' if 'Close' in df2.columns else 'Adj Close'].dropna()
            
            # Align the series
            combined = pd.DataFrame({'s1': s1, 's2': s2}).dropna()
            
            if len(combined) > 0:
                corr = combined['s1'].corr(combined['s2'])
                print(f"   ✅ Correlation calculated successfully: {corr:.3f}")
                print(f"   - Overlapping dates: {len(combined)}")
            else:
                print(f"   ❌ No overlapping dates between the two stocks!")
                
        except Exception as e:
            print(f"   ❌ Error calculating correlation: {e}")

if __name__ == "__main__":
    diagnose_stock_data()