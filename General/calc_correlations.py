import pickle
import pandas as pd
import numpy as np
from itertools import combinations
from datetime import timedelta
import warnings
warnings.filterwarnings('ignore')

def load_stock_data(pickle_file):
    """Load the stock data dictionary from pickle file"""
    with open(pickle_file, 'rb') as f:
        return pickle.load(f)

def prepare_price_series(stock_df, ticker=None):
    """Extract and prepare the closing price series"""
    # Handle multi-level columns (from yfinance multiple ticker download)
    if isinstance(stock_df.columns, pd.MultiIndex):
        # Look for Close column with ticker
        if ticker and ('Close', ticker) in stock_df.columns:
            return stock_df[('Close', ticker)].dropna()
        # Look for any Close column
        close_cols = [col for col in stock_df.columns if col[0] == 'Close']
        if close_cols:
            return stock_df[close_cols[0]].dropna()
        # If no Close, look for Adj Close
        adj_close_cols = [col for col in stock_df.columns if col[0] == 'Adj Close']
        if adj_close_cols:
            return stock_df[adj_close_cols[0]].dropna()
    else:
        # Single-level columns
        if 'Close' in stock_df.columns:
            return stock_df['Close'].dropna()
        elif 'Adj Close' in stock_df.columns:
            return stock_df['Adj Close'].dropna()
    
    raise ValueError("No Close or Adj Close column found")

def calculate_shifted_correlation(series1, series2, shift_days=30):
    """
    Calculate correlation with time shift.
    Positive shift: series2 is shifted forward (future)
    Negative shift: series2 is shifted backward (past)
    """
    # Align the series first
    combined = pd.DataFrame({
        's1': series1,
        's2': series2
    })
    
    # Resample to daily frequency to ensure alignment
    combined = combined.resample('D').last().dropna()
    
    if shift_days > 0:
        # Causal: series1 today vs series2 future
        s1_aligned = combined['s1'].iloc[:-shift_days]
        s2_aligned = combined['s2'].iloc[shift_days:]
    elif shift_days < 0:
        # Acausal: series1 today vs series2 past
        s1_aligned = combined['s1'].iloc[-shift_days:]
        s2_aligned = combined['s2'].iloc[:shift_days]
    else:
        # No shift
        s1_aligned = combined['s1']
        s2_aligned = combined['s2']
    
    # Make sure indices align
    if shift_days != 0:
        s2_aligned.index = s1_aligned.index
    
    # Need at least 30 data points for meaningful correlation
    if len(s1_aligned) < 30:
        return np.nan
    
    return s1_aligned.corr(s2_aligned)

def calculate_all_correlations(data_dict, shift_days=30):
    """
    Calculate all three correlations for every pair of stocks:
    1. Standard correlation (no shift)
    2. Causal shift (stock1 vs stock2 shifted +1 month forward)
    3. Acausal shift (stock1 vs stock2 shifted -1 month backward)
    """
    
    results = []
    ticker_list = list(data_dict.keys())
    total_pairs = len(list(combinations(ticker_list, 2)))
    
    print(f"📊 Calculating correlations for {len(ticker_list)} stocks ({total_pairs} pairs)...")
    print(f"   This will generate 3 correlation values per pair:")
    print(f"   - Standard correlation (same time)")
    print(f"   - Causal (Stock A → Stock B after {shift_days} days)")
    print(f"   - Acausal (Stock A ← Stock B before {shift_days} days)\n")
    
    processed = 0
    
    for ticker1, ticker2 in combinations(ticker_list, 2):
        try:
            # Prepare price series (passing ticker names for multi-level columns)
            series1 = prepare_price_series(data_dict[ticker1], ticker1)
            series2 = prepare_price_series(data_dict[ticker2], ticker2)
            
            # Calculate three correlations
            corr_standard = calculate_shifted_correlation(series1, series2, shift_days=0)
            corr_causal = calculate_shifted_correlation(series1, series2, shift_days=shift_days)
            corr_acausal = calculate_shifted_correlation(series1, series2, shift_days=-shift_days)
            
            results.append({
                'stock1': ticker1,
                'stock2': ticker2,
                'correlation': corr_standard,
                'causal_shift': corr_causal,  # stock1 predicts stock2
                'acausal_shift': corr_acausal  # stock2 predicts stock1
            })
            
            processed += 1
            if processed % 100 == 0:
                print(f"   Processed {processed}/{total_pairs} pairs...")
                
        except Exception as e:
            print(f"⚠️  Error processing {ticker1}-{ticker2}: {e}")
            results.append({
                'stock1': ticker1,
                'stock2': ticker2,
                'correlation': np.nan,
                'causal_shift': np.nan,
                'acausal_shift': np.nan
            })
    
    return pd.DataFrame(results)

def analyze_correlations(corr_df):
    """Provide summary statistics and find interesting patterns"""
    print("\n📈 CORRELATION ANALYSIS SUMMARY")
    print("=" * 50)
    
    # Basic stats
    print("\n1. BASIC STATISTICS:")
    print("-" * 30)
    print(corr_df[['correlation', 'causal_shift', 'acausal_shift']].describe())
    
    # Find highest correlations
    print("\n2. TOP 10 HIGHEST STANDARD CORRELATIONS:")
    print("-" * 30)
    top_corr = corr_df.nlargest(10, 'correlation')[['stock1', 'stock2', 'correlation']]
    for _, row in top_corr.iterrows():
        print(f"   {row['stock1']:6} ↔ {row['stock2']:6} : {row['correlation']:.3f}")
    
    # Find strongest causal relationships
    print("\n3. TOP 10 CAUSAL RELATIONSHIPS (Stock1 → Stock2):")
    print("-" * 30)
    top_causal = corr_df.nlargest(10, 'causal_shift')[['stock1', 'stock2', 'causal_shift']]
    for _, row in top_causal.iterrows():
        print(f"   {row['stock1']:6} → {row['stock2']:6} : {row['causal_shift']:.3f}")
    
    # Find stocks with asymmetric relationships
    print("\n4. MOST ASYMMETRIC RELATIONSHIPS:")
    print("   (Large difference between causal and acausal)")
    print("-" * 30)
    corr_df['asymmetry'] = abs(corr_df['causal_shift'] - corr_df['acausal_shift'])
    top_asymmetric = corr_df.nlargest(10, 'asymmetry')[['stock1', 'stock2', 'causal_shift', 'acausal_shift', 'asymmetry']]
    for _, row in top_asymmetric.iterrows():
        print(f"   {row['stock1']:6} & {row['stock2']:6} : Causal={row['causal_shift']:.3f}, Acausal={row['acausal_shift']:.3f}, Diff={row['asymmetry']:.3f}")
    
    # Find leading indicators
    print("\n5. POTENTIAL LEADING INDICATORS:")
    print("   (Stocks that predict others better than being predicted)")
    print("-" * 30)
    
    # Calculate average causal vs acausal for each stock
    leader_scores = {}
    for ticker in set(corr_df['stock1'].unique()) | set(corr_df['stock2'].unique()):
        # When ticker is stock1 (predicting others)
        as_predictor = corr_df[corr_df['stock1'] == ticker]['causal_shift'].mean()
        # When ticker is stock2 (being predicted by others)
        being_predicted = corr_df[corr_df['stock2'] == ticker]['acausal_shift'].mean()
        
        if pd.notna(as_predictor) and pd.notna(being_predicted):
            leader_scores[ticker] = as_predictor - being_predicted
    
    sorted_leaders = sorted(leader_scores.items(), key=lambda x: x[1], reverse=True)[:10]
    for ticker, score in sorted_leaders:
        print(f"   {ticker:6} : Leadership Score = {score:.3f}")

def main():
    # Load the data
    print("🚀 Starting Correlation Analysis")
    print("=" * 50)
    
    try:
        data_dict = load_stock_data('stock_data_dict.pkl')
        print(f"✅ Loaded data for {len(data_dict)} stocks\n")
    except FileNotFoundError:
        print("❌ Error: stock_data_dict.pkl not found!")
        print("   Please run the data fetching script first.")
        return
    
    # Calculate correlations
    correlations_df = calculate_all_correlations(data_dict, shift_days=30)
    
    # Save results
    output_file = 'stock_correlations.pkl'
    correlations_df.to_pickle(output_file)
    print(f"\n✅ Saved correlation data to {output_file}")
    
    # Also save as CSV for easy viewing
    csv_file = 'stock_correlations.csv'
    correlations_df.to_csv(csv_file, index=False)
    print(f"✅ Also saved as CSV to {csv_file}")
    
    # Analyze and display insights
    analyze_correlations(correlations_df)
    
    print("\n" + "=" * 50)
    print("🎯 COMPLETE! Your correlation dataset is ready.")
    print(f"   - Total pairs analyzed: {len(correlations_df)}")
    print(f"   - Metrics per pair: 3 (standard, causal, acausal)")
    print(f"   - Files created: {output_file}, {csv_file}")

if __name__ == "__main__":
    main()