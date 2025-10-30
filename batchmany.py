import itertools
import pandas as pd
from shifted_correlation import ShiftedCorrelation

def compute_all_shifted_correlations(tickers, shift_days_list=[0, 10, 20, 30], data_path="stock_data_dict.pkl"):
    """
    Compute pairwise shifted correlations for all ticker pairs across multiple shift periods.
    Returns a DataFrame with columns [Ticker1, Ticker2, ShiftDays, Correlation].
    """
    corr_tool = ShiftedCorrelation(data_path)
    results = []
    
    # Get all unique pairs
    ticker_pairs = list(itertools.combinations(tickers, 2))
    total_computations = len(ticker_pairs) * len(shift_days_list)
    completed = 0

    for shift_days in shift_days_list:
        print(f"\n{'='*50}")
        print(f"Computing correlations for {shift_days}-day shift")
        print(f"{'='*50}")
        
        for t1, t2 in ticker_pairs:
            try:
                corr = corr_tool.compute(t1, t2, shift_days)
                results.append({
                    "Ticker1": t1,
                    "Ticker2": t2,
                    "ShiftDays": shift_days,
                    "Correlation": corr
                })
                completed += 1
                
                # Progress indicator every 100 computations
                if completed % 100 == 0:
                    progress = (completed / total_computations) * 100
                    print(f"Progress: {completed}/{total_computations} ({progress:.1f}%)")
                    
            except Exception as e:
                print(f"⚠️ Error computing {t1}-{t2} (shift={shift_days}): {e}")
                # Still add a row with NaN for failed computations
                results.append({
                    "Ticker1": t1,
                    "Ticker2": t2,
                    "ShiftDays": shift_days,
                    "Correlation": None
                })
                completed += 1

    df = pd.DataFrame(results)
    return df


def analyze_correlations(df):
    """
    Provide summary statistics for the computed correlations.
    """
    print("\n" + "="*60)
    print("CORRELATION ANALYSIS SUMMARY")
    print("="*60)
    
    for shift in df['ShiftDays'].unique():
        shift_df = df[df['ShiftDays'] == shift]
        valid_corrs = shift_df['Correlation'].dropna()
        
        print(f"\n📊 {shift}-Day Shift Statistics:")
        print(f"  • Total pairs: {len(shift_df)}")
        print(f"  • Valid correlations: {len(valid_corrs)}")
        print(f"  • Failed computations: {len(shift_df) - len(valid_corrs)}")
        
        if len(valid_corrs) > 0:
            print(f"  • Mean correlation: {valid_corrs.mean():.4f}")
            print(f"  • Std deviation: {valid_corrs.std():.4f}")
            print(f"  • Min correlation: {valid_corrs.min():.4f}")
            print(f"  • Max correlation: {valid_corrs.max():.4f}")
            print(f"  • Median correlation: {valid_corrs.median():.4f}")
            
            # Find top 3 most correlated pairs for this shift
            top_3 = shift_df.nlargest(3, 'Correlation')
            print(f"\n  Top 3 correlations for {shift}-day shift:")
            for _, row in top_3.iterrows():
                print(f"    {row['Ticker1']}-{row['Ticker2']}: {row['Correlation']:.4f}")


if __name__ == "__main__":
    # Define tickers
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

    # Define shift periods to compute
    shift_days_list = [0, 10, 20, 30]
    
    print(f"🚀 Starting batch correlation computation")
    print(f"📊 Tickers: {len(tickers)}")
    print(f"📈 Unique pairs: {len(list(itertools.combinations(tickers, 2)))}")
    print(f"⏰ Shift periods: {shift_days_list}")
    print(f"🔢 Total computations: {len(list(itertools.combinations(tickers, 2))) * len(shift_days_list)}")
    
    # Compute all correlations
    df = compute_all_shifted_correlations(tickers, shift_days_list=shift_days_list)
    
    # Save results in multiple formats
    csv_filename = "all_shifted_correlations.csv"
    pkl_filename = "all_shifted_correlations.pkl"
    excel_filename = "all_shifted_correlations.xlsx"
    
    # Save as CSV
    df.to_csv(csv_filename, index=False)
    print(f"\n✅ Saved to {csv_filename}")
    
    # Save as pickle for faster loading
    df.to_pickle(pkl_filename)
    print(f"✅ Saved to {pkl_filename}")
    
    # Save as Excel with each shift period in a separate sheet
    with pd.ExcelWriter(excel_filename) as writer:
        # Save all data in one sheet
        df.to_excel(writer, sheet_name='All_Shifts', index=False)
        
        # Save each shift period in its own sheet
        for shift in shift_days_list:
            shift_df = df[df['ShiftDays'] == shift]
            sheet_name = f'Shift_{shift}d'
            shift_df.to_excel(writer, sheet_name=sheet_name, index=False)
    print(f"✅ Saved to {excel_filename} (with separate sheets for each shift)")
    
    # Analyze and display summary statistics
    analyze_correlations(df)
    
    # Create a pivot table for easier analysis
    pivot_df = df.pivot_table(
        index=['Ticker1', 'Ticker2'], 
        columns='ShiftDays', 
        values='Correlation',
        aggfunc='first'
    )
    pivot_df.to_csv("correlation_pivot_table.csv")
    print(f"\n✅ Saved pivot table to correlation_pivot_table.csv")
    
    print(f"\n🎉 Batch correlation computation complete!")
    print(f"📁 Output files:")
    print(f"   • {csv_filename} - All correlations in CSV format")
    print(f"   • {pkl_filename} - All correlations in pickle format (fast loading)")
    print(f"   • {excel_filename} - Excel with separate sheets per shift period")
    print(f"   • correlation_pivot_table.csv - Pivot table format")