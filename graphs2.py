"""
Enhanced Stock Temporal Correlation Analysis
============================================
Discovers hidden lead-lag relationships between stocks through shifted correlation analysis.
Features advanced visualization with color-coded correlation strength and comprehensive metrics.
"""

import itertools
import pickle
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.patches import FancyBboxPatch
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set style for beautiful visualizations
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class TemporalCorrelationAnalyzer:
    """
    Advanced correlation analyzer for discovering temporal lead-lag relationships
    between stocks using shifted time series analysis.
    """
    
    def __init__(self, data_path="stock_data_dict.pkl"):
        """Initialize with stock data dictionary."""
        print("🚀 Initializing Temporal Correlation Analyzer...")
        with open(data_path, "rb") as f:
            self.data_dict = pickle.load(f)
        self.validate_data()
        print(f"✅ Loaded data for {len(self.data_dict)} stocks")
    
    def validate_data(self):
        """Validate that all stocks have required data."""
        invalid_tickers = []
        for ticker, df in self.data_dict.items():
            try:
                if isinstance(df, pd.DataFrame):
                    if "Close" not in df.columns:
                        invalid_tickers.append(ticker)
                    elif df["Close"].isna().values.all():  # Use .values.all() to get a proper boolean
                        invalid_tickers.append(ticker)
                else:
                    invalid_tickers.append(ticker)
            except Exception as e:
                print(f"⚠️ Error validating {ticker}: {e}")
                invalid_tickers.append(ticker)
        if invalid_tickers:
            print(f"⚠️ Warning: {len(invalid_tickers)} tickers have invalid data: {invalid_tickers[:5]}...")
    
    def get_close_prices(self, ticker):
        """Extract close prices with error handling."""
        if ticker not in self.data_dict:
            raise ValueError(f"Ticker '{ticker}' not found in dataset")
        df = self.data_dict[ticker]
        if "Close" not in df.columns:
            raise ValueError(f"No 'Close' column for {ticker}")
        return df["Close"].dropna()
    
    def compute_shifted_correlation(self, ticker1, ticker2, shift_days):
        """
        Compute correlation between ticker1(t) and ticker2(t+shift).
        
        Positive shift: ticker1 leads ticker2 by 'shift_days'
        High correlation with positive shift → ticker1 predicts ticker2
        """
        try:
            s1 = self.get_close_prices(ticker1)
            s2 = self.get_close_prices(ticker2)
            
            # Align series and apply shift
            df = pd.concat([s1, s2], axis=1, join="inner")
            df.columns = [ticker1, ticker2]
            
            # Shift ticker2 forward (positive shift means ticker1 leads)
            df[f"{ticker2}_shifted"] = df[ticker2].shift(shift_days)
            df = df.dropna()
            
            # Require minimum sample size for reliability
            if len(df) < 30:
                return np.nan
            
            # Calculate correlation
            correlation = df[ticker1].corr(df[f"{ticker2}_shifted"])
            
            # Also calculate p-value for statistical significance
            from scipy import stats
            _, p_value = stats.pearsonr(df[ticker1], df[f"{ticker2}_shifted"])
            
            return correlation if p_value < 0.05 else np.nan  # Only return significant correlations
            
        except Exception as e:
            return np.nan
    
    def compute_batch_correlations(self, tickers, shifts):
        """
        Compute all pairwise correlations for multiple shift values.
        Returns separate DataFrames for each shift for better organization.
        """
        print("\n📊 Computing Temporal Correlations")
        print("=" * 50)
        
        all_results = {}
        total_pairs = len(list(itertools.combinations(tickers, 2)))
        
        for shift in shifts:
            print(f"\n🔄 Processing shift: {shift} days")
            results = []
            
            for i, (t1, t2) in enumerate(itertools.combinations(tickers, 2), 1):
                corr = self.compute_shifted_correlation(t1, t2, shift)
                
                # Also compute reverse direction
                corr_reverse = self.compute_shifted_correlation(t2, t1, shift)
                
                # Store both directions
                results.append({
                    "Leader": t1,
                    "Follower": t2,
                    "ShiftDays": shift,
                    "Correlation": corr,
                    "Direction": f"{t1}→{t2}"
                })
                
                results.append({
                    "Leader": t2,
                    "Follower": t1,
                    "ShiftDays": shift,
                    "Correlation": corr_reverse,
                    "Direction": f"{t2}→{t1}"
                })
                
                # Progress indicator
                if i % 50 == 0 or i == total_pairs:
                    print(f"  Progress: {i}/{total_pairs} pairs processed", end='\r')
            
            df_shift = pd.DataFrame(results)
            df_shift = df_shift[df_shift["Correlation"].notna()]
            all_results[shift] = df_shift
            
            print(f"  ✅ Shift {shift}d: Found {len(df_shift)} significant correlations")
        
        return all_results

class NetworkVisualizer:
    """
    Advanced network visualization for temporal stock relationships.
    """
    
    def __init__(self, correlation_data, beta_thresholds):
        """
        Initialize visualizer with correlation data.
        
        Parameters:
        -----------
        correlation_data : dict
            Dictionary with shift values as keys and correlation DataFrames as values
        beta_thresholds : dict
            Dictionary with shift values as keys and threshold values
        """
        self.correlation_data = correlation_data
        self.beta_thresholds = beta_thresholds
    
    def build_directed_graph(self, shift, min_correlation=0.7):
        """
        Build a directed graph from correlation data.
        Edge direction: Leader → Follower
        """
        G = nx.DiGraph()
        df = self.correlation_data[shift]
        
        # Filter by correlation threshold
        strong_correlations = df[df["Correlation"].abs() >= min_correlation]
        
        # Add edges with correlation as weight
        for _, row in strong_correlations.iterrows():
            G.add_edge(
                row["Leader"],
                row["Follower"],
                weight=row["Correlation"],
                shift=shift
            )
        
        # Calculate node statistics
        for node in G.nodes():
            G.nodes[node]['out_degree'] = G.out_degree(node)
            G.nodes[node]['in_degree'] = G.in_degree(node)
            G.nodes[node]['influence_score'] = G.out_degree(node) - G.in_degree(node)
        
        return G
    
    def get_edge_colors(self, G, cmap='RdYlGn'):
        """
        Generate edge colors based on correlation strength.
        """
        weights = [G[u][v]['weight'] for u, v in G.edges()]
        if not weights:
            return []
        
        # Normalize weights to [0, 1]
        vmin, vmax = min(weights), max(weights)
        if vmax == vmin:
            norm_weights = [0.5] * len(weights)
        else:
            norm_weights = [(w - vmin) / (vmax - vmin) for w in weights]
        
        # Get colors from colormap
        colormap = cm.get_cmap(cmap)
        edge_colors = [colormap(w) for w in norm_weights]
        
        return edge_colors, weights
    
    def calculate_layout(self, G, layout_type='spring'):
        """
        Calculate optimal layout based on graph properties.
        """
        if layout_type == 'spring':
            # Use influence score for better positioning
            pos = nx.spring_layout(G, k=2/np.sqrt(len(G.nodes())), 
                                 iterations=50, seed=42)
        elif layout_type == 'circular':
            # Sort nodes by influence for circular layout
            sorted_nodes = sorted(G.nodes(), 
                                key=lambda x: G.nodes[x].get('influence_score', 0),
                                reverse=True)
            pos = nx.circular_layout(sorted_nodes)
        elif layout_type == 'hierarchical':
            # Create hierarchical layout based on influence
            pos = nx.spring_layout(G, k=3, iterations=100)
            # Adjust y-position based on influence score
            for node in pos:
                influence = G.nodes[node].get('influence_score', 0)
                pos[node][1] += influence * 0.1
        else:
            pos = nx.kamada_kawai_layout(G)
        
        return pos
    
    def visualize_three_graphs(self, shifts, save_path=None):
        """
        Create comprehensive visualization with three correlation graphs.
        """
        fig = plt.figure(figsize=(24, 8))
        fig.suptitle('🔮 Hidden Temporal Relationships in Stock Market\nLead-Lag Correlation Analysis', 
                     fontsize=16, fontweight='bold', y=1.02)
        
        # Create three subplots with dividers
        gs = fig.add_gridspec(1, 3, hspace=0.3, wspace=0.3)
        
        graphs_data = []
        
        for idx, shift in enumerate(shifts):
            ax = fig.add_subplot(gs[0, idx])
            
            # Build graph for this shift
            threshold = self.beta_thresholds.get(shift, 0.7)
            G = self.build_directed_graph(shift, threshold)
            graphs_data.append(G)
            
            if G.number_of_nodes() == 0:
                ax.text(0.5, 0.5, f'No correlations > {threshold}', 
                       ha='center', va='center', fontsize=12)
                ax.set_title(f'📅 {shift}-Day Shift\n(β > {threshold})', 
                           fontsize=12, fontweight='bold')
                ax.axis('off')
                continue
            
            # Calculate layout
            pos = self.calculate_layout(G, 'spring')
            
            # Get edge colors based on correlation strength
            edge_colors, weights = self.get_edge_colors(G)
            
            # Draw the network
            # Draw nodes with size based on influence
            node_sizes = [300 + G.nodes[node]['influence_score'] * 100 
                         for node in G.nodes()]
            node_colors = [G.nodes[node]['influence_score'] 
                          for node in G.nodes()]
            
            nodes = nx.draw_networkx_nodes(
                G, pos, ax=ax,
                node_size=node_sizes,
                node_color=node_colors,
                cmap='coolwarm',
                vmin=-5, vmax=5,
                alpha=0.9,
                edgecolors='black',
                linewidths=1.5
            )
            
            # Draw edges with colors based on correlation
            if edge_colors:
                nx.draw_networkx_edges(
                    G, pos, ax=ax,
                    edge_color=edge_colors,
                    width=[abs(w) * 3 for w in weights],
                    alpha=0.7,
                    arrows=True,
                    arrowsize=12,
                    arrowstyle='->',
                    connectionstyle='arc3,rad=0.1',
                    node_size=node_sizes
                )
            
            # Draw labels
            nx.draw_networkx_labels(
                G, pos, ax=ax,
                font_size=9,
                font_weight='bold',
                font_color='black'
            )
            
            # Add title and statistics
            ax.set_title(
                f'📅 {shift}-Day Shift Analysis\n'
                f'Threshold: β > {threshold} | '
                f'Nodes: {G.number_of_nodes()} | '
                f'Edges: {G.number_of_edges()}',
                fontsize=11,
                fontweight='bold',
                pad=10
            )
            
            # Add a box around each subplot
            rect = FancyBboxPatch(
                (0.02, 0.02), 0.96, 0.96,
                boxstyle="round,pad=0.01",
                transform=ax.transAxes,
                facecolor='none',
                edgecolor='gray',
                linewidth=2,
                alpha=0.3
            )
            ax.add_patch(rect)
            
            ax.axis('off')
            
            # Add colorbar for edge weights
            if idx == 2 and edge_colors:  # Add colorbar to the last subplot
                sm = plt.cm.ScalarMappable(
                    cmap='RdYlGn',
                    norm=plt.Normalize(vmin=min(weights), vmax=max(weights))
                )
                sm.set_array([])
                cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
                cbar.set_label('Correlation Strength', fontsize=10)
        
        # Add legend for node colors
        legend_ax = fig.add_axes([0.92, 0.7, 0.07, 0.2])
        legend_ax.axis('off')
        legend_ax.text(0.5, 0.9, '🎯 Node Influence', 
                      ha='center', fontweight='bold', fontsize=10)
        legend_ax.text(0.5, 0.7, '🔴 Leader\n(High out-degree)', 
                      ha='center', fontsize=9, color='darkred')
        legend_ax.text(0.5, 0.4, '⚪ Neutral', 
                      ha='center', fontsize=9, color='gray')
        legend_ax.text(0.5, 0.1, '🔵 Follower\n(High in-degree)', 
                      ha='center', fontsize=9, color='darkblue')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"💾 Visualization saved to {save_path}")
        
        plt.show()
        
        return graphs_data

class CorrelationAnalysisReport:
    """
    Generate comprehensive analysis reports for correlation findings.
    """
    
    def __init__(self, correlation_data, graphs):
        self.correlation_data = correlation_data
        self.graphs = graphs
    
    def identify_key_leaders(self, G, top_n=5):
        """Identify stocks that consistently lead others."""
        influence_scores = {
            node: G.nodes[node]['influence_score'] 
            for node in G.nodes()
        }
        return sorted(influence_scores.items(), 
                     key=lambda x: x[1], reverse=True)[:top_n]
    
    def identify_key_followers(self, G, top_n=5):
        """Identify stocks that consistently follow others."""
        influence_scores = {
            node: G.nodes[node]['influence_score'] 
            for node in G.nodes()
        }
        return sorted(influence_scores.items(), 
                     key=lambda x: x[1])[:top_n]
    
    def find_strongest_relationships(self, shift, top_n=10):
        """Find the strongest correlation pairs for a given shift."""
        df = self.correlation_data[shift]
        return df.nlargest(top_n, 'Correlation')[['Leader', 'Follower', 'Correlation']]
    
    def generate_report(self, shifts):
        """Generate a comprehensive analysis report."""
        print("\n" + "="*60)
        print("📈 TEMPORAL CORRELATION ANALYSIS REPORT")
        print("="*60)
        
        for shift, G in zip(shifts, self.graphs):
            print(f"\n🔍 {shift}-DAY SHIFT ANALYSIS")
            print("-" * 40)
            
            if G.number_of_nodes() == 0:
                print("No significant correlations found at this shift.")
                continue
            
            # Key leaders
            print("\n🎯 Top Market Leaders (Predictive Stocks):")
            leaders = self.identify_key_leaders(G)
            for i, (stock, score) in enumerate(leaders, 1):
                print(f"  {i}. {stock:6} | Influence Score: {score:+.1f}")
            
            # Key followers
            print("\n📊 Top Market Followers (Reactive Stocks):")
            followers = self.identify_key_followers(G)
            for i, (stock, score) in enumerate(followers, 1):
                print(f"  {i}. {stock:6} | Influence Score: {score:+.1f}")
            
            # Strongest relationships
            print("\n💪 Strongest Lead-Lag Relationships:")
            strong_pairs = self.find_strongest_relationships(shift, 5)
            for i, row in enumerate(strong_pairs.itertuples(), 1):
                print(f"  {i}. {row.Leader} → {row.Follower} | β = {row.Correlation:.3f}")
            
            # Network statistics
            print(f"\n📊 Network Statistics:")
            if G.number_of_nodes() > 0:
                density = nx.density(G)
                print(f"  • Network Density: {density:.3f}")
                
                if G.number_of_nodes() > 1:
                    try:
                        components = list(nx.weakly_connected_components(G))
                        print(f"  • Connected Components: {len(components)}")
                        
                        if nx.is_weakly_connected(G):
                            diameter = nx.diameter(G.to_undirected())
                            print(f"  • Network Diameter: {diameter}")
                    except:
                        pass
                
                # Average clustering
                try:
                    clustering = nx.average_clustering(G.to_undirected())
                    print(f"  • Average Clustering: {clustering:.3f}")
                except:
                    pass

def main():
    """
    Main execution function for the enhanced correlation analysis.
    """
    # Configuration
    TICKERS = [
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
    ]
    
    SHIFTS = [5, 15, 25]  # Days to shift for lead-lag analysis
    
    # Different thresholds for different time horizons
    BETA_THRESHOLDS = {
        5: 0.80,   # Higher threshold for short-term
        15: 0.75,  # Medium threshold for medium-term
        25: 0.70   # Lower threshold for long-term (harder to maintain)
    }
    
    print("🌟 ENHANCED STOCK TEMPORAL CORRELATION ANALYZER")
    print("=" * 60)
    print(f"Analyzing {len(TICKERS)} stocks across {len(SHIFTS)} time shifts")
    print(f"Looking for hidden lead-lag relationships...")
    print("=" * 60)
    
    # Initialize analyzer
    analyzer = TemporalCorrelationAnalyzer("stock_data_dict.pkl")
    
    # Compute correlations
    start_time = datetime.now()
    correlation_results = analyzer.compute_batch_correlations(TICKERS, SHIFTS)
    computation_time = (datetime.now() - start_time).total_seconds()
    
    print(f"\n⏱️ Analysis completed in {computation_time:.1f} seconds")
    
    # Save correlation data
    print("\n💾 Saving Correlation Data...")
    for shift, df in correlation_results.items():
        filename = f"correlation_shift_{shift}days.csv"
        df.to_csv(filename, index=False)
        print(f"  ✅ Saved {len(df)} correlations to {filename}")
    
    # Create visualizations
    print("\n🎨 Creating Visualizations...")
    visualizer = NetworkVisualizer(correlation_results, BETA_THRESHOLDS)
    graphs = visualizer.visualize_three_graphs(
        SHIFTS, 
        save_path="temporal_correlation_analysis.png"
    )
    
    # Save network files for external analysis
    print("\n💾 Saving Network Files...")
    for shift, G in zip(SHIFTS, graphs):
        if G.number_of_nodes() > 0:
            # Save as GEXF for Gephi
            nx.write_gexf(G, f"correlation_network_{shift}days.gexf")
            print(f"  ✅ Saved network for {shift}-day shift to correlation_network_{shift}days.gexf")
            
            # Save as GraphML for other tools
            nx.write_graphml(G, f"correlation_network_{shift}days.graphml")
    
    # Generate analysis report
    report = CorrelationAnalysisReport(correlation_results, graphs)
    report.generate_report(SHIFTS)
    
    # Create summary DataFrame
    print("\n📊 Creating Summary Report...")
    summary_data = []
    for shift in SHIFTS:
        df = correlation_results[shift]
        summary_data.append({
            'Shift (days)': shift,
            'Total Correlations': len(df),
            'Mean Correlation': df['Correlation'].mean(),
            'Max Correlation': df['Correlation'].max(),
            'Significant (>0.7)': len(df[df['Correlation'] > 0.7]),
            'Very Strong (>0.85)': len(df[df['Correlation'] > 0.85])
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv("correlation_summary.csv", index=False)
    print("  ✅ Saved summary statistics to correlation_summary.csv")
    
    print("\n✨ Analysis Complete!")
    print("=" * 60)
    print("📁 Generated Files:")
    print("  • temporal_correlation_analysis.png - Main visualization")
    print("  • correlation_shift_*days.csv - Detailed correlations")
    print("  • correlation_network_*days.gexf - Network files")
    print("  • correlation_summary.csv - Summary statistics")
    print("=" * 60)

if __name__ == "__main__":
    main()