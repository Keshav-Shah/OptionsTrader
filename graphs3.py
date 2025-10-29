"""
Simple High-Correlation Network Visualizer
==========================================
Focus on only the strongest correlations (>0.9) for clear insights
"""

import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import os

class SimpleCorrelationVisualizer:
    def __init__(self):
        self.shifts = [5, 15, 25]
        self.data = {}
        self.load_data()
    
    def load_data(self):
        """Load saved correlation files"""
        print("Loading correlation data...")
        for shift in self.shifts:
            filename = f"correlation_shift_{shift}days.csv"
            if os.path.exists(filename):
                self.data[shift] = pd.read_csv(filename)
                print(f"✓ Loaded {shift}-day correlations")
    
    def build_graph(self, shift, threshold):
        """Build graph with given threshold"""
        G = nx.DiGraph()
        df = self.data[shift]
        
        # Filter only strong positive correlations
        strong = df[df["Correlation"] >= threshold]
        
        for _, row in strong.iterrows():
            G.add_edge(row["Leader"], row["Follower"], 
                      weight=row["Correlation"])
        
        return G
    
    def visualize_simple(self, thresholds=[0.90, 0.92, 0.95]):
        """Create simple grid visualization"""
        
        # Create figure - much smaller
        fig, axes = plt.subplots(len(thresholds), 3, figsize=(12, 4*len(thresholds)))
        fig.suptitle('Strong Correlations Only (β > 0.9)', fontsize=14, fontweight='bold')
        
        for t_idx, threshold in enumerate(thresholds):
            for s_idx, shift in enumerate(self.shifts):
                ax = axes[t_idx][s_idx] if len(thresholds) > 1 else axes[s_idx]
                
                # Build graph
                G = self.build_graph(shift, threshold)
                
                # Title
                ax.set_title(f'{shift}d | β>{threshold:.2f} | {G.number_of_edges()} edges', 
                           fontsize=10)
                
                if G.number_of_nodes() == 0:
                    ax.text(0.5, 0.5, 'No correlations', ha='center', va='center')
                    ax.axis('off')
                    continue
                
                # Simple layout
                pos = nx.spring_layout(G, seed=42, k=1.5)
                
                # Draw network simply
                nx.draw_networkx_nodes(G, pos, ax=ax, 
                                     node_size=200, 
                                     node_color='lightblue',
                                     edgecolors='black',
                                     linewidths=0.5)
                
                # Draw edges with weights as thickness
                weights = [G[u][v]['weight'] for u, v in G.edges()]
                nx.draw_networkx_edges(G, pos, ax=ax,
                                      width=[w*2 for w in weights],
                                      alpha=0.6,
                                      arrows=True,
                                      arrowsize=8)
                
                # Labels
                nx.draw_networkx_labels(G, pos, ax=ax, font_size=7)
                
                ax.axis('off')
        
        plt.tight_layout()
        plt.savefig('high_correlations.png', dpi=100, bbox_inches='tight')
        plt.show()
        print("Saved: high_correlations.png")
    
    def find_strongest_pairs(self, min_threshold=0.9):
        """Find and display only the strongest relationships"""
        print("\n=== STRONGEST CORRELATIONS (>0.9) ===\n")
        
        all_strong = []
        
        for shift in self.shifts:
            df = self.data[shift]
            strong = df[df["Correlation"] >= min_threshold].copy()
            strong['Shift'] = shift
            all_strong.append(strong)
        
        combined = pd.concat(all_strong)
        combined = combined.sort_values('Correlation', ascending=False)
        
        # Print top 20
        print("Top 20 Strongest Lead-Lag Relationships:")
        print("-" * 50)
        for i, row in enumerate(combined.head(20).itertuples(), 1):
            print(f"{i:2d}. {row.Leader:6} → {row.Follower:6} "
                  f"({row.Shift:2d}d): {row.Correlation:.3f}")
        
        # Save to file
        combined.to_csv('strongest_correlations.csv', index=False)
        print(f"\nTotal correlations >0.9: {len(combined)}")
        print("Saved: strongest_correlations.csv")
        
        return combined
    
    def single_threshold_view(self, threshold=0.92):
        """Single clean view at one threshold"""
        
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        fig.suptitle(f'Correlation Networks (β > {threshold})', fontsize=12, fontweight='bold')
        
        for idx, shift in enumerate(self.shifts):
            ax = axes[idx]
            G = self.build_graph(shift, threshold)
            
            ax.set_title(f'{shift}-Day Shift\n{G.number_of_nodes()} stocks, {G.number_of_edges()} connections', 
                        fontsize=10)
            
            if G.number_of_nodes() == 0:
                ax.text(0.5, 0.5, 'None', ha='center', va='center', fontsize=12)
                ax.axis('off')
                continue
            
            # Layout
            if G.number_of_nodes() <= 8:
                pos = nx.circular_layout(G)
            else:
                pos = nx.spring_layout(G, seed=42)
            
            # Draw
            nx.draw(G, pos, ax=ax,
                   node_size=300,
                   node_color='skyblue',
                   font_size=8,
                   font_weight='bold',
                   width=2,
                   edge_color='gray',
                   arrows=True,
                   arrowsize=10,
                   with_labels=True)
            
            ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(f'network_threshold_{threshold}.png', dpi=100)
        plt.show()
        print(f"Saved: network_threshold_{threshold}.png")

def main():
    print("SIMPLE HIGH-CORRELATION ANALYZER")
    print("=" * 40)
    
    viz = SimpleCorrelationVisualizer()
    
    # Show single clean view at 0.92 threshold
    print("\n1. Creating main visualization (β > 0.92)...")
    viz.single_threshold_view(threshold=0.92)
    
    # Show multiple thresholds
    print("\n2. Comparing different high thresholds...")
    viz.visualize_simple(thresholds=[0.90, 0.93, 0.95])
    
    # Find strongest pairs
    print("\n3. Finding strongest correlations...")
    strongest = viz.find_strongest_pairs(min_threshold=0.9)
    
    # Summary stats
    print("\n=== SUMMARY ===")
    for shift in viz.shifts:
        df = viz.data[shift]
        above_90 = len(df[df["Correlation"] >= 0.9])
        above_95 = len(df[df["Correlation"] >= 0.95])
        print(f"{shift:2d}-day: {above_90} correlations >0.90, {above_95} >0.95")
    
    print("\n✓ Complete!")

if __name__ == "__main__":
    main()