import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import os

def create_directed_graph(df, shift_days, threshold=0.95):
    """
    Create a directed graph from correlation data for a specific shift period.
    Edge direction: Ticker1 -> Ticker2
    """
    # Filter for specific shift and above threshold
    filtered_df = df[(df['ShiftDays'] == shift_days) & (df['Correlation'] >= threshold)]
    
    # Create directed graph
    G = nx.DiGraph()
    
    # Add edges with correlation as weight
    for _, row in filtered_df.iterrows():
        G.add_edge(row['Ticker1'], row['Ticker2'], weight=row['Correlation'])
    
    return G, filtered_df

def plot_directed_graph(G, shift_days, threshold=0.95, output_dir='correlation_graphs'):
    """
    Plot directed graph with color-coded edges based on correlation strength.
    """
    if len(G.edges()) == 0:
        print(f"No edges above threshold {threshold} for {shift_days}-day shift")
        return
    
    # Set up the plot with larger figure for less clutter
    fig_size = min(16, 8 + len(G.nodes()) * 0.3)
    plt.figure(figsize=(fig_size, fig_size))
    
    # Use different layouts based on graph size for better visualization
    if len(G.nodes()) < 10:
        pos = nx.spring_layout(G, k=3, iterations=50)
    elif len(G.nodes()) < 20:
        pos = nx.kamada_kawai_layout(G)
    else:
        # For larger graphs, use circular layout to reduce clutter
        pos = nx.circular_layout(G)
    
    # Extract edge weights for coloring
    edges = G.edges()
    weights = [G[u][v]['weight'] for u, v in edges]
    
    # Color map: higher correlation = darker red
    cmap = cm.Reds
    vmin, vmax = threshold, 1.0
    
    # Draw the graph
    nx.draw_networkx_nodes(G, pos, node_color='lightblue', 
                          node_size=700, alpha=0.9)
    
    # Normalize weights for color mapping
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    edge_colors = [cmap(norm(weight)) for weight in weights]
    
    # Draw edges with colors based on correlation
    nx.draw_networkx_edges(G, pos, edge_color=edge_colors,
                          width=2, alpha=0.7, arrows=True,
                          arrowsize=20,
                          connectionstyle='arc3,rad=0.1')
    
    # Draw labels with better positioning
    nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold')
    
    # Add colorbar
    sm = cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=plt.gca(), fraction=0.046, pad=0.04)
    cbar.set_label('Correlation', rotation=270, labelpad=15)
    
    plt.title(f'Directed Correlation Graph - {shift_days}-Day Shift\n'
              f'(Threshold: {threshold}, Edges: {len(G.edges())})',
              fontsize=14, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    
    # Save the figure to the output directory
    filename = os.path.join(output_dir, f'correlation_graph_{shift_days}d_threshold_{int(threshold*100)}.png')
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"  Saved graph to {filename}")
    plt.show()

def main():
    # Load the correlation data
    df = pd.read_csv('all_shifted_correlations.csv')
    
    # Create output directory for graphs
    output_dir = 'correlation_graphs'
    os.makedirs(output_dir, exist_ok=True)
    print(f"Created/using output directory: {output_dir}/")
    
    # Threshold for correlation
    threshold = 0.88
    
    # Create graphs for each shift period
    shift_days_list = [0, 10, 20, 30]
    
    print(f"\nCreating directed graphs with correlation threshold: {threshold}\n")
    
    for shift_days in shift_days_list:
        print(f"\nProcessing {shift_days}-day shift...")
        
        # Create directed graph
        G, filtered_df = create_directed_graph(df, shift_days, threshold)
        
        # Print statistics
        print(f"  Nodes: {len(G.nodes())}")
        print(f"  Edges: {len(G.edges())}")
        
        if len(G.edges()) > 0:
            correlations = [G[u][v]['weight'] for u, v in G.edges()]
            print(f"  Correlation range: [{min(correlations):.3f}, {max(correlations):.3f}]")
            
            # Find nodes with most outgoing edges
            out_degrees = dict(G.out_degree())
            if out_degrees:
                top_node = max(out_degrees, key=out_degrees.get)
                print(f"  Most connections (outgoing): {top_node} ({out_degrees[top_node]} edges)")
        
        # Plot the graph
        plot_directed_graph(G, shift_days, threshold, output_dir)
    
    print(f"\n✅ All graphs saved to {output_dir}/ folder")

if __name__ == "__main__":
    main()