import pickle
import pandas as pd
import numpy as np
import networkx as nx
from sklearn.cluster import SpectralClustering
import warnings
import argparse
import plotly.graph_objects as go
from rich.console import Console
from rich.table import Table

warnings.filterwarnings('ignore')
console = Console()

def load_correlations(pickle_file='stock_correlations.pkl'):
    return pd.read_pickle(pickle_file)

def build_correlation_network(corr_df, correlation_threshold=0.95):
    all_stocks = sorted(set(corr_df['stock1'].unique()) | set(corr_df['stock2'].unique()))
    G = nx.Graph()
    G.add_nodes_from(all_stocks)

    n_stocks = len(all_stocks)
    corr_matrix = np.zeros((n_stocks, n_stocks))
    stock_to_idx = {stock: i for i, stock in enumerate(all_stocks)}
    edge_weights = {}

    for _, row in corr_df.iterrows():
        if pd.notna(row['correlation']) and abs(row['correlation']) >= correlation_threshold:
            s1, s2 = row['stock1'], row['stock2']
            w = abs(row['correlation'])
            G.add_edge(s1, s2, weight=w)
            edge_weights[(s1, s2)] = w
            i, j = stock_to_idx[s1], stock_to_idx[s2]
            corr_matrix[i, j] = w
            corr_matrix[j, i] = w

    np.fill_diagonal(corr_matrix, 1.0)
    return G, corr_matrix, all_stocks, edge_weights

def cluster_stocks_func(corr_matrix, n_clusters=10):
    clustering = SpectralClustering(n_clusters=n_clusters, affinity='precomputed', random_state=42)
    corr_matrix_clean = np.nan_to_num(corr_matrix, nan=0.0)
    clusters = clustering.fit_predict(corr_matrix_clean)
    return clusters

def get_cluster_centers(G, stocks, clusters):
    centers = {}
    for c_id in np.unique(clusters):
        c_stocks = [s for s, c in zip(stocks, clusters) if c == c_id]
        if not c_stocks:
            continue
        center = max(c_stocks, key=lambda s: G.degree(s))
        centers[c_id] = center
    return centers

def interactive_network(G, pos, clusters, highlight_tickers=None):
    if highlight_tickers is None:
        highlight_tickers = []

    edge_x, edge_y = [], []
    for u, v in G.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]

    
    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        line=dict(width=0.3, color='rgba(120,120,120,0.5)'), 
        hoverinfo='none',
        mode='lines'
    )

    node_x, node_y, node_color, node_text = [], [], [], []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        c = clusters[node]
        node_color.append(c)
        node_text.append(f"{node} | C {c} | D {G.degree(node)}")

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode='markers+text',
        text=node_text,
        hoverinfo='text',
        marker=dict(
            showscale=True,
            colorscale='Rainbow',
            color=node_color,
            size=[15 if n in highlight_tickers else 8 for n in G.nodes()],
            colorbar=dict(title='Cluster')
        ),
        textposition="top center"
    )

    fig = go.Figure(data=[edge_trace, node_trace], layout=go.Layout(
        title="Interactive Stock Correlation Network",
        title_x=0.5,
        showlegend=False,
        hovermode='closest',
        margin=dict(b=20, l=5, r=5, t=40),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        paper_bgcolor='#0a0a0a',
        plot_bgcolor='#0a0a0a'))
    fig.show()

def print_summary(G, clusters):
    table = Table(title="Cluster Summary")
    table.add_column("Cluster", justify="right")
    table.add_column("Size", justify="right")
    table.add_column("Top Stock", justify="left")

    for c in sorted(set(clusters.values())):
        members = [s for s, k in clusters.items() if k == c]
        top = max(members, key=lambda s: G.degree(s))
        table.add_row(str(c), str(len(members)), top)
    console.print(table)

def create_network_visualization(corr_df, focal_tickers=None, highlight_tickers=None, 
                                n_clusters=12, correlation_threshold=0.8):
    if focal_tickers is None:
        focal_tickers = []
    elif isinstance(focal_tickers, str):
        focal_tickers = [focal_tickers]
    if highlight_tickers is None:
        highlight_tickers = focal_tickers
    elif isinstance(highlight_tickers, str):
        highlight_tickers = [highlight_tickers]

    G, corr_matrix, all_stocks, _ = build_correlation_network(corr_df, correlation_threshold)
    G.remove_nodes_from(list(nx.isolates(G)))
    remaining_stocks = list(G.nodes())
    stock_to_idx = {stock: i for i, stock in enumerate(all_stocks)}
    remaining_indices = [stock_to_idx[stock] for stock in remaining_stocks]
    corr_matrix_filtered = corr_matrix[np.ix_(remaining_indices, remaining_indices)]

    clusters_array = cluster_stocks_func(corr_matrix_filtered, n_clusters)
    stock_clusters = {stock: cluster for stock, cluster in zip(remaining_stocks, clusters_array)}
    cluster_centers = get_cluster_centers(G, remaining_stocks, clusters_array)
    pos = nx.spring_layout(G, k=2/np.sqrt(len(G.nodes())), iterations=50, seed=42)

    interactive_network(G, pos, stock_clusters, highlight_tickers)
    print_summary(G, stock_clusters)

    return {'graph': G, 'clusters': stock_clusters, 'positions': pos, 'centers': cluster_centers}

def main():
    parser = argparse.ArgumentParser(description='Create interactive network clustering visualization')
    parser.add_argument('--focal', '-f', type=str, nargs='+', default=['ITRI'])
    parser.add_argument('--highlight', '-hl', type=str, nargs='+', default=None)
    parser.add_argument('--clusters', '-k', type=int, default=12)
    parser.add_argument('--threshold', '-t', type=float, default=0.8)
    parser.add_argument('--file', type=str, default='stock_correlations.pkl')
    args = parser.parse_args()

    highlight_list = args.focal.copy() if args.focal else []
    if args.highlight:
        highlight_list.extend(args.highlight)
    highlight_list = list(set(highlight_list))

    print(f"\n🎯 Creating interactive visualization for: {', '.join(highlight_list)}")
    corr_df = load_correlations(args.file)
    create_network_visualization(corr_df, focal_tickers=args.focal, highlight_tickers=highlight_list, n_clusters=args.clusters, correlation_threshold=args.threshold)

if __name__ == '__main__':
    main()