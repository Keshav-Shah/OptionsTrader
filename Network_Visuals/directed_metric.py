import pandas as pd
import networkx as nx
import plotly.graph_objects as go

def build_directed_graph(data_path, corr_threshold=0.9):
    df = pd.read_csv(data_path)
    G = nx.DiGraph()

    # Add edges with direction from higher causal shift to lower
    for _, row in df.iterrows():
        if abs(row['correlation']) >= corr_threshold:
            s1, s2 = row['stock1'], row['stock2']
            causal, acausal = row['causal_shift'], row['acausal_shift']
            diff = causal - acausal
            if diff > 0:
                G.add_edge(s1, s2, weight=diff)
            elif diff < 0:
                G.add_edge(s2, s1, weight=-diff)
    return G


def plot_causal_network(G):
    # Layout
    pos = nx.spring_layout(G, seed=42, k=2 / (len(G.nodes()) ** 0.5))
    edges = list(G.edges(data=True))

    # Compute max weight for normalization
    weights = [d['weight'] for _, _, d in edges]
    max_w = max(weights) if weights else 1.0

    # Build edge traces (red opacity ~ Δshift strength)
    edge_x, edge_y, edge_colors = [], [], []
    for u, v, d in edges:
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]
        alpha = max(0.1, min(1.0, d['weight'] / max_w))
        edge_colors.append(alpha)

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        mode='lines',
        line=dict(width=2, color='rgba(255,0,0,0.3)'),  # base color red
        hoverinfo='none',
        opacity=0.9
    )

    # Build node info for hover
    node_x, node_y, hover_texts = [], [], []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)

        leaders = [(src, G[src][node]['weight']) for src in G.predecessors(node)]
        laggers = [(dst, G[node][dst]['weight']) for dst in G.successors(node)]

        leaders_text = "<br>".join([f"{src} → {node} ({w:.3f})" for src, w in sorted(leaders, key=lambda x: -x[1])]) or "None"
        laggers_text = "<br>".join([f"{node} → {dst} ({w:.3f})" for dst, w in sorted(laggers, key=lambda x: -x[1])]) or "None"

        hover_text = f"<b>{node}</b><br><br><b>Led by:</b><br>{leaders_text}<br><br><b>Leads:</b><br>{laggers_text}"
        hover_texts.append(hover_text)

    # Nodes all same size, same color, fully interactive
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        text=list(G.nodes()),
        textposition='top center',
        hovertext=hover_texts,
        marker=dict(
            size=14,
            color='white',
            line=dict(width=1, color='gray')
        ),
        hoverinfo='text'
    )

    # Combine
    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(
        title="Causal Stock Correlation Network (|corr| ≥ 0.9)",
        paper_bgcolor='#0a0a0a',
        plot_bgcolor='#0a0a0a',
        font=dict(color='white'),
        showlegend=False,
        hovermode='closest',
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
    )

    fig.show()


if __name__ == "__main__":
    csv_path = "stock_correlations.csv"
    G = build_directed_graph(csv_path)
    plot_causal_network(G)
