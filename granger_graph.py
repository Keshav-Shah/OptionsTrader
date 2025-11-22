import pandas as pd
import networkx as nx
import plotly.graph_objects as go

def plot_granger_graph(csv_path="granger_results.csv", p_threshold=0.05):

    df = pd.read_csv(csv_path)

    # Build directed graph
    G = nx.DiGraph()

    for _, row in df.iterrows():
        A = row['stock1']
        B = row['stock2']

        # Only use significant causal directions
        if row['A_causes_B'] and row['p_AB'] < p_threshold:
            weight = 1 - row['p_AB']   # stronger = thicker
            G.add_edge(A, B, weight=weight)

        if row['B_causes_A'] and row['p_BA'] < p_threshold:
            weight = 1 - row['p_BA']
            G.add_edge(B, A, weight=weight)

    # Compute leader/lagger influence
    influence = {}
    for node in G.nodes():
        influence[node] = G.out_degree(node) - G.in_degree(node)

    nx.set_node_attributes(G, influence, 'influence')

    # Layout
    pos = nx.spring_layout(G, k=0.4, seed=42)

    # Build edges
    edge_x, edge_y = [], []
    for u, v in G.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        mode='lines',
        line=dict(width=1, color='rgba(255,0,0,0.4)'),
        hoverinfo='none'
    )

    # Build nodes
    node_x, node_y = [], []
    text, color, size = [], [], []

    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        infl = influence[node]

        text.append(f"{node}<br>Influence={infl}")
        color.append(infl)
        size.append(10 + abs(infl)*2)

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode='markers+text',
        text=list(G.nodes()),
        textposition='top center',
        hovertext=text,
        marker=dict(
            size=size,
            color=color,
            colorscale='RdBu',
            showscale=True,
            colorbar=dict(title="Influence")
        )
    )

    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(
        title="Granger Causal Network",
        showlegend=False,
        xaxis=dict(showgrid=False, zeroline=False),
        yaxis=dict(showgrid=False, zeroline=False),
        plot_bgcolor='black',
        paper_bgcolor='black',
        font=dict(color='white')
    )

    fig.show()

if __name__ == "__main__":
    plot_granger_graph()
