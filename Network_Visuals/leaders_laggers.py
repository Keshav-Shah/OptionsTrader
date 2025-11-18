import pandas as pd
import networkx as nx
import plotly.graph_objects as go

def build_directed_graph(data_path, corr_threshold=0.8):
    df = pd.read_csv(data_path)
    G = nx.DiGraph()

    for _, row in df.iterrows():
        if abs(row['correlation']) >= corr_threshold:
            stock1, stock2 = row['stock1'], row['stock2']
            causal, acausal = row['causal_shift'], row['acausal_shift']

            if causal > acausal:
                G.add_edge(stock1, stock2, weight=causal)
            elif acausal > causal:
                G.add_edge(stock2, stock1, weight=acausal)

    return G, df

def compute_leader_scores(G):
    scores = {}
    for node in G.nodes:
        score = G.out_degree(node, weight='weight') - G.in_degree(node, weight='weight')
        scores[node] = score
    nx.set_node_attributes(G, scores, 'leader_score')
    return scores

def plot_leader_lagger_network(G, df):
    pos = nx.spring_layout(G, seed=42, k=2/len(G.nodes())**0.5)

    # Extract node attributes
    leader_scores = nx.get_node_attributes(G, 'leader_score')
    all_scores = list(leader_scores.values())
    max_abs = max(abs(min(all_scores)), abs(max(all_scores)))

    # Normalize colors between -1 and 1
    node_colors = [
        (score / max_abs if max_abs != 0 else 0) for score in leader_scores.values()
    ]

    # Build node trace
    node_x, node_y, node_text = [], [], []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(f"{node}<br>Leader score: {leader_scores[node]:.2f}")

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        text=[n for n in G.nodes()],
        hovertext=node_text,
        textposition='top center',
        marker=dict(
            size=[10 + 2*(G.degree(n)) for n in G.nodes()],
            color=node_colors,
            colorscale='RdBu_r',  # reversed: red = leader, blue = lagger
            cmin=-1, cmax=1,
            showscale=True,
            colorbar=dict(
                title=dict(text="Leader → Lagger", side="right"),
                tickvals=[-1, 0, 1],
                ticktext=["Leader", "Neutral", "Lagger"],
                tickmode='array'
            ),
            line=dict(width=0.5, color='white')
        ),
        hoverinfo='text'
    )


    # Build edge traces (faint lines)
    edge_x, edge_y = [], []
    for u, v, data in G.edges(data=True):
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        mode='lines',
        line=dict(color='rgba(255,255,255,0.1)', width=1),
        hoverinfo='none'
    )

    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(
        title="Stock Influence Network (Leader–Lagger Heatmap)",
        showlegend=False,
        hovermode='closest',
        paper_bgcolor='#0a0a0a',
        plot_bgcolor='#0a0a0a',
        font=dict(color='white'),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
    )

    fig.show()


if __name__ == "__main__":
    csv_path = "stock_correlations.csv"
    G, df = build_directed_graph(csv_path, corr_threshold=0.75)
    compute_leader_scores(G)
    plot_leader_lagger_network(G, df)
