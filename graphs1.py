import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

class CorrelationGraph:
    def __init__(self, corr_path, beta_threshold=0.9):
        """
        Initialize the graph from a correlation DataFrame file.
        corr_path: path to CSV or PKL file with columns [Ticker1, Ticker2, Correlation].
        beta_threshold: correlation threshold above which to connect nodes.
        """
        if corr_path.endswith(".csv"):
            self.df = pd.read_csv(corr_path)
        elif corr_path.endswith(".pkl"):
            self.df = pd.read_pickle(corr_path)
        else:
            raise ValueError("File must be .csv or .pkl")

        self.beta_threshold = beta_threshold
        self.graph = nx.Graph()

    def build_graph(self):
        """
        Build a NetworkX graph with edges between stocks whose correlation > beta_threshold.
        """
        filtered = self.df[self.df["Correlation"] > self.beta_threshold]
        for _, row in filtered.iterrows():
            t1, t2, corr = row["Ticker1"], row["Ticker2"], row["Correlation"]
            self.graph.add_edge(t1, t2, weight=corr)
        return self.graph

    def show_graph(self, max_nodes=100):
        """
        Display the correlation graph (limited to max_nodes for clarity).
        """
        if self.graph.number_of_nodes() == 0:
            print("⚠️ Graph is empty. Call build_graph() first.")
            return

        G = self.graph
        if G.number_of_nodes() > max_nodes:
            print(f"Graph has {G.number_of_nodes()} nodes. Showing only first {max_nodes}.")
            G = G.subgraph(list(G.nodes())[:max_nodes])

        pos = nx.spring_layout(G, k=0.4)
        plt.figure(figsize=(12, 8))
        nx.draw_networkx_nodes(G, pos, node_size=300, node_color='skyblue', alpha=0.8)
        nx.draw_networkx_edges(G, pos, width=1.5, alpha=0.6)
        nx.draw_networkx_labels(G, pos, font_size=9)
        plt.title(f"Stock Correlation Graph (β > {self.beta_threshold})")
        plt.axis("off")
        plt.tight_layout()
        plt.show()

    def get_connected_components(self):
        """
        Return list of connected components (each a set of tickers).
        """
        if self.graph.number_of_nodes() == 0:
            print("⚠️ Graph is empty. Call build_graph() first.")
            return []
        components = list(nx.connected_components(self.graph))
        components_sorted = sorted(components, key=len, reverse=True)
        return components_sorted


if __name__ == "__main__":
    beta_threshold = 0.95   # correlation cutoff
    corr_path = "shifted_correlations_3d.csv"

    cg = CorrelationGraph(corr_path, beta_threshold)
    G = cg.build_graph()

    print(f"✅ Graph built with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")

    components = cg.get_connected_components()
    print(f"🔗 Found {len(components)} connected components")
    print("Largest component:", list(components[0])[:10], "...")

    cg.show_graph(max_nodes=50)
