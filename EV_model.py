import pandas as pd
import numpy as np
import networkx as nx
import yfinance as yf

# ======================================================
# CONFIG
# ======================================================
CSV_PATH = "stock_correlations.csv"
CORR_THRESHOLD = 0.8
ALPHA = 0.2  # self vs network weighting
MAX_DAYS = 30  # how many past days of returns to average
COMBINE_WEIGHTS = (0.75, 0.25)

# ======================================================
# 1. BUILD GRAPH FROM CSV
# ======================================================
def build_graph(data_path, corr_threshold=0.8):
    df = pd.read_csv(data_path)
    G = nx.DiGraph()

    for _, row in df.iterrows():
        if abs(row["correlation"]) >= corr_threshold:
            stock1, stock2 = row["stock1"], row["stock2"]
            causal, acausal = row["causal_shift"], row["acausal_shift"]

            # direction and strength
            if causal > acausal:
                w = causal - acausal
                G.add_edge(stock1, stock2, weight=w)
            elif acausal > causal:
                w = acausal - causal
                G.add_edge(stock2, stock1, weight=w)

    # normalize outgoing edges so they sum to 1 per node
    for node in G.nodes():
        out_edges = list(G.out_edges(node, data=True))
        total = sum(abs(d["weight"]) for _, _, d in out_edges)
        if total > 0:
            for u, v, d in out_edges:
                d["weight"] = abs(d["weight"]) / total

    return G


# ======================================================
# 2. FETCH RECENT RETURNS
# ======================================================
def get_recent_returns(tickers, days=5):
    data = yf.download(tickers, period=f"{days+1}d")["Close"]
    returns = data.pct_change().iloc[1:]  # daily %
    mean_returns = returns.mean().to_dict()
    return mean_returns


# ======================================================
# 3. PROPAGATION LOGIC
# ======================================================
def propagate(G, returns, alpha=0.4):
    EV = {}
    for i in G.nodes():
        r_i = returns.get(i, 0)
        incoming = 0
        for j in G.predecessors(i):
            w = G[j][i]["weight"]
            r_j = returns.get(j, 0)
            incoming += w * r_j
        EV[i] = alpha * r_i + (1 - alpha) * incoming
    return EV


# ======================================================
# 4. TWO-PASS EXPECTED VALUE PROPAGATION
# ======================================================
def two_pass_expected_value(G, returns, alpha=0.4, combine_weights=(0.75, 0.25)):
    first_pass = propagate(G, returns, alpha)
    second_pass = propagate(G, first_pass, alpha)
    EV = {
        node: combine_weights[0] * first_pass[node] + combine_weights[1] * second_pass[node]
        for node in G.nodes()
    }
    return EV, first_pass, second_pass


# ======================================================
# 5. MAIN EXECUTION
# ======================================================
if __name__ == "__main__":
    print("Building graph...")
    G = build_graph(CSV_PATH, corr_threshold=CORR_THRESHOLD)

    tickers = list(G.nodes())
    print(f"Fetching returns for {len(tickers)} stocks...")
    returns = get_recent_returns(tickers, days=MAX_DAYS)

    EV, first, second = two_pass_expected_value(G, returns, alpha=ALPHA, combine_weights=COMBINE_WEIGHTS)

    # Rank winners and laggers
    winners = sorted(EV.items(), key=lambda x: x[1], reverse=True)[:10]
    laggers = sorted(EV.items(), key=lambda x: x[1])[:10]

    print("\n===== TOP 10 EXPECTED WINNERS =====")
    for s, v in winners:
        print(f"{s:8s}  EV={v:+.4f}")

    print("\n===== BOTTOM 10 EXPECTED LOSERS =====")
    for s, v in laggers:
        print(f"{s:8s}  EV={v:+.4f}")
