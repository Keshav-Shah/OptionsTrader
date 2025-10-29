# beta_graph_poc.py
import yfinance as yf
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

# ---------------------- USER PARAMETERS ----------------------
tickers = ["AAPL","MSFT","AMZN","NVDA","TSLA","JPM","XOM","PFE","WMT","META"]
start = "2022-01-01"
end = "2025-01-01"
window = 60               # rolling window (days) used to compute betas
beta_threshold = 0.4      # prune edges with |beta| <= threshold
top_k = 3                 # how many "predicted up" stocks to hold each day
transaction_cost_per_trade = 0.0005  # cost fraction per stock traded (round-trip)
# -------------------------------------------------------------

# 1) Download prices and compute daily returns
prices = yf.download(tickers, start=start, end=end)['Adj Close']
prices = prices.dropna(how='all')  # drop empty rows if any
returns = prices.pct_change().dropna()  # r_i(t) = (P_t - P_{t-1}) / P_{t-1}

# 2) Pre-compute rolling stds (for beta formula)
rolling_std = returns.rolling(window).std()

# 3) Compute rolling betas for all ordered pairs using correlation approximation
#    beta_ij(t) ≈ corr(r_i(t), r_j(t-1)) * (σ_i(t) / σ_j(t))
# We'll produce a dict of DataFrames: betas[i][j] is a Series indexed by date.
betas = {i: {} for i in tickers}
lagged_returns = returns.shift(1)  # r_j(t-1) used as regressor

for i in tickers:
    for j in tickers:
        if i == j:
            continue
        # rolling correlation between r_i(t) and r_j(t-1)
        corr_ij = returns[i].rolling(window).corr(lagged_returns[j])
        # compute beta = corr * (sigma_i / sigma_j)
        beta_series = corr_ij * (rolling_std[i] / rolling_std[j])
        betas[i][j] = beta_series

# 4) Backtest loop: for each day t (starting at `window`), build pruned directed graph and predict r_i(t+1)
dates = returns.index
start_idx = window            # first index where rolling betas are valid
strategy_returns = []         # daily strategy returns
daily_hit_rates = []          # fraction of picks that were correct
cumulative_positions = None   # previous day's holdings for turnover calc
daily_pnl_series = pd.Series(index=dates[start_idx+1:])  # store strategy returns by date

prev_selection = set()  # for turnover calc

for t in range(start_idx, len(dates)-1):
    date_t = dates[t]
    date_next = dates[t+1]
    # 4.1) Build predictions for each stock:
    preds = {}
    for i in tickers:
        s = 0.0
        for j in tickers:
            if i == j:
                continue
            b = betas[i][j].iloc[t]
            # prune by threshold
            if pd.isna(b) or abs(b) <= beta_threshold:
                continue
            # use last-observed return of j (r_j(t)) as input
            r_j_t = returns[j].iloc[t]
            s += b * r_j_t
        preds[i] = s

    # 4.2) choose top_k predicted-up stocks
    ranked = sorted(preds.items(), key=lambda x: x[1], reverse=True)
    selection = [r[0] for r in ranked[:top_k]]

    # 4.3) compute realized next-day returns of selected stocks
    realized = returns.loc[date_next, selection]

    # 4.4) compute strategy return (equal-weighted on selected stocks)
    if len(selection) == 0:
        strat_ret = 0.0
        hit_rate = 0.0
    else:
        strat_ret = realized.mean()

        # 4.5) simple transaction cost model:
        # count trades as number of stocks that changed from previous selection
        trades = len(set(selection).symmetric_difference(prev_selection))
        cost = trades * transaction_cost_per_trade
        strat_ret = strat_ret - cost

        # hit rate = fraction of selections that had positive realized return
        hit_rate = (realized > 0).sum() / len(selection)

    # record
    daily_pnl_series.loc[date_next] = strat_ret
    strategy_returns.append(strat_ret)
    daily_hit_rates.append(hit_rate)

    # update prev selection
    prev_selection = set(selection)

# 5) Performance metrics & plots
strategy_returns_arr = np.array(strategy_returns)
mean_ret = np.nanmean(strategy_returns_arr)
std_ret = np.nanstd(strategy_returns_arr)
annualized_sharpe = (mean_ret / std_ret) * np.sqrt(252) if std_ret > 0 else np.nan

print("POC results:")
print(f"Days simulated: {len(strategy_returns_arr)}")
print(f"Average daily return (strategy): {mean_ret:.6f}")
print(f"Std dev daily return: {std_ret:.6f}")
print(f"Annualized Sharpe (approx): {annualized_sharpe:.2f}")
print(f"Average hit rate (per day): {np.nanmean(daily_hit_rates):.3f}")

# cumulative returns plot
daily_pnl_series = daily_pnl_series.dropna()
cum = (1 + daily_pnl_series).cumprod()
plt.figure(figsize=(10,5))
plt.plot(cum.index, cum.values)
plt.title("Cumulative Return of Beta-Graph POC (equal-weighted top_k picks)")
plt.xlabel("Date")
plt.ylabel("Cumulative return (1 = start)")
plt.grid(True)
plt.show()

# Optional: visualize the last pruned directed graph
last_t = len(dates) - 2  # second-last valid day
G = nx.DiGraph()
for i in tickers:
    for j in tickers:
        if i == j: continue
        b = betas[i][j].iloc[last_t]
        if pd.isna(b): continue
        if abs(b) > beta_threshold:
            # j -> i (j influences i)
            G.add_edge(j, i, weight=b)

plt.figure(figsize=(8,6))
pos = nx.spring_layout(G, seed=42)
nx.draw(G, pos, with_labels=True, node_size=900)
edge_labels = {(u,v): f"{d['weight']:.2f}" for u,v,d in G.edges(data=True)}
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
plt.title("Pruned Beta Graph (last snapshot)")
plt.show()
