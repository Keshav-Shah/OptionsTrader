"""
granger_causality.py (FAST VERSION)

This version is optimized for speed:
 • uses lag = 1 (common in finance)
 • NO AIC lag selection (huge speedup)
 • only computes stationary returns
 • no nested loops of lag testing
"""

import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import grangercausalitytests
import warnings
warnings.filterwarnings("ignore")


# ----------------------------------------------
# Make series stationary (percentage returns)
# ----------------------------------------------
def make_stationary(series):
    return series.pct_change().dropna()


# ----------------------------------------------
# FAST Granger test: fixed lag = 1
# ----------------------------------------------
def granger_pair(series_A, series_B, lag=1):
    """
    Run real Granger causality tests with fixed lag = 1.

    Returns:
        {
            'lag': 1,
            'A_causes_B': bool,
            'B_causes_A': bool,
            'p_AB': p-value,
            'p_BA': p-value
        }
    """

    # Convert price → returns
    A = make_stationary(series_A)
    B = make_stationary(series_B)

    # Align series
    df = pd.concat([A, B], axis=1, join="inner")
    df.columns = ['A', 'B']
    df = df.dropna()

    if len(df) < 50:
        return None

    # A → B
    try:
        res_AB = grangercausalitytests(df[['B', 'A']], maxlag=lag, verbose=False)
        p_AB = res_AB[lag][0]['ssr_ftest'][1]
    except:
        p_AB = 1.0

    # B → A
    try:
        res_BA = grangercausalitytests(df[['A', 'B']], maxlag=lag, verbose=False)
        p_BA = res_BA[lag][0]['ssr_ftest'][1]
    except:
        p_BA = 1.0

    return {
        'lag': lag,
        'A_causes_B': p_AB < 0.05,
        'B_causes_A': p_BA < 0.05,
        'p_AB': p_AB,
        'p_BA': p_BA
    }


# ----------------------------------------------
# Run Granger causality for ALL stock pairs
# ----------------------------------------------
def run_granger_on_dict(data_dict, max_lag=1):
    """
    Run FAST Granger causality tests for every stock pair.
    Uses lag = 1 (ignores max_lag).
    """

    tickers = list(data_dict.keys())
    results = []

    for i in range(len(tickers)):
        for j in range(i+1, len(tickers)):
            t1, t2 = tickers[i], tickers[j]

            try:
                s1 = data_dict[t1]['Close'].dropna()
                s2 = data_dict[t2]['Close'].dropna()
            except:
                continue

            res = granger_pair(s1, s2, lag=1)
            if res is None:
                continue

            results.append({
                'stock1': t1,
                'stock2': t2,
                'lag': 1,
                'A_causes_B': res['A_causes_B'],
                'B_causes_A': res['B_causes_A'],
                'p_AB': res['p_AB'],
                'p_BA': res['p_BA']
            })

    return pd.DataFrame(results)
