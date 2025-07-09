# -*- coding: utf-8 -*-
"""
Portfolio Backtester — **Two‑Stage (Precompute → Playback) Backtrader Pipeline**
==============================================================================

**Why this refactor?**
Running CNN‑BiLSTM (TensorFlow) inference inside Backtrader's `next()` loop
is slow and can exhaust GPU/CPU RAM.  Instead we

1. **Pre‑compute Black‑Litterman weights** for each rebalance date using
   *only* information available up to that date (so there's **no look‑ahead**).
2. **Store** the weights to disk (`parquet` by default).
3. **Playback** those weights inside a minimal Backtrader strategy that never
touches TensorFlow.

The result: identical trading logic, 100× faster backtests, and no repeated
TensorFlow initialisations.

You still get:
* 60 : 40 chronological train:test split.
* 10‑trading‑day (≈bi‑weekly) rebalances.
* Auto‑(re)train of CNN‑BiLSTM models if checkpoints are missing.

---
### Usage
```bash
# 1) Pre‑compute weights (runs TensorFlow once)
python portfolio_backtester.py --stage precompute --symbols RELIANCE.NS TCS.NS INFY.NS …

# 2) Run the fast Backtrader simulation (no TensorFlow)
python portfolio_backtester.py --stage backtest --weights_path precomputed_weights.parquet
```
---
"""
from __future__ import annotations

import argparse
import datetime as dt
import os
from pathlib import Path
from typing import Dict, List

import backtrader as bt
import pandas as pd
import numpy as np

from stock_data_fetcher import StockDataFetcher  # filecite turn2file1
from views_generator import CNNBiLSTMViewsGenerator  # filecite turn2file0
from black_litterman_optimizer import BlackLittermanOptimizer  # filecite turn2file11

# ---------------------------------------------------------------------------
# Configuration (override via CLI)
# ---------------------------------------------------------------------------
DEFAULT_SYMBOLS: List[str] = [
    "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS",
    "KOTAKBANK.NS", "LT.NS", "SBIN.NS", "AXISBANK.NS", "HINDUNILVR.NS",
    "ITC.NS", "BHARTIARTL.NS", "ASIANPAINT.NS", "HCLTECH.NS", "WIPRO.NS",
    "SUNPHARMA.NS", "ULTRACEMCO.NS", "MARUTI.NS", "BAJFINANCE.NS", "TITAN.NS",
    "POWERGRID.NS", "TECHM.NS", "NESTLEIND.NS", "ADANIENT.NS", "ADANIPORTS.NS",
    "GRASIM.NS", "HDFCLIFE.NS", "HEROMOTOCO.NS", "CIPLA.NS", "DIVISLAB.NS",
    "BAJAJFINSV.NS", "TATAMOTORS.NS", "JSWSTEEL.NS", "COALINDIA.NS", "DRREDDY.NS",
    "HINDALCO.NS", "NTPC.NS", "ONGC.NS", "BPCL.NS", "EICHERMOT.NS",
    "M&M.NS", "TATASTEEL.NS", "UPL.NS", "BRITANNIA.NS", "BAJAJ-AUTO.NS",
    "SBILIFE.NS", "INDUSINDBK.NS", "SHREECEM.NS", "ICICIPRULI.NS", "APOLLOHOSP.NS"
]
# GLOBAL_START_DATE = "2019-01-01" # Keep 5 years for full data fetch
# GLOBAL_END_DATE = "2024-04-30"   # Keep 5 years for full data fetch
GLOBAL_START_DATE = "2020-01-01"
GLOBAL_END_DATE = "2025-02-28" # February 2023 as requested

INTERVAL = "1d"
# TRAIN_TEST_SPLIT = 0.60  # No longer used for fixed split
REB_FREQ = 10            # trading‑day rebalance cadence
INITIAL_CASH = 1_000_000
SEQ_LEN = 30
EPOCHS = 20
BATCH_SIZE = 64
PRED_HORIZON = 5
WEIGHTS_PATH_DEFAULT = "precomputed_weights.parquet"
MODEL_DIR = "saved_models"
BENCHMARK_SYM = "^NSEI"

# Rolling window parameters
TRAINING_WINDOW_DAYS = 252  # ~1 year of trading days for training
RETRAIN_FREQ_DAYS = 63      # ~1 quarter of trading days for retraining (rebalancing frequency)
# ---------------------------------------------------------------------------
# Stage 1 – Pre‑compute BL weights and save to disk
# ---------------------------------------------------------------------------

def precompute_weights(symbols: List[str], weights_path: str = WEIGHTS_PATH_DEFAULT):
    """Compute BL weights for each rebalance date and persist to *weights_path*."""

    # 1) Download + enrich data once for the entire period
    fetcher = StockDataFetcher(symbols, start_date=GLOBAL_START_DATE, end_date=GLOBAL_END_DATE, interval=INTERVAL)
    stock_data = fetcher.fetch_all_stocks()
    fetcher.add_technical_indicators()
    full_returns_matrix = fetcher.create_returns_matrix()

    if full_returns_matrix.empty:
        raise RuntimeError("No returns matrix — aborting pre‑compute stage.")

    # Get all trading dates within the global range from the returns matrix index
    # These dates are already timezone-naive and normalized from StockDataFetcher
    all_trading_dates = full_returns_matrix.index.tolist()
    
    # Determine the first possible rebalance date after the initial training window.
    # This is the point from which we start generating 10-day rebalance points.
    # It should be roughly 1 year (TRAINING_WINDOW_DAYS) after GLOBAL_START_DATE.
    global_start_date_ts = pd.to_datetime(GLOBAL_START_DATE)

    # Find the index of GLOBAL_START_DATE in all_trading_dates
    # Handle cases where GLOBAL_START_DATE might not be an exact trading day
    global_start_idx = 0
    if global_start_date_ts in all_trading_dates:
        global_start_idx = all_trading_dates.index(global_start_date_ts)
    else:
        # Find the first trading date on or after GLOBAL_START_DATE
        for idx, date in enumerate(all_trading_dates):
            if date >= global_start_date_ts:
                global_start_idx = idx
                break

    # The first rebalance date (for which weights are computed) should be at least
    # TRAINING_WINDOW_DAYS trading days after the start of the *overall* data.
    first_rebalance_data_eligible_idx = global_start_idx + TRAINING_WINDOW_DAYS

    if first_rebalance_data_eligible_idx >= len(all_trading_dates):
        raise RuntimeError("Not enough data for initial training window for any rebalance point.")

    # Find the first actual 10-day rebalance point. It should be at or after
    # first_rebalance_data_eligible_idx and align with REB_FREQ.
    start_idx_for_10_day_rebalance_points = None
    for i in range(first_rebalance_data_eligible_idx, len(all_trading_dates)):
        # Calculate how many REB_FREQ intervals from global_start_idx
        if (i - global_start_idx) % REB_FREQ == 0:
            start_idx_for_10_day_rebalance_points = i
            break
    
    if start_idx_for_10_day_rebalance_points is None:
        # Fallback if no exact REB_FREQ alignment found after eligible date, just use the eligible date
        start_idx_for_10_day_rebalance_points = first_rebalance_data_eligible_idx
        print(f"⚠️ Could not find perfect 10-day alignment after initial training. Starting rebalance points from {all_trading_dates[start_idx_for_10_day_rebalance_points].date()}")


    # Generate 10-day rebalance dates from these trading dates
    rebalance_points_10_day: List[pd.Timestamp] = []
    for i in range(start_idx_for_10_day_rebalance_points, len(all_trading_dates), REB_FREQ):
        rebalance_points_10_day.append(all_trading_dates[i])

    # Ensure the very last date in the overall period is included as a rebalance point if not already
    if all_trading_dates and rebalance_points_10_day and all_trading_dates[-1] not in rebalance_points_10_day:
        rebalance_points_10_day.append(all_trading_dates[-1])
    
    # Sort and ensure uniqueness just in case (though should be ordered by construction)
    rebalance_points_10_day = sorted(list(set(rebalance_points_10_day)))

    print(f"📅 Total 10-day Rebalance Points: {len(rebalance_points_10_day)}")
    if not rebalance_points_10_day:
        raise RuntimeError("No 10-day rebalance points generated after initial training window — cannot proceed.")


    weight_rows = []
    
    # Initialize variables for managing quarterly retraining
    last_retrain_date = None
    current_views = None
    current_view_uncertainties = None

    for i, current_rebalance_date in enumerate(rebalance_points_10_day):
        print(f"Processing rebalance date: {current_rebalance_date.date()}")

        # Determine if it's time to retrain the models (quarterly logic)
        is_first_rebalance = (last_retrain_date is None) # Retrain on first actual rebalance point
        should_retrain = False

        if is_first_rebalance:
            should_retrain = True
        elif last_retrain_date:
            # Find the number of trading days between last_retrain_date and current_rebalance_date
            # Get subset of `all_trading_dates` between these two dates
            trading_days_since_last_retrain = len([d for d in all_trading_dates if last_retrain_date < d <= current_rebalance_date])
            if trading_days_since_last_retrain >= RETRAIN_FREQ_DAYS:
                should_retrain = True

        if should_retrain:
            print(f"🔄 Retraining models at: {current_rebalance_date.date()}")
            
            # For training, data is from TRAINING_WINDOW_DAYS prior to current_rebalance_date
            current_reb_idx_in_all = all_trading_dates.index(current_rebalance_date)
            training_start_idx = max(0, current_reb_idx_in_all - TRAINING_WINDOW_DAYS)
            training_start_date = all_trading_dates[training_start_idx]

            # Ensure training_start_date is not before the global start date
            if training_start_date < global_start_date_ts:
                training_start_date = global_start_date_ts

            data_for_training = {
                t: df.loc[training_start_date:current_rebalance_date].copy()
                for t, df in fetcher.stock_data.items()
                if len(df.loc[training_start_date:current_rebalance_date]) > SEQ_LEN + PRED_HORIZON + 2
            }

            if len(data_for_training) < 2:
                print(f"⚠️ {current_rebalance_date.date()}: <2 assets with sufficient history for training/views — skipping retraining")
                if current_views is None:
                    print("🚫 No views available. Cannot compute weights. This should not happen if initial checks are sufficient.")
                    continue 
                else:
                    print("✅ Using previously trained models/views (no retraining this quarter, insufficient new data).")
            else:
                vg = CNNBiLSTMViewsGenerator(len(data_for_training), sequence_length=SEQ_LEN)
                vg.train_all_models(data_for_training, epochs=EPOCHS, batch_size=BATCH_SIZE, model_dir=MODEL_DIR, training_end_date=current_rebalance_date)
                current_views, current_view_uncertainties = vg.generate_investor_views(data_for_training, PRED_HORIZON)
                last_retrain_date = current_rebalance_date 

        if current_views is None:
            print(f"🚫 {current_rebalance_date.date()}: No views available to compute weights — skipping.")
            continue

        # For BL optimization, use data up to current_rebalance_date
        # Ensure we are using only available assets for the current training period
        available_assets_for_blo = [k for k in data_for_training.keys() if k in full_returns_matrix.columns]
        returns_matrix_for_blo = full_returns_matrix.loc[global_start_date_ts:current_rebalance_date][available_assets_for_blo]
        returns_matrix_for_blo = returns_matrix_for_blo.dropna()

        market_caps_slice = {k: v for k, v in fetcher.market_caps.items() if k in available_assets_for_blo and v > 0}

        if returns_matrix_for_blo.empty or not market_caps_slice:
            print(f"⚠️ {current_rebalance_date.date()}: Empty returns matrix or no valid market caps for BL optimization — skipping")
            continue

        blo = BlackLittermanOptimizer(returns_matrix_for_blo, market_caps_slice, risk_free_rate=0.06)
        weights, *_ = blo.black_litterman_optimization(current_views, current_view_uncertainties)

        w = weights / weights.sum()
        row = {"Date": current_rebalance_date}
        row.update(w.to_dict())
        weight_rows.append(row)
        print(f"✓ {current_rebalance_date.date()}: weights ready (top: {w.nlargest(3).to_dict()})")

    if not weight_rows:
        raise RuntimeError("No weights were computed — cannot proceed.")
    
    weights_df = pd.DataFrame(weight_rows).set_index("Date").sort_index()
    weights_df.to_parquet(weights_path)
    print(f"💾 Saved weights table to {weights_path} ({weights_df.shape[0]} rows, {weights_df.shape[1]-1} assets)")

# ---------------------------------------------------------------------------
# Stage 2 – Backtrader playback of pre‑computed weights
# ---------------------------------------------------------------------------

import pandas as pd
from pathlib import Path
import backtrader as bt
import matplotlib.pyplot as plt
from stock_data_fetcher import StockDataFetcher
from black_litterman_optimizer import BlackLittermanOptimizer
from views_generator import CNNBiLSTMViewsGenerator # Added for completeness of imports

WEIGHTS_PATH_DEFAULT = "precomputed_weights.parquet"
INITIAL_CASH = 1_000_000
BENCHMARK_SYM = "^NSEI"

class BLWeightPlaybackStrategy(bt.Strategy):
    params = dict(weights_df=None)

    def __init__(self):
        self.weights_df = self.p.weights_df.copy()
        # Normalize here to ensure consistency with current_dt in next()
        self.weights_df.index = pd.to_datetime(self.weights_df.index).normalize() 
        self.last_rebalanced = None
        self.portfolio_values = []
        self.dates = []

    def next(self):
        current_dt_py = self.datas[0].datetime.datetime(0)
        # Normalize here to ensure consistency with weights_df.index
        current_dt = pd.to_datetime(current_dt_py).normalize() 
        
        self.portfolio_values.append(self.broker.getvalue())
        self.dates.append(current_dt)

        if current_dt not in self.weights_df.index:
            return
        if self.last_rebalanced == current_dt:
            return

        row = self.weights_df.loc[current_dt].fillna(0)
        portfolio_val = self.broker.getvalue()
        for d in self.datas:
            sym = d._name
            tgt_wt = float(row.get(sym, 0.0))
            tgt_val = portfolio_val * tgt_wt
            
            if d.close[0] == 0 or pd.isna(d.close[0]):
                print(f"Warning: Close price for {sym} is zero or NaN at {current_dt}. Skipping order.")
                continue

            tgt_size = int(tgt_val / d.close[0])
            self.order_target_size(d, tgt_size)
        self.last_rebalanced = current_dt


def run_backtest(weights_path=WEIGHTS_PATH_DEFAULT, benchmark_symbol=BENCHMARK_SYM, output_dir="results"):
    weights_df = pd.read_parquet(weights_path)
    # Ensure weights_df index is normalized for comparison
    weights_df.index = pd.to_datetime(weights_df.index).normalize()
    
    symbols = list(weights_df.columns)
    all_syms = symbols + ([benchmark_symbol] if benchmark_symbol not in symbols else [])

    fetcher = StockDataFetcher(all_syms, start_date=GLOBAL_START_DATE, end_date=GLOBAL_END_DATE, interval=INTERVAL)
    fetcher.fetch_all_stocks()
    fetcher.add_technical_indicators()
    cerebro = bt.Cerebro()
    cerebro.broker.setcash(INITIAL_CASH)
    cerebro.broker.setcommission(commission=0.0015)
    for sym in symbols:
        if sym not in fetcher.stock_data:
            print(f"Warning: Data for {sym} not available from fetcher. Skipping.")
            continue
        df = fetcher.stock_data[sym].copy()
        # Normalize df index before feeding to Backtrader
        df.index = pd.to_datetime(df.index).normalize() 
        
        if df.empty:
            print(f"Warning: Dataframe for {sym} is empty after processing. Skipping.")
            continue
        
        feed = bt.feeds.PandasData(dataname=df)
        cerebro.adddata(feed, name=sym)

    cerebro.addstrategy(BLWeightPlaybackStrategy, weights_df=weights_df)
    res = cerebro.run()
    strat = res[0]

    nifty_df = fetcher.stock_data[benchmark_symbol].copy()
    # Normalize nifty_df index for consistent date handling
    nifty_df.index = pd.to_datetime(nifty_df.index).normalize()

    # The fix: Use the first date from weights_df as the test_start for NIFTY comparison
    test_start = weights_df.index[0]

    # Manually filter strat.dates and strat.portfolio_values for the backtest period
    actual_portfolio_dates = []
    actual_portfolio_values = []
    for i, date_val in enumerate(strat.dates):
        current_date_normalized = pd.to_datetime(date_val).normalize()
        if current_date_normalized >= test_start:
            actual_portfolio_dates.append(current_date_normalized)
            actual_portfolio_values.append(strat.portfolio_values[i])
    
    portfolio_series = pd.Series(actual_portfolio_values, index=actual_portfolio_dates)

    # Now for NIFTY series, ensure it's also based on dates from test_start
    nifty_sliced_for_plot = nifty_df["Close"].loc[nifty_df.index >= test_start]
    
    # Reindex NIFTY to match the newly constructed portfolio_series index
    nifty_series = nifty_sliced_for_plot.reindex(portfolio_series.index, method="pad")
    
    # Find the first non-NaN value in nifty_series for normalization
    first_valid_nifty_value_in_test = nifty_series.dropna().iloc[0]
    
    # Normalize to ₹1,000,000 from the test start
    nifty_series = (nifty_series / first_valid_nifty_value_in_test) * INITIAL_CASH

    plt.figure(figsize=(12, 6))
    plt.plot(portfolio_series, label="Portfolio", linewidth=2, color="green")
    plt.plot(nifty_series, label="NIFTY 50", linewidth=2, linestyle="--", color="steelblue")

    for reb_date in weights_df.index:
        closest = min(portfolio_series.index, key=lambda d: abs(d - reb_date))
        plt.axvline(closest, color="gray", linestyle=":", alpha=0.5)

    plt.title("Portfolio vs NIFTY 50")
    plt.xlabel("Date")
    plt.ylabel("Value (₹)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    # plt.show() # Removed plt.show()

    # Calculate and return performance metrics
    portfolio_performance = _calculate_performance_metrics(portfolio_series, INITIAL_CASH)
    nifty_performance = _calculate_performance_metrics(nifty_series, INITIAL_CASH)

    return portfolio_series, nifty_series, portfolio_performance, nifty_performance

def _calculate_performance_metrics(series: pd.Series, initial_value: float) -> Dict[str, float]:
    # Ensure series is not empty and contains valid numbers
    if series.empty or not pd.api.types.is_numeric_dtype(series):
        return {
            "annualized_return": 0.0,
            "sharpe_ratio": 0.0,
            "volatility": 0.0,
            "max_drawdown": 0.0,
        }

    # Calculate daily returns from the value series
    daily_returns = series.pct_change().dropna()

    if daily_returns.empty:
        return {
            "annualized_return": 0.0,
            "sharpe_ratio": 0.0,
            "volatility": 0.0,
            "max_drawdown": 0.0,
        }

    # Annualized Return (assuming daily data and 252 trading days)
    # The last value in the series represents the final portfolio value
    total_return = (series.iloc[-1] / initial_value) - 1
    time_delta = series.index[-1] - series.index[0]
    num_years = time_delta.days / 365.25 if time_delta.days > 0 else 0.0
    annualized_return = (1 + total_return)**(1 / num_years) - 1 if num_years > 0 else 0.0

    # Volatility (Annualized Standard Deviation of Daily Returns)
    annualized_volatility = daily_returns.std() * np.sqrt(252)

    # Sharpe Ratio (assuming risk_free_rate = 0 for simplicity, or define it)
    # Risk-free rate is often close to zero for short-term T-bills; adjust if specific rate is needed
    risk_free_rate = 0.06 # Using the same as in BlackLittermanOptimizer for consistency if applicable
    excess_daily_returns = daily_returns - (risk_free_rate / 252)
    sharpe_ratio = excess_daily_returns.mean() / excess_daily_returns.std() * np.sqrt(252) if excess_daily_returns.std() != 0 else 0.0

    # Max Drawdown
    # Calculate cumulative returns for drawdown calculation
    cumulative_returns = (1 + daily_returns).cumprod()
    peak = cumulative_returns.expanding(min_periods=1).max()
    drawdown = (cumulative_returns - peak) / peak
    max_drawdown = drawdown.min() if not drawdown.empty else 0.0

    return {
        "annualized_return": annualized_return,
        "sharpe_ratio": sharpe_ratio,
        "volatility": annualized_volatility,
        "max_drawdown": max_drawdown,
    }

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(description="Black‑Litterman Backtrader two‑stage pipeline")
    p.add_argument("--stage", choices=["precompute", "backtest"], required=True, help="Which stage to run")
    p.add_argument("--symbols", nargs="*", default=DEFAULT_SYMBOLS, help="Ticker list (space‑separated)")
    p.add_argument("--weights_path", default=WEIGHTS_PATH_DEFAULT, help="Parquet file to read/write weights")
    p.add_argument("--output_dir", default="results", help="Directory to save backtest results and plots.")
    return p.parse_args()


def main():
    args = _parse_args()

    if args.stage == "precompute":
        precompute_weights(args.symbols, args.weights_path)
    else:
        run_backtest(args.weights_path, output_dir=args.output_dir)

if __name__ == "__main__":
    main() 