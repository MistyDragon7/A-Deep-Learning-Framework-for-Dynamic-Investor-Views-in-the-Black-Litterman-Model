import os
import random
import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from portfolio_backtester import PortfolioBacktester
import shutil

SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
tf.keras.utils.set_random_seed(SEED)
tf.config.experimental.enable_op_determinism()
nifty_50_stocks = [
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

backtester = PortfolioBacktester(stock_list=nifty_50_stocks)

results = backtester.run_comprehensive_backtest(
    sequence_length=30,
    epochs=25,
    batch_size=32,
    prediction_horizon=5,
    tau=0.025,
    output_dir="results",
    use_frozen_data=True,
    frozen_data_path="data/frozen_data.pkl"
)

backtester.plot_weight_variance_over_time(
    backtest_type='type_2',
    top_n_assets=10,
    save_path="results/weight_variance_plot.png"
)

summary = []
res = results.get("type_2", {})
p = res.get('portfolio_performance')
n = res.get('nifty_performance')

if p and n:
    summary.append({
        "Backtest": "Out-of-Sample (Bi-weekly)",
        "Portfolio Return": p['annualized_return'],
        "Portfolio Sharpe": p['sharpe_ratio'],
        "Portfolio Volatility": p['volatility'],
        "Nifty Return": n['annualized_return'],
        "Nifty Sharpe": n['sharpe_ratio'],
        "Nifty Volatility": n['volatility'],
        "Excess Return": p['annualized_return'] - n['annualized_return']
    })

summary_df = pd.DataFrame(summary)
summary_df.to_csv("backtest_summary.csv", index=False)
print("✅ Saved summary metrics to backtest_summary.csv")

# ✅ 7. Zip all output
output_dir = "results"
zip_path = f"{output_dir}.zip"

if os.path.exists(zip_path):
    os.remove(zip_path)
shutil.make_archive(output_dir, 'zip', output_dir)
print(f"✅ Zipped all output to: {zip_path}")