import argparse
import datetime as dt
import os
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import tensorflow as tf

from backtrader_tester import precompute_weights, run_backtest, WEIGHTS_PATH_DEFAULT, GLOBAL_START_DATE, GLOBAL_END_DATE, BENCHMARK_SYM

# Seed for reproducibility
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
tf.random.set_seed(SEED)
tf.config.experimental.enable_op_determinism()

def main():
    parser = argparse.ArgumentParser(description="Run Black-Litterman Portfolio Optimization and Backtest.")
    parser.add_argument('--stage', type=str, choices=['precompute', 'backtest'], required=True,
                        help="Specify 'precompute' to calculate and save weights, or 'backtest' to run simulation with precomputed weights.")
    parser.add_argument('--weights_path', type=str, default=WEIGHTS_PATH_DEFAULT,
                        help=f"Path to precomputed weights file (default: {WEIGHTS_PATH_DEFAULT}).")
    parser.add_argument('--output_dir', type=str, default="results",
                        help="Directory to save backtest results and plots.")
    
    args = parser.parse_args()

    # Ensure output directory exists
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    if args.stage == 'precompute':
        print("Starting precomputation stage...")
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
        precompute_weights(nifty_50_stocks, args.weights_path)
        print(f"Precomputation complete. Weights saved to {args.weights_path}")

    elif args.stage == 'backtest':
        print("Starting backtest stage...")
        portfolio_returns, benchmark_returns, portfolio_performance, nifty_performance = run_backtest(
            weights_path=args.weights_path,
            benchmark_symbol=BENCHMARK_SYM,
            output_dir=args.output_dir
        )

        # Save performance comparison plot
        print("Saving performance comparison chart...")
        plt.figure(figsize=(12, 6))
        plt.plot(portfolio_returns.index, portfolio_returns.values, label='Portfolio Returns')
        plt.plot(benchmark_returns.index, benchmark_returns.values, label='NIFTY 50 Returns')
        plt.title('Cumulative Returns: Portfolio vs. NIFTY 50')
        plt.xlabel('Date')
        plt.ylabel('Cumulative Returns')
        plt.legend()
        plt.grid(True)
        plot_path = Path(args.output_dir) / "performance_comparison.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved performance comparison chart to {plot_path}")

        # Save summary CSV
        print("Saving summary metrics...")
        summary = []
        summary.append({
            "Backtest": "Portfolio",
            "Annualized Return": portfolio_performance['annualized_return'],
            "Sharpe Ratio": portfolio_performance['sharpe_ratio'],
            "Volatility": portfolio_performance['volatility'],
            "Max Drawdown": portfolio_performance['max_drawdown'],
            "Excess Return": portfolio_performance['annualized_return'] - nifty_performance['annualized_return']
        })
        summary.append({
            "Backtest": "NIFTY 50",
            "Annualized Return": nifty_performance['annualized_return'],
            "Sharpe Ratio": nifty_performance['sharpe_ratio'],
            "Volatility": nifty_performance['volatility'],
            "Max Drawdown": nifty_performance['max_drawdown'],
            "Excess Return": 0 # Nifty doesn't have excess return over itself
        })
        summary_df = pd.DataFrame(summary)
        summary_path = Path(args.output_dir) / "backtest_summary.csv"
        summary_df.to_csv(summary_path, index=False)
        print(f"Saved summary metrics to {summary_path}")

        # Save cumulative returns CSVs
        portfolio_returns_path = Path(args.output_dir) / "cumulative_portfolio_returns.csv"
        benchmark_returns_path = Path(args.output_dir) / "cumulative_nifty_returns.csv"
        portfolio_returns.to_csv(portfolio_returns_path)
        benchmark_returns.to_csv(benchmark_returns_path)
        print(f"Saved cumulative portfolio returns to {portfolio_returns_path}")
        print(f"Saved cumulative NIFTY 50 returns to {benchmark_returns_path}")

        # Zip output
        zip_path = Path(args.output_dir).parent / f"{Path(args.output_dir).name}.zip"
        if zip_path.exists():
            os.remove(zip_path)
        shutil.make_archive(args.output_dir, 'zip', args.output_dir)
        print(f"✅ Zipped all output to: {zip_path}")
    else:
        print("Invalid stage specified. Use 'precompute' or 'backtest'.")

if __name__ == '__main__':
    main()
