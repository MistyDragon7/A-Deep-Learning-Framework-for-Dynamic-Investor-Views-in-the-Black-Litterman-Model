import itertools
import pandas as pd
from portfolio_backtester import PortfolioBacktester
import os
import seaborn as sns
import matplotlib.pyplot as plt

# Nifty 50 stock list (same as in main.py)
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

# Define parameter grid
risk_aversion_list = [1.0, 2.0, 3.0, 5.0, 10.0]
tau_list = [0.001, 0.01, 0.025, 0.05, 0.1]
grid = list(itertools.product(risk_aversion_list, tau_list))

# Store sensitivity results
results = []

for risk_aversion, tau in grid:
    print(f"\n🔍 Testing risk_aversion={risk_aversion}, tau={tau}")
    backtester = PortfolioBacktester(stock_list=nifty_50_stocks)

    output_dir = f"results/sensitivity_r{risk_aversion}_t{tau}"
    os.makedirs(output_dir, exist_ok=True)

    try:
        res = backtester.run_comprehensive_backtest(
            sequence_length=30,
            epochs=5,
            batch_size=32,
            prediction_horizon=5,
            risk_aversion=risk_aversion,
            tau=tau,
            use_frozen_data=True,
            frozen_data_path="data/frozen_data.pkl",
            output_dir=output_dir,
            save_plot_path=os.path.join(output_dir, "performance_comparison.png")
        )

        if res.get("type_2") and res["type_2"].get("portfolio_performance"):
            perf = res["type_2"]["portfolio_performance"]
            results.append({
                "risk_aversion": risk_aversion,
                "tau": tau,
                "annualized_return": perf["annualized_return"],
                "sharpe_ratio": perf["sharpe_ratio"],
                "volatility": perf["volatility"],
                "max_drawdown": perf["max_drawdown"]
            })

    except Exception as e:
        print(f"❌ Failed for risk_aversion={risk_aversion}, tau={tau}: {e}")

# Save results to CSV
df = pd.DataFrame(results)
df.to_csv("sensitivity_results.csv", index=False)
print("\n✅ Saved sensitivity analysis results to 'sensitivity_results.csv'")

# Plot heatmaps
if not df.empty:
    pivot_sharpe = df.pivot("risk_aversion", "tau", "sharpe_ratio")
    pivot_return = df.pivot("risk_aversion", "tau", "annualized_return")

    plt.figure(figsize=(10, 6))
    sns.heatmap(pivot_sharpe, annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Sharpe Ratio Sensitivity")
    plt.savefig("sharpe_ratio_heatmap.png", dpi=300, bbox_inches='tight')
    plt.show()

    plt.figure(figsize=(10, 6))
    sns.heatmap(pivot_return, annot=True, fmt=".2%", cmap="viridis")
    plt.title("Annualized Return Sensitivity")
    plt.savefig("annualized_return_heatmap.png", dpi=300, bbox_inches='tight')
    plt.show()
