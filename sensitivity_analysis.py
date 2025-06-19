import itertools
import pandas as pd
from portfolio_backtester import PortfolioBacktester
import os
import seaborn as sns
import matplotlib.pyplot as plt

# Nifty 50 stock list (same as in main.py)
nifty_50_stocks = [
    "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS"
]

# Define parameter grid
# risk_aversion_list = [1.0, 2.0, 3.0, 5.0, 10.0]
tau_list = [0.01, 0.025, 0.05]

# Store sensitivity results
results = []

for  tau in tau_list:
    print(f"\n🔍 Testing tau={tau}")
    backtester = PortfolioBacktester(stock_list=nifty_50_stocks)

    output_dir = f"results/sensitivity_t{tau}"
    os.makedirs(output_dir, exist_ok=True)

    try:
        res = backtester.run_comprehensive_backtest(
            sequence_length=30,
            epochs=5,
            batch_size=32,
            prediction_horizon=5,
            tau=tau,
            use_frozen_data=True,
            frozen_data_path="data/frozen_data.pkl",
            output_dir=output_dir,
            save_plot_path=os.path.join(output_dir, "performance_comparison.png")
        )

        if res.get("type_2") and res["type_2"].get("portfolio_performance"):
            perf = res["type_2"]["portfolio_performance"]
            results.append({
                "tau": tau,
                "annualized_return": perf["annualized_return"],
                "sharpe_ratio": perf["sharpe_ratio"],
                "volatility": perf["volatility"],
                "max_drawdown": perf["max_drawdown"]
            })

    except Exception as e:
        print(f"❌ Failed for tau={tau}: {e}")

# Save results to CSV
df = pd.DataFrame(results)
df.to_csv("sensitivity_results.csv", index=False)
print("\n✅ Saved sensitivity analysis results to 'sensitivity_results.csv'")

# Plot heatmaps
if not df.empty:
    pivot_sharpe = df.pivot(columns="tau", values="sharpe_ratio", index=df.index)
    pivot_return = df.pivot(columns="tau", values="annualized_return", index=df.index)

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
