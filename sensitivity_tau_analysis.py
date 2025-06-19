import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
from portfolio_backtester import PortfolioBacktester

# Define Nifty 50 subset (can expand)
nifty_50_stocks = [
    "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS"
]

# Only vary tau
tau_list = [0.005, 0.01, 0.025, 0.05, 0.1]

results = []

for tau in tau_list:
    print(f"\n🔍 Testing tau = {tau} (λ is dynamically estimated)")
    backtester = PortfolioBacktester(stock_list=nifty_50_stocks)

    output_dir = f"results/sensitivity_tau_{tau}"
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
        print(f"❌ Failed for tau = {tau}: {e}")

# Save results
df = pd.DataFrame(results)
df.to_csv("sensitivity_tau_results.csv", index=False)
print("\n✅ Saved tau sensitivity results to 'sensitivity_tau_results.csv'")

# Plot results
if not df.empty:
    plt.figure(figsize=(10, 5))
    sns.lineplot(x="tau", y="sharpe_ratio", data=df, marker="o")
    plt.title("Sensitivity of Sharpe Ratio to Tau (λ dynamically estimated)")
    plt.xlabel("Tau")
    plt.ylabel("Sharpe Ratio")
    plt.grid(True)
    plt.savefig("sharpe_ratio_tau_sensitivity.png", dpi=300, bbox_inches='tight')
    plt.show()

    plt.figure(figsize=(10, 5))
    sns.lineplot(x="tau", y="annualized_return", data=df, marker="o")
    plt.title("Sensitivity of Annualized Return to Tau (λ dynamically estimated)")
    plt.xlabel("Tau")
    plt.ylabel("Annualized Return")
    plt.grid(True)
    plt.savefig("annualized_return_tau_sensitivity.png", dpi=300, bbox_inches='tight')
    plt.show()
