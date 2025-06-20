from portfolio_backtester import PortfolioBacktester
import numpy as np
# Define the tau range
tau_values = np.round(np.linspace(0.005, 0.100, 20), 4)

# Subset of stocks for analysis
subset_stocks = [
    "RELIANCE.NS", "HDFCBANK.NS", "TCS.NS", "INFY.NS", "ICICIBANK.NS",
    "BHARTIARTL.NS", "ASIANPAINT.NS", "TITAN.NS", "LT.NS", "BAJFINANCE.NS"
]

# Initialize a results dictionary
sensitivity_results = []

# Loop through tau values
for tau in tau_values:
    print(f"\n🔁 Running backtest for τ = {tau}")
    backtester = PortfolioBacktester(stock_list=subset_stocks)

    results = backtester.run_comprehensive_backtest(
        sequence_length=30,
        epochs=25,
        batch_size=32,
        prediction_horizon=5,
        tau=tau,
        output_dir=f"results/tau_{tau}",
        use_frozen_data=False  # You can use True if data is already cached
    )

    # Extract and store key metrics
    perf = results['type_2']['portfolio_performance']
    sensitivity_results.append({
        "tau": tau,
        "annualized_return": perf['annualized_return'],
        "sharpe_ratio": perf['sharpe_ratio'],
        "volatility": perf['volatility'],
        "max_drawdown": perf['max_drawdown']
    })

# Convert to DataFrame for export or plotting
import pandas as pd
df_sensitivity = pd.DataFrame(sensitivity_results)
df_sensitivity.to_csv("sensitivity_tau_results.csv", index=False)
print("✅ Saved sensitivity analysis results to sensitivity_tau_results.csv")
