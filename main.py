from portfolio_backtester import PortfolioBacktester
import matplotlib.pyplot as plt

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

backtester.run_comprehensive_backtest(
    sequence_length=30,
    epochs=25,
    batch_size=32,
    prediction_horizon=5,
    risk_aversion=3.0,
    tau=0.025,
    save_plot_path="results/performance_comparison.png",
    output_dir="results"
)

# Save performance plot
print("Saving performance comparison chart...")
plt.savefig("performance_comparison.png", dpi=300, bbox_inches='tight')
print("Saved as performance_comparison.png")

# Optional: Save summary metrics to CSV
import pandas as pd
summary = []
for backtest_type, res in results.items():
    label = "Full Training" if backtest_type == "type_1" else "Out-of-Sample"
    p = res['portfolio_performance']
    n = res['nifty_performance']
    if p and n:
        summary.append({
            "Backtest": label,
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
print("Saved summary metrics to backtest_summary.csv")
