import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from views_generator import CNNBiLSTMViewsGenerator
from black_litterman_optimizer import BlackLittermanOptimizer
from stock_data_fetcher import StockDataFetcher

tickers = ["RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS"]
shock_multipliers = {
    "Baseline": 1.0,
    "+20% Shock": 1.2,
    "-20% Shock": 0.8,
    "+50% Shock": 1.5,
    "-50% Shock": 0.5
}
test_start = "2023-06-01"
test_end = "2024-06-01"

fetcher = StockDataFetcher(tickers, period="5y", interval="1d")
fetcher.fetch_all_stocks()
fetcher.add_technical_indicators()
returns_matrix = fetcher.create_returns_matrix()

generator = CNNBiLSTMViewsGenerator(len(tickers), sequence_length=30)
generator.train_all_models(fetcher.stock_data, epochs=1, batch_size=32)  # loads pre-trained
views, uncertainties = generator.generate_investor_views(fetcher.stock_data, prediction_horizon=5)

results = []

for label, factor in shock_multipliers.items():
    shocked_views = {k: v * factor for k, v in views.items()}

    optimizer = BlackLittermanOptimizer(
        returns_matrix=returns_matrix,
        market_caps=fetcher.market_caps,
        risk_free_rate=0.06
    )

    weights, _, _ = optimizer.black_litterman_optimization(
        shocked_views,
        uncertainties,
        tau=0.025
    )

    ret_window = returns_matrix.loc[test_start:test_end]
    if ret_window.empty:
        print(f"⚠️ No return data in {test_start} to {test_end}")
        continue

    weights = weights.reindex(ret_window.columns).fillna(0)
    weights = weights / weights.sum()

    port_returns = (ret_window * weights).sum(axis=1)
    total_ret = (1 + port_returns).prod() - 1
    annual_ret = (1 + total_ret) ** (252 / len(port_returns)) - 1
    volatility = port_returns.std() * np.sqrt(252)
    sharpe = (annual_ret - 0.06) / volatility if volatility > 0 else 0

    results.append({
        "Scenario": label,
        "Annualized Return": annual_ret,
        "Sharpe Ratio": sharpe
    })

df = pd.DataFrame(results)
df.to_csv("stress_test_summary.csv", index=False)
print("✅ Saved to stress_test_summary.csv")

plt.figure(figsize=(10, 5))
plt.bar(df["Scenario"], df["Sharpe Ratio"], color='skyblue')
plt.title("Stress Test: Sharpe Ratios")
plt.ylabel("Sharpe Ratio")
plt.xticks(rotation=15)
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("stress_sharpe_chart.png")
plt.show()

plt.figure(figsize=(10, 5))
plt.bar(df["Scenario"], df["Annualized Return"], color='lightgreen')
plt.title("Stress Test: Annualized Return")
plt.ylabel("Return")
plt.xticks(rotation=15)
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("stress_return_chart.png")
plt.show()
