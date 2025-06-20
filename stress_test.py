import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from black_litterman_optimizer import BlackLittermanOptimizer
from stock_data_fetcher import StockDataFetcher
from views_generator import CNNBiLSTMViewsGenerator
import os

subset = ["RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS"]
risk_free_rate = 0.06
tau = 0.025
output_dir = "stress_test_outputs"
os.makedirs(output_dir, exist_ok=True)

fetcher = StockDataFetcher(subset, period="5y", interval="1d")
fetcher.fetch_all_stocks()
fetcher.add_technical_indicators()

returns_matrix = fetcher.create_returns_matrix()
market_caps = fetcher.market_caps

views_generator = CNNBiLSTMViewsGenerator(n_stocks=len(subset))
views_generator.train_all_models(fetcher.stock_data, epochs=0)
base_views, base_uncertainties = views_generator.generate_investor_views(fetcher.stock_data)

shock_scenarios = {
    "Baseline": base_views,
    "Bullish Shock": {k: v * 1.20 for k, v in base_views.items()},
    "Bearish Shock": {k: v * 0.80 for k, v in base_views.items()},
    "Single Asset Shock (RELIANCE)": {**base_views, "RELIANCE.NS": base_views["RELIANCE.NS"] * 1.50},
    "Sector Downside (Tech)": {
        **base_views,
        "TCS.NS": base_views["TCS.NS"] * 0.7,
        "INFY.NS": base_views["INFY.NS"] * 0.7
    },
}

weights_dict = {}
sharpe_dict = {}
returns_dict = {}

for name, scenario_views in shock_scenarios.items():
    bl = BlackLittermanOptimizer(returns_matrix, market_caps, risk_free_rate=risk_free_rate)
    weights, _, _ = bl.black_litterman_optimization(scenario_views, base_uncertainties, tau=tau)

    weights_dict[name] = weights

    aligned_weights = weights.reindex(returns_matrix.columns).fillna(0)
    port_returns = (returns_matrix * aligned_weights).sum(axis=1)

    ann_return = (1 + port_returns.mean()) ** 252 - 1
    ann_vol = port_returns.std() * np.sqrt(252)
    sharpe = (ann_return - risk_free_rate) / ann_vol if ann_vol > 0 else 0

    sharpe_dict[name] = sharpe
    returns_dict[name] = ann_return

df_weights = pd.DataFrame(weights_dict).T.fillna(0)
df_sharpe = pd.Series(sharpe_dict, name="Sharpe Ratio")
df_returns = pd.Series(returns_dict, name="Annualized Return")

df_weights.to_csv(f"{output_dir}/stress_weights.csv")
df_sharpe.to_csv(f"{output_dir}/stress_sharpe.csv")
df_returns.to_csv(f"{output_dir}/stress_returns.csv")
print("✅ CSV files saved to:", output_dir)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
df_returns.sort_values(ascending=False).plot(kind='bar', ax=axes[0], color='navy', title="Annualized Returns")
df_sharpe.sort_values(ascending=False).plot(kind='bar', ax=axes[1], color='darkgreen', title="Sharpe Ratios")
for ax in axes:
    ax.set_xlabel("Stress Scenario")
    ax.set_ylabel("Metric Value")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=25)
    ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"{output_dir}/stress_metrics_comparison.png", dpi=300)
plt.show()
print("✅ Charts saved as PNG")
