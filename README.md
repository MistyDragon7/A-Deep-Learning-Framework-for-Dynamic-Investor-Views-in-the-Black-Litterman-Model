
# Black-Litterman + CNN-BiLSTM Portfolio Optimizer for NIFTY 50

This repository contains a comprehensive portfolio optimization pipeline that integrates the **Black-Litterman model** with **machine learning-based investor views**, using CNN-BiLSTM networks and Monte Carlo Dropout to model return expectations and uncertainty.

## Project Highlights

- **Black-Litterman Optimization**: Market-consistent asset allocation framework with dynamic risk aversion.
- **Machine Learning Views**: Forecasting expected returns and confidence using CNN-BiLSTM models.
- **Monte Carlo Dropout**: Uncertainty estimation for model confidence via Bayesian deep learning approximation.
- **Backtesting Framework**: Full training and out-of-sample (bi-weekly rebalanced) evaluations.
- **Benchmarking**: Compared against NIFTY 50 Index with visualizations and excess return computation.

---

## Repository Structure

```
.
├── main.py                         # Entry point to run full backtest
├── black_litterman_optimizer.py   # Core optimizer with market weight, lambda, and BL integration
├── portfolio_backtester.py        # Backtesting engine (Type 1 and Type 2)
├── stock_data_fetcher.py          # Data pipeline using yFinance and feature engineering
├── technical_indicators.py        # Custom technical indicators used as model features
├── views_generator.py             # CNN-BiLSTM + MC Dropout view generator
├── results/                       # Output directory for weights, metrics, plots
└── data/                          # Frozen stock data cache for reproducibility
```

---

## How It Works

### 1. Fetch Data
Retrieves historical daily prices and market caps for NIFTY 50 constituents via Yahoo Finance.

### 2. Feature Engineering
Computes technical indicators (RSI, Bollinger Bands, MACD, volatility, etc.) to enhance signal quality.

### 3. Train CNN-BiLSTM Models
Each stock has its own deep learning model trained on a sliding window of historical data to predict cumulative 5-day returns.

### 4. Generate Views
Expected returns and uncertainties are extracted using **Monte Carlo dropout** inference.

### 5. Black-Litterman Integration
Combines implied equilibrium returns with model-driven views via:
- `τ`: view uncertainty scalar
- `λ`: dynamic risk aversion estimated from market conditions

### 6. Portfolio Optimization
Allocates capital across assets using mean-variance optimization adjusted for view-integrated expected returns.

### 7. Backtesting
Supports:
- **Type 1**: Full in-sample training and testing
- **Type 2**: Out-of-sample with bi-weekly rebalancing

---

## Sample Output

- `performance_comparison.png`: Cumulative return plots for strategy vs NIFTY 50.
- `backtest_summary.csv`: Sharpe ratio, volatility, max drawdown, excess return.
- `weights_type_*.csv`: Final portfolio weights for each backtest.
- `views_type_*.csv`: CNN-BiLSTM model outputs.
- `cumulative_returns_type_*.csv`: Strategy growth over time.

---

## 🛠️ Usage

### Requirements
- Python 3.8+
- TensorFlow 2.10+
- yFinance, scikit-learn, joblib, pandas, matplotlib, seaborn

Install dependencies:

```bash
pip install -r requirements.txt
```

### Run Backtest

```bash
python main.py
```

All results will be saved in the `results/` directory, including performance metrics, weights, and plots.

---

## Configuration

Modify these parameters inside `main.py`:

```python
sequence_length = 30
epochs = 25
batch_size = 32
prediction_horizon = 5
tau = 0.025
use_frozen_data = True  # speeds up testing
```

---

## Notes on ML Views

Each model:
- Uses 19 technical features
- Employs Gaussian noise, dropout, and batch normalization
- Is trained individually per stock with early stopping
- Outputs **expected 5-day return** + **uncertainty** via standard deviation over 50 MC dropout samples

---

## Evaluation Metrics

- **Annualized Return**
- **Volatility**
- **Sharpe Ratio**
- **Max Drawdown**
- **Excess Return over NIFTY 50**

---

## Credits

Developed for academic and quant research purposes. Inspired by the integration of subjective ML views into probabilistic portfolio construction frameworks.

## References

1. **Black, F. & Litterman, R. (1991).**  
   *Asset Allocation: Combining Investor Views with Market Equilibrium.*  
   *Journal of Fixed Income.*  
   ➤ Introduced the Black-Litterman model for portfolio optimization.

2. **Idzorek, T. M. (2005).**  
   *A Step-By-Step Guide to the Black-Litterman Model: Incorporating User-Specified Confidence Levels.*  
   *Ibbotson Associates.*  
   ➤ Enhanced the Black-Litterman model by integrating confidence levels into investor views.

3. **Barua, R., & Sharma, A. K. (2022).**  
   *Dynamic Black-Litterman portfolios with views derived via CNN-BiLSTM predictions.*  
   *Finance Research Letters, 49, 103111.*  
   [https://doi.org/10.1016/j.frl.2022.103111](https://doi.org/10.1016/j.frl.2022.103111)  
   ➤ Used CNN-BiLSTM models to generate absolute views and uncertainties in a dynamic Black-Litterman optimization framework, showing superior performance vs benchmarks.
