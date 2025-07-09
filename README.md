# A Deep Learning Framework for Dynamic Investor Views in the Black-Litterman Model

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
├── backtrader_tester.py        # Backtesting engine
├── stock_data_fetcher.py          # Data pipeline using yFinance and feature engineering
├── technical_indicators.py        # Custom technical indicators used as model features
├── views_generator.py             # CNN-BiLSTM + MC Dropout view generator
├── stress_test.py                 # Evaluates model robustness under extreme market conditions
├── sensitivity_analysis.py        # Analyzes parameter impact on portfolio performance
├── precomputed_weights.parquet    # Precomputed optimal weights for various market regimes
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

The backtesting process is designed to be efficient by separating model training from strategy evaluation. It involves two main steps to avoid unnecessary re-running of TensorFlow models within the backtesting loop:

1.  **Pre-computation of Views**: CNN-BiLSTM models are trained once on historical data to generate expected returns and uncertainties (views). These views are then saved.
2.  **Strategy Simulation**: The Black-Litterman optimization and portfolio allocation are performed using the pre-computed views in a bi-weekly rebalanced out-of-sample simulation.

---

## Additional Files

- `requirements.txt`: Lists all Python dependencies required to run the project.
- `precomputed_weights.parquet`: Stores optimal portfolio weights computed by the Black-Litterman model, used for efficient backtesting and analysis.

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
- Key Libraries: `yfinance`, `scikit-learn`, `joblib`, `pandas`, `matplotlib`, `seaborn`
- Deep Learning & Utilities: `tensorflow-io-gcs-filesystem`, `absl-py`, `astunparse`, `flatbuffers`, `gast`, `google-pasta`, `grpcio`, `h5py`, `keras`, `libclang`, `markdown`, `markdown-it-py`, `markupsafe`, `ml_dtypes`, `namex`, `opt_einsum`, `optree`, `protobuf`, `pygments`, `rich`, `tensorboard`, `tensorboard-data-server`, `termcolor`, `wrapt`
- Data Management & Web: `peewee`, `websockets`, `multitasking`
- Scientific Computing: `scipy`

Install dependencies:

```bash
pip install -r requirements.txt
```

### Run Backtest

```bash
python main.py --stage precompute
python main.py --stage backtest
```

All results will be saved in the `results/` directory, including performance metrics, weights, and plots.

---

## Configuration

Modify these parameters inside `main.py`:

```python
sequence_length = 30
epochs = 2
batch_size = 34
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
   _Asset Allocation: Combining Investor Views with Market Equilibrium._  
   _Journal of Fixed Income._  
   ➤ Introduced the Black-Litterman model for portfolio optimization.

2. **Idzorek, T. M. (2005).**  
   _A Step-By-Step Guide to the Black-Litterman Model: Incorporating User-Specified Confidence Levels._  
   _Ibbotson Associates._  
   ➤ Enhanced the Black-Litterman model by integrating confidence levels into investor views.

3. **Barua, R., & Sharma, A. K. (2022).**  
   _Dynamic Black-Litterman portfolios with views derived via CNN-BiLSTM predictions._  
   _Finance Research Letters, 49, 103111._  
   [https://doi.org/10.1016/j.frl.2022.103111](https://doi.org/10.1016/j.frl.2022.103111)  
   ➤ Used CNN-BiLSTM models to generate absolute views and uncertainties in a dynamic Black-Litterman optimization framework, showing superior performance vs benchmarks.
