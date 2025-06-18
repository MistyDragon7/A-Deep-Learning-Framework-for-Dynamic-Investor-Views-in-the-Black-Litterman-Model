import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Conv1D, MaxPooling1D,
                                   Bidirectional, LSTM, Dense, Dropout,
                                   BatchNormalization, Concatenate)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')

from black_litterman_optimizer import BlackLittermanOptimizer
from stock_data_fetcher import StockDataFetcher
from views_generator import CNNBiLSTMViewsGenerator
from technical_indicators import TechnicalIndicators
from backtesting_system import PortfolioBacktester  # Import our new backtesting system

# Nifty 50 constituent stocks
NIFTY50_STOCKS = [
    'ADANIENT.NS', 'ADANIPORTS.NS', 'APOLLOHOSP.NS', 'ASIANPAINT.NS', 'AXISBANK.NS',
    'BAJAJ-AUTO.NS', 'BAJFINANCE.NS', 'BAJAJFINSV.NS', 'BEL.NS', 'BHARTIARTL.NS',
    'CIPLA.NS', 'COALINDIA.NS', 'DRREDDY.NS', 'EICHERMOT.NS', 'ETERNAL.NS',
    'GRASIM.NS', 'HCLTECH.NS', 'HDFCBANK.NS', 'HDFCLIFE.NS', 'HEROMOTOCO.NS',
    'HINDALCO.NS', 'HINDUNILVR.NS', 'ICICIBANK.NS', 'ITC.NS', 'INDUSINDBK.NS',
    'INFY.NS', 'JSWSTEEL.NS', 'JIOFIN.NS', 'KOTAKBANK.NS', 'LT.NS',
    'M&M.NS', 'MARUTI.NS', 'NTPC.NS', 'NESTLEIND.NS', 'ONGC.NS',
    'POWERGRID.NS', 'RELIANCE.NS', 'SBILIFE.NS', 'SHRIRAMFIN.NS', 'SBIN.NS',
    'SUNPHARMA.NS', 'TCS.NS', 'TATACONSUM.NS', 'TATAMOTORS.NS', 'TATASTEEL.NS',
    'TECHM.NS', 'TITAN.NS', 'TRENT.NS', 'ULTRACEMCO.NS', 'WIPRO.NS'
]

def run_single_optimization():
    """Original single optimization function (for current approach)"""
    print("CNN-BiLSTM Black-Litterman Portfolio Optimization")
    print("=" * 60)

    # Configuration
    SEQUENCE_LENGTH = 30
    EPOCHS = 30
    BATCH_SIZE = 32
    PREDICTION_HORIZON = 5
    RISK_AVERSION = 3.0
    TAU = 0.025

    # Step 1: Fetch stock data
    selected_stocks = NIFTY50_STOCKS
    print(f"Selected stocks: {len(selected_stocks)} stocks")

    fetcher = StockDataFetcher(selected_stocks, period="5y", interval="1d")
    stock_data = fetcher.fetch_all_stocks()

    # Filter out stocks that didn't fetch enough data
    sufficient_data_stocks = {
        ticker: df for ticker, df in stock_data.items()
        if len(df) > SEQUENCE_LENGTH + PREDICTION_HORIZON + 2
    }

    if len(sufficient_data_stocks) < 2:
        print(f"Insufficient stock data fetched after filtering. Only {len(sufficient_data_stocks)} stocks available.")
        return None

    print(f"\nUsing {len(sufficient_data_stocks)} stocks with sufficient data.")
    fetcher.stock_data = sufficient_data_stocks
    fetcher.stock_list = list(sufficient_data_stocks.keys())

    # Step 2: Add technical indicators
    fetcher.add_technical_indicators()

    # Step 3: Create returns matrix
    returns_matrix = fetcher.create_returns_matrix()
    print(f"\nReturns matrix shape: {returns_matrix.shape}")

    if returns_matrix.empty or returns_matrix.shape[0] < 2 or returns_matrix.shape[1] < 2:
        print("Insufficient data in returns matrix to proceed with optimization.")
        return None

    # Step 4: Train CNN-BiLSTM models
    views_generator = CNNBiLSTMViewsGenerator(len(fetcher.stock_data), SEQUENCE_LENGTH)
    views_generator.train_all_models(fetcher.stock_data, epochs=EPOCHS, batch_size=BATCH_SIZE)

    if not views_generator.models:
        print("No models were successfully trained. Cannot generate views.")
        return None

    # Step 5: Generate investor views
    views, view_uncertainties = views_generator.generate_investor_views(
        fetcher.stock_data, PREDICTION_HORIZON
    )

    # Step 6: Black-Litterman optimization
    if not views:
        print("No investor views were generated. Using market weights.")
        bl_optimizer = BlackLittermanOptimizer(
            returns_matrix, fetcher.market_caps, risk_free_rate=0.06
        )
        optimal_weights, bl_returns, bl_cov = bl_optimizer.black_litterman_optimization(
            {}, {}, RISK_AVERSION, TAU
        )
    else:
        bl_optimizer = BlackLittermanOptimizer(
            returns_matrix, fetcher.market_caps, risk_free_rate=0.06
        )
        optimal_weights, bl_returns, bl_cov = bl_optimizer.black_litterman_optimization(
            views, view_uncertainties, RISK_AVERSION, TAU
        )

    # Step 7: Display results
    print("\n" + "=" * 60)
    print("PORTFOLIO OPTIMIZATION RESULTS")
    print("=" * 60)

    print(f"\nMarket Capitalization Weights:")
    if not bl_optimizer.market_weights.empty:
        for asset, weight in bl_optimizer.market_weights.items():
            print(f"{asset}: {weight:.2%}")
    else:
        print("Market weights could not be calculated.")

    print(f"\nCNN-BiLSTM Generated Views (Expected Returns):")
    if views:
        for asset, view in views.items():
            uncertainty = view_uncertainties.get(asset, 0)
            print(f"{asset}: {view:.4f} ± {uncertainty:.4f}")
    else:
        print("No CNN-BiLSTM views were generated.")

    print(f"\nBlack-Litterman Optimal Weights:")
    if not optimal_weights.empty:
        total_weight = 0
        for asset, weight in optimal_weights.items():
            print(f"{asset}: {weight:.2%}")
            total_weight += weight
        print(f"Total weight: {total_weight:.2%}")
    else:
        print("Optimal weights could not be calculated.")

    print(f"\nBlack-Litterman Expected Returns (Annualized):")
    if not bl_returns.empty:
        for asset, ret in bl_returns.items():
            print(f"{asset}: {ret:.2%}")
    else:
        print("Black-Litterman expected returns could not be calculated.")

    # Calculate portfolio metrics
    portfolio_return = np.nan
    portfolio_volatility = np.nan
    sharpe_ratio = np.nan

    if not optimal_weights.empty and not bl_returns.empty and not bl_cov.empty:
        try:
            aligned_returns = bl_returns[optimal_weights.index]
            aligned_cov = bl_cov.loc[optimal_weights.index, optimal_weights.index]

            portfolio_return = np.sum(optimal_weights * aligned_returns)
            portfolio_variance = np.dot(optimal_weights.T, np.dot(aligned_cov, optimal_weights))
            portfolio_variance = max(0, portfolio_variance)
            portfolio_volatility = np.sqrt(portfolio_variance)

            if portfolio_volatility > 1e-9:
                sharpe_ratio = (portfolio_return - 0.06) / portfolio_volatility  # Assuming 6% risk-free rate
            else:
                sharpe_ratio = np.inf

        except Exception as e:
            print(f"\nError calculating portfolio metrics: {e}")
            portfolio_return = np.nan
            portfolio_volatility = np.nan
            sharpe_ratio = np.nan

    # Step 8: Display portfolio metrics
    print(f"\n" + "=" * 60)
    print("PORTFOLIO PERFORMANCE METRICS")
    print("=" * 60)

    if not np.isnan(portfolio_return):
        print(f"Expected Portfolio Return (Annualized): {portfolio_return:.2%}")
    else:
        print("Expected Portfolio Return: Could not be calculated")

    if not np.isnan(portfolio_volatility):
        print(f"Portfolio Volatility (Annualized): {portfolio_volatility:.2%}")
    else:
        print("Portfolio Volatility: Could not be calculated")

    if not np.isnan(sharpe_ratio) and not np.isinf(sharpe_ratio):
        print(f"Sharpe Ratio: {sharpe_ratio:.4f}")
    else:
        print("Sharpe Ratio: Could not be calculated")

    # Step 9: Create summary results dictionary
    results = {
        'optimal_weights': optimal_weights,
        'bl_returns': bl_returns,
        'bl_covariance': bl_cov,
        'views': views,
        'view_uncertainties': view_uncertainties,
        'market_weights': bl_optimizer.market_weights if bl_optimizer else None,
        'portfolio_return': portfolio_return,
        'portfolio_volatility': portfolio_volatility,
        'sharpe_ratio': sharpe_ratio,
        'returns_matrix': returns_matrix,
        'stock_data': fetcher.stock_data,
        'models': views_generator.models if views_generator else None
    }

    print(f"\n" + "=" * 60)
    print("OPTIMIZATION COMPLETED SUCCESSFULLY")
    print("=" * 60)

    return results


def run_backtesting_optimization():
    """Enhanced function with backtesting capabilities"""
    print("CNN-BiLSTM Black-Litterman Portfolio Optimization with Backtesting")
    print("=" * 70)

    # Configuration
    SEQUENCE_LENGTH = 30
    EPOCHS = 30
    BATCH_SIZE = 32
    PREDICTION_HORIZON = 5
    RISK_AVERSION = 3.0
    TAU = 0.025
    INITIAL_CAPITAL = 1000000  # 1 million
    REBALANCE_FREQUENCY = 'monthly'  # 'daily', 'weekly', 'monthly', 'quarterly'

    # Step 1: Fetch stock data
    selected_stocks = NIFTY50_STOCKS
    print(f"Selected stocks: {len(selected_stocks)} stocks")

    fetcher = StockDataFetcher(selected_stocks, period="5y", interval="1d")
    stock_data = fetcher.fetch_all_stocks()

    # Filter out stocks that didn't fetch enough data
    sufficient_data_stocks = {
        ticker: df for ticker, df in stock_data.items()
        if len(df) > SEQUENCE_LENGTH + PREDICTION_HORIZON + 100  # More data needed for backtesting
    }

    if len(sufficient_data_stocks) < 2:
        print(f"Insufficient stock data for backtesting. Only {len(sufficient_data_stocks)} stocks available.")
        return None

    print(f"\nUsing {len(sufficient_data_stocks)} stocks with sufficient data for backtesting.")
    fetcher.stock_data = sufficient_data_stocks
    fetcher.stock_list = list(sufficient_data_stocks.keys())

    # Step 2: Add technical indicators
    fetcher.add_technical_indicators()

    # Step 3: Create returns matrix
    returns_matrix = fetcher.create_returns_matrix()
    print(f"\nReturns matrix shape: {returns_matrix.shape}")

    if returns_matrix.empty or returns_matrix.shape[0] < 100:  # Need more data for backtesting
        print("Insufficient data in returns matrix to proceed with backtesting.")
        return None

    # Step 4: Initialize backtesting system
    backtester = PortfolioBacktester(
        stock_data=fetcher.stock_data,
        initial_capital=INITIAL_CAPITAL,
        transaction_cost=0.001,  # 0.1% transaction cost
        rebalance_frequency=REBALANCE_FREQUENCY
    )

    # Step 5: Run backtesting with CNN-BiLSTM Black-Litterman strategy
    backtest_results = backtester.run_backtest(
        optimization_func=lambda data, start_date, end_date: _optimize_portfolio_for_period(
            data, start_date, end_date, SEQUENCE_LENGTH, EPOCHS, BATCH_SIZE,
            PREDICTION_HORIZON, RISK_AVERSION, TAU, fetcher.market_caps
        ),
        start_date=None,  # Will use default split
        end_date=None
    )

    # Step 6: Display backtesting results
    if backtest_results:
        print(f"\n" + "=" * 70)
        print("BACKTESTING RESULTS")
        print("=" * 70)

        performance_metrics = backtest_results.get('performance_metrics', {})

        print(f"Total Return: {performance_metrics.get('total_return', 0):.2%}")
        print(f"Annualized Return: {performance_metrics.get('annualized_return', 0):.2%}")
        print(f"Annualized Volatility: {performance_metrics.get('annualized_volatility', 0):.2%}")
        print(f"Sharpe Ratio: {performance_metrics.get('sharpe_ratio', 0):.4f}")
        print(f"Maximum Drawdown: {performance_metrics.get('max_drawdown', 0):.2%}")
        print(f"Calmar Ratio: {performance_metrics.get('calmar_ratio', 0):.4f}")

        # Additional metrics
        portfolio_values = backtest_results.get('portfolio_values', pd.Series())
        if not portfolio_values.empty:
            print(f"Final Portfolio Value: ${portfolio_values.iloc[-1]:,.2f}")
            print(f"Number of Rebalancing Periods: {len(backtest_results.get('rebalancing_dates', []))}")

    return backtest_results


def _optimize_portfolio_for_period(stock_data, start_date, end_date, sequence_length,
                                 epochs, batch_size, prediction_horizon, risk_aversion,
                                 tau, market_caps):
    """
    Helper function to optimize portfolio for a specific time period during backtesting
    """
    try:
        # Filter data for the training period (before start_date)
        training_data = {}
        for ticker, df in stock_data.items():
            if start_date:
                train_df = df[df.index < start_date].copy()
            else:
                train_df = df.copy()

            if len(train_df) > sequence_length + prediction_horizon:
                training_data[ticker] = train_df

        if len(training_data) < 2:
            return None

        # Create returns matrix for training period
        returns_list = []
        common_dates = None

        for ticker, df in training_data.items():
            returns = df['Close'].pct_change().dropna()
            if common_dates is None:
                common_dates = returns.index
            else:
                common_dates = common_dates.intersection(returns.index)

        for ticker, df in training_data.items():
            returns = df['Close'].pct_change().dropna()
            aligned_returns = returns.loc[common_dates]
            returns_list.append(aligned_returns)

        returns_matrix = pd.concat(returns_list, axis=1, keys=training_data.keys())
        returns_matrix = returns_matrix.dropna()

        if returns_matrix.empty or returns_matrix.shape[0] < 30:
            return None

        # Train models and generate views
        views_generator = CNNBiLSTMViewsGenerator(len(training_data), sequence_length)
        views_generator.train_all_models(training_data, epochs=epochs, batch_size=batch_size)

        if not views_generator.models:
            return None

        views, view_uncertainties = views_generator.generate_investor_views(
            training_data, prediction_horizon
        )

        # Black-Litterman optimization
        bl_optimizer = BlackLittermanOptimizer(
            returns_matrix, market_caps, risk_free_rate=0.06
        )

        optimal_weights, bl_returns, bl_cov = bl_optimizer.black_litterman_optimization(
            views, view_uncertainties, risk_aversion, tau
        )

        return optimal_weights if not optimal_weights.empty else None

    except Exception as e:
        print(f"Error in portfolio optimization for period {start_date} to {end_date}: {e}")
        return None


# Main execution function
def main():
    """
    Main function to run either single optimization or backtesting
    """
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == '--backtest':
        results = run_backtesting_optimization()
    else:
        results = run_single_optimization()

    return results


if __name__ == "__main__":
    results = main()
