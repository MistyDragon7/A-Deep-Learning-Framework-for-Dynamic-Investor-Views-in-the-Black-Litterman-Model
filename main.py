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
# Nifty 50 constituent stocks
NIFTY50_STOCKS = [
    'ADANIENT.NS',
    'ADANIPORTS.NS',
    'APOLLOHOSP.NS',
    'ASIANPAINT.NS',
    'AXISBANK.NS',
    'BAJAJ-AUTO.NS',
    'BAJFINANCE.NS',
    'BAJAJFINSV.NS',
    'BEL.NS',
    'BHARTIARTL.NS',
    'CIPLA.NS',
    'COALINDIA.NS',
    'DRREDDY.NS',
    'EICHERMOT.NS',
    'ETERNAL.NS',
    'GRASIM.NS',
    'HCLTECH.NS',
    'HDFCBANK.NS',
    'HDFCLIFE.NS',
    'HEROMOTOCO.NS',
    'HINDALCO.NS',
    'HINDUNILVR.NS',
    'ICICIBANK.NS',
    'ITC.NS',
    'INDUSINDBK.NS',
    'INFY.NS',
    'JSWSTEEL.NS',
    'JIOFIN.NS',
    'KOTAKBANK.NS',
    'LT.NS',
    'M&M.NS',
    'MARUTI.NS',
    'NTPC.NS',
    'NESTLEIND.NS',
    'ONGC.NS',
    'POWERGRID.NS',
    'RELIANCE.NS',
    'SBILIFE.NS',
    'SHRIRAMFIN.NS',
    'SBIN.NS',
    'SUNPHARMA.NS',
    'TCS.NS',
    'TATACONSUM.NS',
    'TATAMOTORS.NS',
    'TATASTEEL.NS',
    'TECHM.NS',
    'TITAN.NS',
    'TRENT.NS',
    'ULTRACEMCO.NS',
    'WIPRO.NS'
]

def main():
    """Main execution function"""
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
    # Using all NIFTY50_STOCKS for a potentially more robust returns matrix
    selected_stocks = NIFTY50_STOCKS  # Use the full list
    print(f"Selected stocks: {selected_stocks}")

    # Fetch data for a longer period to get more data points for training and returns matrix
    fetcher = StockDataFetcher(selected_stocks, period="5y", interval="1d")
    stock_data = fetcher.fetch_all_stocks()

    # Filter out stocks that didn't fetch enough data BEFORE adding indicators and making matrix
    sufficient_data_stocks = {
        ticker: df for ticker, df in stock_data.items()
        if len(df) > SEQUENCE_LENGTH + PREDICTION_HORIZON + 2 # Need enough data for sequence + next day + buffer
    }

    if len(sufficient_data_stocks) < 2: # Need at least 2 stocks for covariance matrix
        print(f"Insufficient stock data fetched after filtering. Only {len(sufficient_data_stocks)} stocks available.")
        return None

    print(f"\nUsing {len(sufficient_data_stocks)} stocks with sufficient data.")
    fetcher.stock_data = sufficient_data_stocks # Update fetcher's data to use only valid stocks
    fetcher.stock_list = list(sufficient_data_stocks.keys()) # Update stock list

    # Step 2: Add technical indicators
    fetcher.add_technical_indicators()

    # Step 3: Create returns matrix
    # Ensure returns matrix is created from the data AFTER adding indicators and handling NaNs/Infs
    returns_matrix = fetcher.create_returns_matrix()
    print(f"\nReturns matrix shape: {returns_matrix.shape}")

    # Ensure the returns matrix has data before proceeding
    if returns_matrix.empty or returns_matrix.shape[0] < 2 or returns_matrix.shape[1] < 2:
        print("Insufficient data in returns matrix to proceed with optimization.")
        return None


    # Step 4: Train CNN-BiLSTM models to generate views
    # Pass the filtered stock_data
    views_generator = CNNBiLSTMViewsGenerator(len(fetcher.stock_data), SEQUENCE_LENGTH)
    views_generator.train_all_models(fetcher.stock_data, epochs=EPOCHS, batch_size=BATCH_SIZE)

    # Ensure there are trained models before generating views
    if not views_generator.models:
        print("No models were successfully trained. Cannot generate views.")
        return None


    # Step 5: Generate investor views
    # Pass the filtered stock_data
    views, view_uncertainties = views_generator.generate_investor_views(
        fetcher.stock_data, PREDICTION_HORIZON
    )

    # Ensure views were generated
    if not views:
        print("No investor views were generated. Cannot perform Black-Litterman optimization.")
        # Fallback to Black-Litterman with no views (market weights)
        bl_optimizer = BlackLittermanOptimizer(
            returns_matrix, fetcher.market_caps, risk_free_rate=0.06
        )
        optimal_weights, bl_returns, bl_cov = bl_optimizer.black_litterman_optimization(
            {}, {}, RISK_AVERSION, TAU # Pass empty views and uncertainties
        )
    else:
        # Step 6: Black-Litterman optimization
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


    # Calculate portfolio metrics only if optimal weights and BL returns are available
    portfolio_return = np.nan
    portfolio_volatility = np.nan
    sharpe_ratio = np.nan

    if not optimal_weights.empty and not bl_returns.empty and not bl_cov.empty:
         try:
            # Align indices just in case
            aligned_returns = bl_returns[optimal_weights.index]
            aligned_cov = bl_cov.loc[optimal_weights.index, optimal_weights.index]

            portfolio_return = np.sum(optimal_weights * aligned_returns)
            portfolio_variance = np.dot(optimal_weights.T, np.dot(aligned_cov, optimal_weights))

            # Ensure variance is non-negative before taking sqrt
            portfolio_variance = max(0, portfolio_variance)
            portfolio_volatility = np.sqrt(portfolio_variance)

            # Avoid division by zero if volatility is zero
            if portfolio_volatility > 1e-9: # Use a small threshold
                 sharpe_ratio = (portfolio_return) / portfolio_volatility
            else:
                 sharpe_ratio = np.inf # Infinite Sharpe ratio if risk is zero (unlikely)


         except Exception as e:
             print(f"\nError calculating portfolio metrics: {e}")
             portfolio_return = np.nan
             portfolio_volatility = np.nan
             sharpe_ratio = np.nan

    print(f"\nPortfolio Metrics:")
    print(f"Expected Return: {portfolio_return:.2%}" if not np.isnan(portfolio_return) else "Expected Return: N/A")
    print(f"Volatility: {portfolio_volatility:.2%}" if not np.isnan(portfolio_volatility) else "Volatility: N/A")
    print(f"Sharpe Ratio: {sharpe_ratio:.3f}" if not np.isnan(sharpe_ratio) else "Sharpe Ratio: N/A")


    return {
        'optimal_weights': optimal_weights if not optimal_weights.empty else None,
        'views': views if views else None,
        'view_uncertainties': view_uncertainties if view_uncertainties else None,
        'bl_returns': bl_returns if not bl_returns.empty else None,
        'portfolio_metrics': {
            'return': portfolio_return,
            'volatility': portfolio_volatility,
            'sharpe_ratio': sharpe_ratio
        }
    }

if __name__ == "__main__":
    results = main()
