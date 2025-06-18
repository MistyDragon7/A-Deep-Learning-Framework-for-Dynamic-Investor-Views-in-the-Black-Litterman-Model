import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import yfinance as yf
from black_litterman_optimizer import BlackLittermanOptimizer
from stock_data_fetcher import StockDataFetcher
from views_generator import CNNBiLSTMViewsGenerator
import warnings
warnings.filterwarnings('ignore')

class PortfolioBacktester:
    """
    Comprehensive backtesting system for Black-Litterman CNN-BiLSTM strategy
    """

    def __init__(self, stock_list, nifty_ticker="^NSEI"):
        self.stock_list = stock_list
        self.nifty_ticker = nifty_ticker
        self.results = {}

    def fetch_nifty_data(self, start_date, end_date):
        """Fetch Nifty 50 index data for benchmark comparison"""
        try:
            nifty_data = yf.download(self.nifty_ticker, start=start_date, end=end_date)
            nifty_returns = nifty_data['Close'].pct_change().dropna()
            return nifty_returns
        except Exception as e:
            print(f"Error fetching Nifty data: {e}")
            return pd.Series()

    def calculate_portfolio_performance(self, weights, returns_matrix, start_date, end_date):
        """Calculate portfolio performance metrics"""
        # Filter returns matrix for the specified period
        period_returns = returns_matrix.loc[start_date:end_date]

        if period_returns.empty:
            return None

        # Align weights with available stocks in returns matrix
        aligned_weights = weights.reindex(period_returns.columns, fill_value=0)
        aligned_weights = aligned_weights / aligned_weights.sum()  # Renormalize

        # Calculate daily portfolio returns
        portfolio_returns = (period_returns * aligned_weights).sum(axis=1)

        # Calculate metrics
        total_return = (1 + portfolio_returns).prod() - 1
        annualized_return = (1 + total_return) ** (252 / len(portfolio_returns)) - 1
        volatility = portfolio_returns.std() * np.sqrt(252)
        sharpe_ratio = (annualized_return - 0.06) / volatility if volatility > 0 else 0

        # Maximum drawdown
        cumulative_returns = (1 + portfolio_returns).cumprod()
        rolling_max = cumulative_returns.expanding().max()
        drawdowns = (cumulative_returns - rolling_max) / rolling_max
        max_drawdown = drawdowns.min()

        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'portfolio_returns': portfolio_returns,
            'cumulative_returns': cumulative_returns
        }

    def backtest_type_1_full_training(self, sequence_length=30, epochs=30, batch_size=32,
                                     prediction_horizon=5, risk_aversion=3.0, tau=0.025):
        """
        Backtesting Type 1: Train on full 5 years, backtest on same period
        """
        print("=" * 80)
        print("BACKTESTING TYPE 1: FULL TRAINING PERIOD")
        print("=" * 80)

        # Fetch 5 years of data
        fetcher = StockDataFetcher(self.stock_list, period="5y", interval="1d")
        stock_data = fetcher.fetch_all_stocks()

        # Filter stocks with sufficient data
        sufficient_data_stocks = {
            ticker: df for ticker, df in stock_data.items()
            if len(df) > sequence_length + prediction_horizon + 2
        }

        if len(sufficient_data_stocks) < 2:
            print("Insufficient stock data for Type 1 backtesting")
            return None

        fetcher.stock_data = sufficient_data_stocks
        fetcher.stock_list = list(sufficient_data_stocks.keys())
        fetcher.add_technical_indicators()
        returns_matrix = fetcher.create_returns_matrix()

        if returns_matrix.empty:
            print("Empty returns matrix for Type 1 backtesting")
            return None

        # Train models on full dataset
        views_generator = CNNBiLSTMViewsGenerator(len(fetcher.stock_data), sequence_length)
        views_generator.train_all_models(fetcher.stock_data, epochs=epochs, batch_size=batch_size)

        if not views_generator.models:
            print("No models trained for Type 1 backtesting")
            return None

        # Generate views
        views, view_uncertainties = views_generator.generate_investor_views(
            fetcher.stock_data, prediction_horizon
        )

        # Optimize portfolio
        bl_optimizer = BlackLittermanOptimizer(
            returns_matrix, fetcher.market_caps, risk_free_rate=0.06
        )

        optimal_weights, bl_returns, bl_cov = bl_optimizer.black_litterman_optimization(
            views, view_uncertainties, risk_aversion, tau
        )

        # Calculate performance over full period
        start_date = returns_matrix.index[0]
        end_date = returns_matrix.index[-1]

        portfolio_performance = self.calculate_portfolio_performance(
            optimal_weights, returns_matrix, start_date, end_date
        )

        # Get Nifty benchmark performance
        nifty_returns = self.fetch_nifty_data(start_date, end_date)
        nifty_performance = None

        if not nifty_returns.empty:
            nifty_total_return = (1 + nifty_returns).prod() - 1
            nifty_annualized_return = (1 + nifty_total_return) ** (252 / len(nifty_returns)) - 1
            nifty_volatility = nifty_returns.std() * np.sqrt(252)
            nifty_sharpe = (nifty_annualized_return - 0.06) / nifty_volatility if nifty_volatility > 0 else 0

            nifty_cumulative = (1 + nifty_returns).cumprod()
            nifty_rolling_max = nifty_cumulative.expanding().max()
            nifty_drawdowns = (nifty_cumulative - nifty_rolling_max) / nifty_rolling_max
            nifty_max_drawdown = nifty_drawdowns.min()

            nifty_performance = {
                'total_return': nifty_total_return,
                'annualized_return': nifty_annualized_return,
                'volatility': nifty_volatility,
                'sharpe_ratio': nifty_sharpe,
                'max_drawdown': nifty_max_drawdown,
                'cumulative_returns': nifty_cumulative
            }

        self.results['type_1'] = {
            'portfolio_performance': portfolio_performance,
            'nifty_performance': nifty_performance,
            'optimal_weights': optimal_weights,
            'views': views,
            'period': f"{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}",
            'training_period': 'Full 5 years',
            'testing_period': 'Same as training (Full 5 years)'
        }

        return self.results['type_1']

    def backtest_type_2_out_of_sample(self, sequence_length=30, epochs=30, batch_size=32,
                                     prediction_horizon=5, risk_aversion=3.0, tau=0.025):
        """
        Backtesting Type 2: Train on 3 years, test on future 2 years
        """
        print("=" * 80)
        print("BACKTESTING TYPE 2: OUT-OF-SAMPLE TESTING")
        print("=" * 80)

        # Fetch 5 years of data
        fetcher = StockDataFetcher(self.stock_list, period="5y", interval="1d")
        stock_data = fetcher.fetch_all_stocks()

        # Filter stocks with sufficient data
        sufficient_data_stocks = {
            ticker: df for ticker, df in stock_data.items()
            if len(df) > sequence_length + prediction_horizon + 2
        }

        if len(sufficient_data_stocks) < 2:
            print("Insufficient stock data for Type 2 backtesting")
            return None

        fetcher.stock_data = sufficient_data_stocks
        fetcher.stock_list = list(sufficient_data_stocks.keys())
        fetcher.add_technical_indicators()

        # Split data: 3 years training, 2 years testing
        full_returns_matrix = fetcher.create_returns_matrix()
        if full_returns_matrix.empty:
            print("Empty returns matrix for Type 2 backtesting")
            return None

        # Calculate split point (60% for training, 40% for testing)
        split_point = int(len(full_returns_matrix) * 0.6)
        split_date = full_returns_matrix.index[split_point]

        print(f"Training period: {full_returns_matrix.index[0].strftime('%Y-%m-%d')} to {split_date.strftime('%Y-%m-%d')}")
        print(f"Testing period: {split_date.strftime('%Y-%m-%d')} to {full_returns_matrix.index[-1].strftime('%Y-%m-%d')}")

        # Create training dataset (first 3 years)
        training_stock_data = {}
        for ticker, df in fetcher.stock_data.items():
            training_data = df.loc[:split_date].copy()
            if len(training_data) > sequence_length + prediction_horizon + 2:
                training_stock_data[ticker] = training_data

        if len(training_stock_data) < 2:
            print("Insufficient training data for Type 2 backtesting")
            return None

        # Create training returns matrix
        training_fetcher = StockDataFetcher(list(training_stock_data.keys()))
        training_fetcher.stock_data = training_stock_data
        training_returns_matrix = training_fetcher.create_returns_matrix()

        # Train models on training data only
        views_generator = CNNBiLSTMViewsGenerator(len(training_stock_data), sequence_length)
        views_generator.train_all_models(training_stock_data, epochs=epochs, batch_size=batch_size)

        if not views_generator.models:
            print("No models trained for Type 2 backtesting")
            return None

        # Generate views using training data
        views, view_uncertainties = views_generator.generate_investor_views(
            training_stock_data, prediction_horizon
        )

        # Optimize portfolio using training data
        bl_optimizer = BlackLittermanOptimizer(
            training_returns_matrix, training_fetcher.market_caps, risk_free_rate=0.06
        )

        optimal_weights, bl_returns, bl_cov = bl_optimizer.black_litterman_optimization(
            views, view_uncertainties, risk_aversion, tau
        )

        # Test portfolio performance on out-of-sample data (last 2 years)
        test_start_date = split_date
        test_end_date = full_returns_matrix.index[-1]

        portfolio_performance = self.calculate_portfolio_performance(
            optimal_weights, full_returns_matrix, test_start_date, test_end_date
        )

        # Get Nifty benchmark performance for test period
        nifty_returns = self.fetch_nifty_data(test_start_date, test_end_date)
        nifty_performance = None

        if not nifty_returns.empty:
            nifty_total_return = (1 + nifty_returns).prod() - 1
            nifty_annualized_return = (1 + nifty_total_return) ** (252 / len(nifty_returns)) - 1
            nifty_volatility = nifty_returns.std() * np.sqrt(252)
            nifty_sharpe = (nifty_annualized_return - 0.06) / nifty_volatility if nifty_volatility > 0 else 0

            nifty_cumulative = (1 + nifty_returns).cumprod()
            nifty_rolling_max = nifty_cumulative.expanding().max()
            nifty_drawdowns = (nifty_cumulative - nifty_rolling_max) / nifty_rolling_max
            nifty_max_drawdown = nifty_drawdowns.min()

            nifty_performance = {
                'total_return': nifty_total_return,
                'annualized_return': nifty_annualized_return,
                'volatility': nifty_volatility,
                'sharpe_ratio': nifty_sharpe,
                'max_drawdown': nifty_max_drawdown,
                'cumulative_returns': nifty_cumulative
            }

        self.results['type_2'] = {
            'portfolio_performance': portfolio_performance,
            'nifty_performance': nifty_performance,
            'optimal_weights': optimal_weights,
            'views': views,
            'split_date': split_date,
            'training_period': f"{full_returns_matrix.index[0].strftime('%Y-%m-%d')} to {split_date.strftime('%Y-%m-%d')}",
            'testing_period': f"{test_start_date.strftime('%Y-%m-%d')} to {test_end_date.strftime('%Y-%m-%d')}"
        }

        return self.results['type_2']

    def display_results(self):
        """Display comprehensive backtesting results"""
        if not self.results:
            print("No backtesting results available")
            return

        print("\n" + "=" * 100)
        print("COMPREHENSIVE BACKTESTING RESULTS")
        print("=" * 100)

        for backtest_type, results in self.results.items():
            type_name = "FULL TRAINING PERIOD" if backtest_type == 'type_1' else "OUT-OF-SAMPLE TESTING"
            print(f"\n{type_name}")
            print("-" * 80)
            print(f"Training Period: {results['training_period']}")
            print(f"Testing Period: {results['testing_period']}")

            portfolio_perf = results['portfolio_performance']
            nifty_perf = results['nifty_performance']

            if portfolio_perf:
                print(f"\n📊 PORTFOLIO PERFORMANCE:")
                print(f"Total Return: {portfolio_perf['total_return']:.2%}")
                print(f"Annualized Return: {portfolio_perf['annualized_return']:.2%}")
                print(f"Volatility: {portfolio_perf['volatility']:.2%}")
                print(f"Sharpe Ratio: {portfolio_perf['sharpe_ratio']:.3f}")
                print(f"Max Drawdown: {portfolio_perf['max_drawdown']:.2%}")

            if nifty_perf:
                print(f"\n📈 NIFTY 50 BENCHMARK:")
                print(f"Total Return: {nifty_perf['total_return']:.2%}")
                print(f"Annualized Return: {nifty_perf['annualized_return']:.2%}")
                print(f"Volatility: {nifty_perf['volatility']:.2%}")
                print(f"Sharpe Ratio: {nifty_perf['sharpe_ratio']:.3f}")
                print(f"Max Drawdown: {nifty_perf['max_drawdown']:.2%}")

                if portfolio_perf:
                    excess_return = portfolio_perf['annualized_return'] - nifty_perf['annualized_return']
                    print(f"\n🎯 EXCESS RETURN: {excess_return:.2%}")

            # Display top portfolio weights
            if 'optimal_weights' in results and not results['optimal_weights'].empty:
                top_weights = results['optimal_weights'].sort_values(ascending=False).head(10)
                print(f"\n💼 TOP 10 PORTFOLIO WEIGHTS:")
                for asset, weight in top_weights.items():
                    print(f"{asset}: {weight:.2%}")

    def plot_performance_comparison(self):
        """Plot performance comparison charts"""
        if not self.results:
            print("No results to plot")
            return

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Black-Litterman CNN-BiLSTM Strategy Performance Analysis', fontsize=16)

        colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']

        for idx, (backtest_type, results) in enumerate(self.results.items()):
            row = idx // 2
            col = idx % 2

            type_name = "Full Training" if backtest_type == 'type_1' else "Out-of-Sample"

            portfolio_perf = results['portfolio_performance']
            nifty_perf = results['nifty_performance']

            if portfolio_perf and nifty_perf:
                # Plot cumulative returns
                portfolio_cum_returns = portfolio_perf['cumulative_returns']
                nifty_cum_returns = nifty_perf['cumulative_returns']

                # Align dates
                common_dates = portfolio_cum_returns.index.intersection(nifty_cum_returns.index)
                if not common_dates.empty:
                    axes[row, col].plot(common_dates,
                                      portfolio_cum_returns.loc[common_dates],
                                      label='BL CNN-BiLSTM Strategy',
                                      color=colors[0], linewidth=2)
                    axes[row, col].plot(common_dates,
                                      nifty_cum_returns.loc[common_dates],
                                      label='Nifty 50',
                                      color=colors[1], linewidth=2)

                    axes[row, col].set_title(f'{type_name} - Cumulative Returns')
                    axes[row, col].set_xlabel('Date')
                    axes[row, col].set_ylabel('Cumulative Return')
                    axes[row, col].legend()
                    axes[row, col].grid(True, alpha=0.3)

        # If only one backtest type, use remaining subplots for metrics comparison
        if len(self.results) == 1:
            # Create metrics comparison bar chart
            result = list(self.results.values())[0]
            portfolio_perf = result['portfolio_performance']
            nifty_perf = result['nifty_performance']

            if portfolio_perf and nifty_perf:
                metrics = ['Total Return', 'Annualized Return', 'Volatility', 'Sharpe Ratio']
                portfolio_values = [
                    portfolio_perf['total_return'],
                    portfolio_perf['annualized_return'],
                    portfolio_perf['volatility'],
                    portfolio_perf['sharpe_ratio']
                ]
                nifty_values = [
                    nifty_perf['total_return'],
                    nifty_perf['annualized_return'],
                    nifty_perf['volatility'],
                    nifty_perf['sharpe_ratio']
                ]

                x = np.arange(len(metrics))
                width = 0.35

                axes[0, 1].bar(x - width/2, portfolio_values, width,
                              label='BL CNN-BiLSTM Strategy', color=colors[0])
                axes[0, 1].bar(x + width/2, nifty_values, width,
                              label='Nifty 50', color=colors[1])

                axes[0, 1].set_title('Performance Metrics Comparison')
                axes[0, 1].set_xlabel('Metrics')
                axes[0, 1].set_ylabel('Value')
                axes[0, 1].set_xticks(x)
                axes[0, 1].set_xticklabels(metrics, rotation=45)
                axes[0, 1].legend()
                axes[0, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    def run_comprehensive_backtest(self, **kwargs):
        """Run both types of backtesting"""
        print("Starting Comprehensive Backtesting...")

        # Run Type 1 backtesting
        type1_results = self.backtest_type_1_full_training(**kwargs)

        # Run Type 2 backtesting
        type2_results = self.backtest_type_2_out_of_sample(**kwargs)

        # Display results
        self.display_results()

        # Plot results
        self.plot_performance_comparison()

        return self.results
