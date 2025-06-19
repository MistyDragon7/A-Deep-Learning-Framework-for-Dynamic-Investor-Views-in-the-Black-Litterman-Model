import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from portfolio_backtester import PortfolioBacktester
from stock_data_fetcher import StockDataFetcher
from black_litterman_optimizer import BlackLittermanOptimizer
from views_generator import CNNBiLSTMViewsGenerator
import os
import warnings
warnings.filterwarnings('ignore')

class TauSensitivityAnalysis:
    """
    Perform sensitivity analysis on the tau parameter in Black-Litterman optimization
    """

    def __init__(self, stock_list, tau_range=None):
        self.stock_list = stock_list
        self.tau_range = tau_range if tau_range is not None else np.arange(0.005, 0.101, 0.005)
        self.results = {}

    def run_sensitivity_analysis(self, sequence_length=30, epochs=25, batch_size=32,
                                prediction_horizon=5, output_dir="sensitivity_results",
                                use_frozen_data=True, frozen_data_path="data/frozen_data.pkl"):
        """
        Run sensitivity analysis across different tau values
        """
        print("=" * 80)
        print("STARTING TAU PARAMETER SENSITIVITY ANALYSIS")
        print("=" * 80)
        print(f"Testing tau values: {list(self.tau_range)}")

        os.makedirs(output_dir, exist_ok=True)

        # Prepare data once
        fetcher = StockDataFetcher(self.stock_list, period="5y", interval="1d")
        if use_frozen_data and os.path.exists(frozen_data_path):
            from portfolio_backtester import load_frozen_data
            load_frozen_data(fetcher, frozen_data_path)
        else:
            fetcher.fetch_all_stocks()
            from portfolio_backtester import save_frozen_data
            save_frozen_data(fetcher, frozen_data_path)
        fetcher.add_technical_indicators()

        # Filter stocks with sufficient data
        sufficient_data_stocks = {
            ticker: df for ticker, df in fetcher.stock_data.items()
            if len(df) > sequence_length + prediction_horizon + 2
        }

        if len(sufficient_data_stocks) < 2:
            print("Insufficient stock data for sensitivity analysis")
            return None

        fetcher.stock_data = sufficient_data_stocks
        fetcher.stock_list = list(sufficient_data_stocks.keys())
        returns_matrix = fetcher.create_returns_matrix()

        if returns_matrix.empty:
            print("Empty returns matrix for sensitivity analysis")
            return None

        # Train models once
        print("Training CNN-BiLSTM models...")
        views_generator = CNNBiLSTMViewsGenerator(len(fetcher.stock_data), sequence_length)
        views_generator.train_all_models(fetcher.stock_data, epochs=epochs, batch_size=batch_size)

        if not views_generator.models:
            print("No models trained for sensitivity analysis")
            return None

        # Generate views once
        views, view_uncertainties = views_generator.generate_investor_views(
            fetcher.stock_data, prediction_horizon
        )

        # Test different tau values
        for tau in self.tau_range:
            print(f"\n--- Testing tau = {tau:.3f} ---")

            # Create optimizer
            bl_optimizer = BlackLittermanOptimizer(
                returns_matrix, fetcher.market_caps, risk_free_rate=0.06
            )

            # Optimize portfolio with current tau
            optimal_weights, bl_returns, bl_cov = bl_optimizer.black_litterman_optimization(
                views, view_uncertainties, tau=tau
            )

            # Calculate performance metrics
            start_date = returns_matrix.index[0]
            end_date = returns_matrix.index[-1]

            # Portfolio performance
            portfolio_performance = self._calculate_performance_metrics(
                optimal_weights, returns_matrix, start_date, end_date
            )

            # Store results
            self.results[tau] = {
                'optimal_weights': optimal_weights,
                'bl_returns': bl_returns,
                'bl_cov': bl_cov,
                'portfolio_performance': portfolio_performance,
                'risk_aversion': bl_optimizer.dynamic_risk_aversion,
                'market_weights': bl_optimizer.market_weights
            }

            if portfolio_performance:
                print(f"Tau {tau:.3f}: Return={portfolio_performance['annualized_return']:.2%}, "
                      f"Sharpe={portfolio_performance['sharpe_ratio']:.3f}, "
                      f"Volatility={portfolio_performance['volatility']:.2%}")

        # Generate analysis report
        self._generate_analysis_report(output_dir)
        self._plot_sensitivity_results(output_dir)
        self._save_detailed_results(output_dir)

        return self.results

    def _calculate_performance_metrics(self, weights, returns_matrix, start_date, end_date):
        """Calculate portfolio performance metrics"""
        period_returns = returns_matrix.loc[start_date:end_date]

        if period_returns.empty:
            return None

        # Align weights with available stocks
        aligned_weights = weights.reindex(period_returns.columns, fill_value=0)
        aligned_weights = aligned_weights / aligned_weights.sum()

        # Calculate portfolio returns
        portfolio_returns = (period_returns * aligned_weights).sum(axis=1)

        # Calculate metrics
        total_return = float((1 + portfolio_returns).prod() - 1)
        annualized_return = float((1 + total_return) ** (252 / len(portfolio_returns)) - 1)
        volatility = float(portfolio_returns.std() * np.sqrt(252))
        sharpe_ratio = (annualized_return - 0.06) / volatility if volatility > 0 else 0

        # Maximum drawdown
        cumulative_returns = (1 + portfolio_returns).cumprod()
        rolling_max = cumulative_returns.expanding().max()
        drawdowns = (cumulative_returns - rolling_max) / rolling_max
        max_drawdown = float(drawdowns.min())

        # Weight concentration (Herfindahl index)
        herfindahl_index = (aligned_weights ** 2).sum()

        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'herfindahl_index': herfindahl_index,
            'portfolio_returns': portfolio_returns,
            'cumulative_returns': cumulative_returns,
            'weights': aligned_weights
        }

    def _generate_analysis_report(self, output_dir):
        """Generate comprehensive analysis report"""
        print("\nGenerating sensitivity analysis report...")

        # Create summary dataframe
        summary_data = []
        for tau, result in self.results.items():
            perf = result['portfolio_performance']
            if perf:
                summary_data.append({
                    'Tau': tau,
                    'Annualized_Return': perf['annualized_return'],
                    'Volatility': perf['volatility'],
                    'Sharpe_Ratio': perf['sharpe_ratio'],
                    'Max_Drawdown': perf['max_drawdown'],
                    'Herfindahl_Index': perf['herfindahl_index'],
                    'Risk_Aversion': result['risk_aversion']
                })

        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(os.path.join(output_dir, 'tau_sensitivity_summary.csv'), index=False)

        # Find optimal tau values
        if not summary_df.empty:
            best_return_tau = summary_df.loc[summary_df['Annualized_Return'].idxmax(), 'Tau']
            best_sharpe_tau = summary_df.loc[summary_df['Sharpe_Ratio'].idxmax(), 'Tau']
            min_vol_tau = summary_df.loc[summary_df['Volatility'].idxmin(), 'Tau']

            print(f"\nSENSITIVITY ANALYSIS SUMMARY:")
            print(f"Best Return Tau: {best_return_tau:.3f} ({summary_df.loc[summary_df['Tau']==best_return_tau, 'Annualized_Return'].iloc[0]:.2%})")
            print(f"Best Sharpe Tau: {best_sharpe_tau:.3f} ({summary_df.loc[summary_df['Tau']==best_sharpe_tau, 'Sharpe_Ratio'].iloc[0]:.3f})")
            print(f"Min Volatility Tau: {min_vol_tau:.3f} ({summary_df.loc[summary_df['Tau']==min_vol_tau, 'Volatility'].iloc[0]:.2%})")

        return summary_df

    def _plot_sensitivity_results(self, output_dir):
        """Create visualization plots for sensitivity analysis"""
        print("Creating sensitivity analysis plots...")

        # Prepare data for plotting
        tau_values = list(self.results.keys())
        returns = [self.results[tau]['portfolio_performance']['annualized_return']
                  for tau in tau_values if self.results[tau]['portfolio_performance']]
        volatilities = [self.results[tau]['portfolio_performance']['volatility']
                       for tau in tau_values if self.results[tau]['portfolio_performance']]
        sharpe_ratios = [self.results[tau]['portfolio_performance']['sharpe_ratio']
                        for tau in tau_values if self.results[tau]['portfolio_performance']]
        max_drawdowns = [self.results[tau]['portfolio_performance']['max_drawdown']
                        for tau in tau_values if self.results[tau]['portfolio_performance']]
        herfindahl_indices = [self.results[tau]['portfolio_performance']['herfindahl_index']
                             for tau in tau_values if self.results[tau]['portfolio_performance']]

        # Create subplot figure
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Tau Parameter Sensitivity Analysis', fontsize=16, fontweight='bold')

        # Plot 1: Returns vs Tau
        axes[0, 0].plot(tau_values, [r*100 for r in returns], 'b-o', linewidth=2, markersize=6)
        axes[0, 0].set_xlabel('Tau Parameter')
        axes[0, 0].set_ylabel('Annualized Return (%)')
        axes[0, 0].set_title('Portfolio Returns vs Tau')
        axes[0, 0].grid(True, alpha=0.3)

        # Plot 2: Volatility vs Tau
        axes[0, 1].plot(tau_values, [v*100 for v in volatilities], 'r-o', linewidth=2, markersize=6)
        axes[0, 1].set_xlabel('Tau Parameter')
        axes[0, 1].set_ylabel('Volatility (%)')
        axes[0, 1].set_title('Portfolio Volatility vs Tau')
        axes[0, 1].grid(True, alpha=0.3)

        # Plot 3: Sharpe Ratio vs Tau
        axes[0, 2].plot(tau_values, sharpe_ratios, 'g-o', linewidth=2, markersize=6)
        axes[0, 2].set_xlabel('Tau Parameter')
        axes[0, 2].set_ylabel('Sharpe Ratio')
        axes[0, 2].set_title('Sharpe Ratio vs Tau')
        axes[0, 2].grid(True, alpha=0.3)

        # Plot 4: Max Drawdown vs Tau
        axes[1, 0].plot(tau_values, [d*100 for d in max_drawdowns], 'orange', marker='o', linewidth=2, markersize=6)
        axes[1, 0].set_xlabel('Tau Parameter')
        axes[1, 0].set_ylabel('Max Drawdown (%)')
        axes[1, 0].set_title('Maximum Drawdown vs Tau')
        axes[1, 0].grid(True, alpha=0.3)

        # Plot 5: Herfindahl Index vs Tau (Portfolio Concentration)
        axes[1, 1].plot(tau_values, herfindahl_indices, 'purple', marker='o', linewidth=2, markersize=6)
        axes[1, 1].set_xlabel('Tau Parameter')
        axes[1, 1].set_ylabel('Herfindahl Index')
        axes[1, 1].set_title('Portfolio Concentration vs Tau')
        axes[1, 1].grid(True, alpha=0.3)

        # Plot 6: Risk-Return Scatter
        axes[1, 2].scatter([v*100 for v in volatilities], [r*100 for r in returns],
                          c=tau_values, cmap='viridis', s=60, alpha=0.7)
        axes[1, 2].set_xlabel('Volatility (%)')
        axes[1, 2].set_ylabel('Annualized Return (%)')
        axes[1, 2].set_title('Risk-Return Profile')
        cbar = plt.colorbar(axes[1, 2].collections[0], ax=axes[1, 2])
        cbar.set_label('Tau Parameter')
        axes[1, 2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'tau_sensitivity_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Saved sensitivity plots to {output_dir}/tau_sensitivity_analysis.png")

    def _save_detailed_results(self, output_dir):
        """Save detailed results for each tau value"""
        print("Saving detailed results...")

        for tau, result in self.results.items():
            tau_dir = os.path.join(output_dir, f'tau_{tau:.3f}')
            os.makedirs(tau_dir, exist_ok=True)

            # Save weights
            if 'optimal_weights' in result and not result['optimal_weights'].empty:
                result['optimal_weights'].to_csv(os.path.join(tau_dir, 'optimal_weights.csv'))

            # Save BL returns
            if 'bl_returns' in result and not result['bl_returns'].empty:
                result['bl_returns'].to_csv(os.path.join(tau_dir, 'bl_returns.csv'))

            # Save performance metrics
            if 'portfolio_performance' in result and result['portfolio_performance']:
                perf_dict = {k: v for k, v in result['portfolio_performance'].items()
                           if not isinstance(v, pd.Series)}
                pd.Series(perf_dict).to_csv(os.path.join(tau_dir, 'performance_metrics.csv'))

        print(f"✅ Saved detailed results to {output_dir}")

    def get_optimal_tau(self, metric='sharpe_ratio'):
        """
        Get the optimal tau value based on specified metric

        Args:
            metric (str): Metric to optimize ('sharpe_ratio', 'annualized_return', 'volatility')

        Returns:
            float: Optimal tau value
        """
        if not self.results:
            print("No results available. Run sensitivity analysis first.")
            return None

        best_tau = None
        best_value = None

        for tau, result in self.results.items():
            perf = result['portfolio_performance']
            if perf and metric in perf:
                value = perf[metric]

                if best_tau is None:
                    best_tau = tau
                    best_value = value
                elif metric == 'volatility':  # Lower is better for volatility
                    if value < best_value:
                        best_tau = tau
                        best_value = value
                else:  # Higher is better for returns and sharpe ratio
                    if value > best_value:
                        best_tau = tau
                        best_value = value

        return best_tau

    def compare_tau_values(self, tau_list):
        """
        Compare specific tau values

        Args:
            tau_list (list): List of tau values to compare

        Returns:
            pd.DataFrame: Comparison results
        """
        comparison_data = []

        for tau in tau_list:
            if tau in self.results:
                perf = self.results[tau]['portfolio_performance']
                if perf:
                    comparison_data.append({
                        'Tau': tau,
                        'Annualized_Return': f"{perf['annualized_return']:.2%}",
                        'Volatility': f"{perf['volatility']:.2%}",
                        'Sharpe_Ratio': f"{perf['sharpe_ratio']:.3f}",
                        'Max_Drawdown': f"{perf['max_drawdown']:.2%}",
                        'Herfindahl_Index': f"{perf['herfindahl_index']:.3f}"
                    })

        return pd.DataFrame(comparison_data)


def run_comprehensive_tau_sensitivity():
    """
    Run tau sensitivity analysis on NIFTY 50 stocks with a single predefined range
    """
    nifty50_stocks = [
        'RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'HINDUNILVR.NS', 'INFY.NS',
        'ICICIBANK.NS', 'HDFC.NS', 'KOTAKBANK.NS', 'BHARTIARTL.NS', 'ITC.NS',
        'SBIN.NS', 'BAJFINANCE.NS', 'LICI.NS', 'ASIANPAINT.NS', 'MARUTI.NS',
        'HCLTECH.NS', 'AXISBANK.NS', 'LT.NS', 'TITAN.NS', 'SUNPHARMA.NS',
        'ULTRACEMCO.NS', 'WIPRO.NS', 'ONGC.NS', 'NTPC.NS', 'JSWSTEEL.NS',
        'POWERGRID.NS', 'M&M.NS', 'TATAMOTORS.NS', 'TECHM.NS', 'NESTLEIND.NS',
        'ADANIENTS.NS', 'HDFCLIFE.NS', 'COALINDIA.NS', 'SBILIFE.NS', 'INDUSINDBK.NS',
        'GRASIM.NS', 'BAJAJFINSV.NS', 'CIPLA.NS', 'TATACONSUM.NS', 'DRREDDY.NS',
        'EICHERMOT.NS', 'APOLLOHOSP.NS', 'UPL.NS', 'DIVISLAB.NS', 'BRITANNIA.NS',
        'BPCL.NS', 'BAJAJ-AUTO.NS', 'TATASTEEL.NS', 'HEROMOTOCO.NS', 'ADANIPORTS.NS'
    ]

    tau_range = np.arange(0.005, 0.051, 0.005)  # Fine-grained

    print(f"\n{'='*60}")
    print("Running tau sensitivity analysis")
    print(f"Tau range: {tau_range[0]:.3f} to {tau_range[-1]:.3f}")
    print(f"{'='*60}")

    analyzer = TauSensitivityAnalysis(nifty50_stocks, tau_range)
    results = analyzer.run_sensitivity_analysis(
        sequence_length=30,
        epochs=25,
        batch_size=32,
        prediction_horizon=5,
        output_dir="tau_sensitivity",
        use_frozen_data=True,
        frozen_data_path="data/nifty50_frozen_data.pkl"
    )

    if results:
        best_tau = analyzer.get_optimal_tau('sharpe_ratio')
        print(f"\n✅ Best Tau (Sharpe-optimal): {best_tau:.3f}")
        comparison_df = analyzer.compare_tau_values([best_tau])
        print(comparison_df.to_string(index=False))

    return results

if __name__ == "__main__":
    # Run comprehensive analysis
    comprehensive_results = run_comprehensive_tau_sensitivity()
    print("\nTau sensitivity analysis complete!")
    print("Check the output directories for detailed results and visualizations.")
