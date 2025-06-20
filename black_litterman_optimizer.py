import pandas as pd
import numpy as np

class BlackLittermanOptimizer:
    def __init__(self, returns_matrix, market_caps, risk_free_rate=0.06):
        self.returns_matrix = returns_matrix
        self.market_caps = market_caps
        self.risk_free_rate = risk_free_rate
        self.assets = list(returns_matrix.columns)

        if not self.returns_matrix.empty:
            self.mean_returns = self.returns_matrix.mean() * 252 - self.risk_free_rate
            self.cov_matrix = self.returns_matrix.cov() * 252
        else:
            print("Warning: Returns matrix is empty.")
            self.mean_returns = pd.Series(0, index=self.assets)
            self.cov_matrix = pd.DataFrame(0, index=self.assets, columns=self.assets)

        self.market_weights = self.calculate_market_weights()
        self.dynamic_risk_aversion = self.compute_dynamic_risk_aversion()

    def calculate_market_weights(self):
        total_market_cap = sum(self.market_caps.values())
    
        if total_market_cap <= 0 or not self.market_caps:
            print("⚠️ Warning: Market capitalizations are missing or zero. Using equal weights.")
            return pd.Series(1.0 / len(self.assets), index=self.assets) if self.assets else pd.Series()
    
        weights = {
            asset: self.market_caps.get(asset, 0.0) / total_market_cap
            for asset in self.assets
        }
        return pd.Series(weights, index=self.assets)


    def compute_dynamic_risk_aversion(self):
        if self.mean_returns.empty or self.cov_matrix.empty or self.market_weights.empty:
            return 3.0
        try:
            expected_excess_return = self.mean_returns.mean()
            market_variance = float(self.market_weights.T @ self.cov_matrix @ self.market_weights)
            implied_lambda = expected_excess_return / market_variance
            return implied_lambda if 0 < implied_lambda < 20 else 3.0
        except:
            return 3.0

    def calculate_implied_returns(self, risk_aversion=None):
        if risk_aversion is None:
            risk_aversion = self.dynamic_risk_aversion
        implied_excess_returns = risk_aversion * np.dot(self.cov_matrix, self.market_weights)
        return pd.Series(implied_excess_returns, index=self.assets)

    def black_litterman_optimization(self, views, view_uncertainties, risk_aversion=None, tau=0.025):
        if risk_aversion is None:
            risk_aversion = self.dynamic_risk_aversion

        implied_returns = self.calculate_implied_returns(risk_aversion)

        view_assets = [asset for asset in views.keys() if asset in self.assets]
        n_views = len(view_assets)
        n_assets = len(self.assets)

        if n_views == 0:
            P = np.eye(n_assets)
            Q = np.zeros(n_assets)
            Omega = np.eye(n_assets) * 1e6
        else:
            P = np.zeros((n_views, n_assets))
            Q = np.zeros(n_views)
            omega_diag = np.zeros(n_views)
            for i, asset in enumerate(view_assets):
                idx = self.assets.index(asset)
                P[i, idx] = 1.0
                Q[i] = views[asset] - (self.risk_free_rate * 5 / 252)  # Convert to excess 5-day return
                unc = view_uncertainties.get(asset, 0.001)
                omega_diag[i] = (unc) ** 2
            Omega = np.diag(omega_diag)

        tau_cov = tau * self.cov_matrix
        try:
            M1 = np.linalg.inv(tau_cov) + P.T @ np.linalg.inv(Omega) @ P
            M2 = np.linalg.inv(tau_cov) @ implied_returns + P.T @ np.linalg.inv(Omega) @ Q
            bl_returns = np.linalg.solve(M1, M2)
        except:
            bl_returns = implied_returns

        try:
            cov_bl = np.linalg.inv(M1)
        except:
            cov_bl = self.cov_matrix

        weights = self.optimize_portfolio(bl_returns, cov_bl, risk_aversion)
        return weights, pd.Series(bl_returns, index=self.assets), pd.DataFrame(cov_bl, index=self.assets, columns=self.assets)

    def optimize_portfolio(self, expected_returns, cov_matrix, risk_aversion):
        try:
            inv_cov = np.linalg.inv(cov_matrix)
            raw_weights = inv_cov @ expected_returns
            weights = raw_weights / risk_aversion
            weights = np.maximum(weights, 0)
            total = weights.sum()
            return pd.Series(weights / total if total > 0 else weights, index=self.assets)
        except:
            return pd.Series(np.ones(len(self.assets)) / len(self.assets), index=self.assets)
