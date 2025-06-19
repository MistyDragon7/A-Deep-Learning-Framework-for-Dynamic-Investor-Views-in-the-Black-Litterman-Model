import numpy as np
import pandas as pd
class BlackLittermanOptimizer:
    """Black-Litterman portfolio optimization with CNN-BiLSTM views"""

    def __init__(self, returns_matrix, market_caps, risk_free_rate=0.06):
        self.returns_matrix = returns_matrix
        self.market_caps = market_caps
        self.risk_free_rate = risk_free_rate
        self.assets = list(returns_matrix.columns)

        # Calculate market parameters
        # Ensure mean_returns and cov_matrix are calculated on the full dataset before slicing
        if not self.returns_matrix.empty:
            self.mean_returns = self.returns_matrix.mean() * 252  # Annualized
            self.cov_matrix = self.returns_matrix.cov() * 252  # Annualized
        else:
            # Handle case where returns_matrix is empty
            print("Warning: Returns matrix is empty. Cannot calculate market parameters.")
            self.mean_returns = pd.Series(0, index=self.assets)
            self.cov_matrix = pd.DataFrame(0, index=self.assets, columns=self.assets)

        self.market_weights = self.calculate_market_weights()

        # Calculate dynamic risk aversion
        self.dynamic_risk_aversion = self.compute_dynamic_risk_aversion()

    def calculate_market_weights(self):
        """Calculate market capitalization weights"""
        total_market_cap = sum(self.market_caps.values())
        weights = {}
        for asset in self.assets:
            if asset in self.market_caps and total_market_cap > 0:
                weights[asset] = self.market_caps[asset] / total_market_cap
            else:
                weights[asset] = 1.0 / len(self.assets) if len(self.assets) > 0 else 0  # Equal weight or 0 if no assets

        # Normalize to ensure sum = 1, handle case of zero assets
        total_weight = sum(weights.values())
        if total_weight > 0:
            for asset in weights:
                weights[asset] /= total_weight
        elif len(self.assets) > 0:
            # If total_weight is 0 but there are assets, assign equal weights
            weights = {asset: 1.0 / len(self.assets) for asset in self.assets}
        else:
             # If no assets, weights dict remains empty
             pass

        return pd.Series(weights, index=self.assets)

    def compute_dynamic_risk_aversion(self):
        """
        Computes implied risk aversion coefficient using:
            lambda = (E[r] - r_f) / (w_mkt^T Σ w_mkt)
        """
        if self.mean_returns.empty or self.cov_matrix.empty or self.market_weights.empty:
            print("Warning: Cannot compute dynamic risk aversion with empty data. Using default value of 3.0")
            return 3.0

        try:
            expected_return = self.mean_returns.mean()  # Already annualized
            excess_return = expected_return - self.risk_free_rate
            market_variance = float(self.market_weights.T @ self.cov_matrix @ self.market_weights)

            if market_variance <= 0:
                print("Warning: Market variance is non-positive. Using default risk aversion of 3.0")
                return 3.0

            implied_lambda = excess_return / market_variance

            # Ensure reasonable bounds for risk aversion
            if implied_lambda <= 0 or implied_lambda > 20:
                print(f"Warning: Computed risk aversion {implied_lambda:.2f} is outside reasonable bounds. Using default of 3.0")
                return 3.0

            print(f"Dynamic risk aversion computed: {implied_lambda:.3f}")
            return implied_lambda

        except Exception as e:
            print(f"Error computing dynamic risk aversion: {e}. Using default value of 3.0")
            return 3.0

    def calculate_implied_returns(self, risk_aversion=None):
        """Calculate implied equilibrium returns"""
        if risk_aversion is None:
            risk_aversion = self.dynamic_risk_aversion

        # π = λ * Σ * w_market
        # Ensure matrix multiplication is possible
        if self.cov_matrix.shape[0] == self.market_weights.shape[0] and not self.cov_matrix.empty:
            implied_returns = risk_aversion * np.dot(self.cov_matrix, self.market_weights)
            return pd.Series(implied_returns, index=self.assets)
        else:
            print("Warning: Cannot calculate implied returns. Covariance matrix or market weights are invalid.")
            return pd.Series(0, index=self.assets)

    def black_litterman_optimization(self, views, view_uncertainties, risk_aversion=None, tau=0.025):
        """
        Perform Black-Litterman optimization

        Parameters:
        - views: Dictionary of expected returns from CNN-BiLSTM
        - view_uncertainties: Dictionary of view uncertainties
        - risk_aversion: Risk aversion parameter (if None, uses dynamic calculation)
        - tau: Uncertainty of prior (typically 0.01 to 0.05)
        """
        print("\nPerforming Black-Litterman optimization...")

        # Use dynamic risk aversion if not provided
        if risk_aversion is None:
            risk_aversion = self.dynamic_risk_aversion
            print(f"Using dynamic risk aversion: {risk_aversion:.3f}")

        # Step 1: Calculate implied returns
        implied_returns = self.calculate_implied_returns(risk_aversion)

        # Check if implied_returns are valid
        if implied_returns.isnull().any() or np.isinf(implied_returns).any():
             print("Error: Implied returns calculation resulted in NaNs or Infs. Using implied_returns as zeros.")
             implied_returns = pd.Series(0, index=self.assets)

        # Step 2: Set up views
        # P matrix: picking matrix (which assets the views relate to)
        # Q vector: the views (expected returns)
        # Ω matrix: uncertainty of views

        view_assets = [asset for asset in views.keys() if asset in self.assets]
        n_views = len(view_assets)
        n_assets = len(self.assets)

        if n_views == 0:
            print("No valid views found, using market weights for optimization base")
            # Still perform optimization but with views Q=0 and large uncertainty Omega=I
            Q = np.zeros(n_assets) # Views are zero for all assets
            P = np.eye(n_assets)   # P matrix is identity
            Omega = np.eye(n_assets) * 1e6 # Large uncertainty
            n_views = n_assets # Adjust n_views for matrix shapes

        else:
            # Create picking matrix P (identity for absolute views)
            P = np.zeros((n_views, n_assets))
            Q = np.zeros(n_views)
            omega_diag = np.zeros(n_views)

            for i, asset in enumerate(view_assets):
                asset_idx = self.assets.index(asset)
                P[i, asset_idx] = 1.0  # Absolute view on this asset
                Q[i] = views[asset] * 252  # Annualized view
                # Ensure view uncertainty is not NaN, Inf, or zero before squaring
                unc = view_uncertainties.get(asset, 0.001)
                if pd.isna(unc) or np.isinf(unc) or unc <= 0:
                    unc = 0.001 # Default minimum
                omega_diag[i] = (unc * 252) ** 2  # Annualized variance

            # Ensure no zero variances in Omega
            omega_diag[omega_diag <= 0] = 1e-6 # Replace zero or negative variance with a small positive number
            Omega = np.diag(omega_diag)

        # Step 3: Black-Litterman formula
        # μ_BL = [(τΣ)^(-1) + P'Ω^(-1)P]^(-1) * [(τΣ)^(-1)π + P'Ω^(-1)Q]

        # Ensure cov_matrix is not singular or empty
        if self.cov_matrix.empty or np.linalg.det(self.cov_matrix) == 0:
            print("Warning: Covariance matrix is empty or singular. Cannot perform Black-Litterman.")
            # Fallback to equal weights or market weights
            if not self.market_weights.empty:
                 return self.market_weights, implied_returns, self.cov_matrix
            else:
                 equal_weights = pd.Series(1.0/len(self.assets), index=self.assets) if len(self.assets) > 0 else pd.Series()
                 return equal_weights, implied_returns, self.cov_matrix

        try:
            tau_cov = tau * self.cov_matrix
            tau_cov_inv = np.linalg.inv(tau_cov)
            omega_inv = np.linalg.inv(Omega)

            # Calculate the new expected returns
            M1 = tau_cov_inv + np.dot(P.T, np.dot(omega_inv, P))
            M2 = np.dot(tau_cov_inv, implied_returns) + np.dot(P.T, np.dot(omega_inv, Q))

            # Ensure M1 is invertible
            if np.linalg.det(M1) == 0:
                 print("Warning: Matrix M1 is singular. Cannot calculate BL returns. Falling back to implied returns.")
                 bl_returns = implied_returns
                 bl_cov = self.cov_matrix # Keep original cov matrix
                 optimal_weights = self.optimize_portfolio(bl_returns, bl_cov, risk_aversion) # Optimize with implied returns
                 return optimal_weights, bl_returns, bl_cov

            bl_returns = np.dot(np.linalg.inv(M1), M2)
            bl_returns = pd.Series(bl_returns, index=self.assets)

            # Calculate new covariance matrix
            bl_cov = np.linalg.inv(M1)
            bl_cov = pd.DataFrame(bl_cov, index=self.assets, columns=self.assets)

        except np.linalg.LinAlgError as e:
            print(f"Linear algebra error during Black-Litterman calculation: {e}")
            print("Falling back to implied returns and original covariance matrix.")
            bl_returns = implied_returns
            bl_cov = self.cov_matrix # Keep original cov matrix
            # Optimize with implied returns
            optimal_weights = self.optimize_portfolio(bl_returns, bl_cov, risk_aversion)
            return optimal_weights, bl_returns, bl_cov

        # Step 4: Optimize portfolio weights
        optimal_weights = self.optimize_portfolio(bl_returns, bl_cov, risk_aversion)

        print("Black-Litterman optimization completed")
        return optimal_weights, bl_returns, bl_cov

    def optimize_portfolio(self, expected_returns, cov_matrix, risk_aversion):
        """Optimize portfolio using mean-variance optimization"""
        # w* = (1/λ) * Σ^(-1) * μ
        if cov_matrix.empty or np.linalg.det(cov_matrix) == 0:
            print("Warning: Covariance matrix is singular or empty during optimization, using equal weights")
            equal_weights = np.ones(len(self.assets)) / len(self.assets) if len(self.assets) > 0 else []
            return pd.Series(equal_weights, index=self.assets)

        try:
            # Ensure expected_returns is aligned with cov_matrix columns
            if len(expected_returns) != cov_matrix.shape[0]:
                 print("Error: Mismatch between expected returns and covariance matrix dimensions.")
                 # Fallback to equal weights
                 equal_weights = np.ones(len(self.assets)) / len(self.assets) if len(self.assets) > 0 else []
                 return pd.Series(equal_weights, index=self.assets)

            cov_inv = np.linalg.inv(cov_matrix)
            # Ensure expected_returns are numeric and finite
            expected_returns_cleaned = expected_returns.replace([np.inf, -np.inf], np.nan).fillna(0)

            # Ensure result of dot product is finite
            intermediate_result = np.dot(cov_inv, expected_returns_cleaned)
            if np.isinf(intermediate_result).any() or np.isnan(intermediate_result).any():
                 print("Warning: Intermediate optimization result contains NaNs or Infs, using equal weights")
                 equal_weights = np.ones(len(self.assets)) / len(self.assets) if len(self.assets) > 0 else []
                 return pd.Series(equal_weights, index=self.assets)

            optimal_weights = intermediate_result / risk_aversion

            # Normalize weights to sum to 1
            sum_weights = np.sum(optimal_weights)
            if sum_weights != 0:
                optimal_weights = optimal_weights / sum_weights
            else:
                 # If sum is 0, fall back to equal weights (unless there are no assets)
                 print("Warning: Sum of optimal weights is zero after initial calculation, using equal weights")
                 optimal_weights = np.ones(len(self.assets)) / len(self.assets) if len(self.assets) > 0 else []

            # Ensure no negative weights (long-only constraint)
            optimal_weights = np.maximum(optimal_weights, 0)

            # Re-normalize after applying long-only constraint
            sum_positive_weights = np.sum(optimal_weights)
            if sum_positive_weights > 0:
                optimal_weights = optimal_weights / sum_positive_weights
            elif len(self.assets) > 0:
                # If sum of positive weights is zero but there are assets, assign equal weights
                print("Warning: Sum of positive weights is zero, using equal weights")
                optimal_weights = np.ones(len(self.assets)) / len(self.assets)
            else:
                 # If no assets, weights list remains empty
                 optimal_weights = []

            return pd.Series(optimal_weights, index=self.assets)

        except np.linalg.LinAlgError:
            print("Warning: Singular covariance matrix during optimization, using equal weights")
            equal_weights = np.ones(len(self.assets)) / len(self.assets) if len(self.assets) > 0 else []
            return pd.Series(equal_weights, index=self.assets)
        except Exception as e:
             print(f"An error occurred during portfolio optimization: {e}")
             print("Falling back to equal weights")
             equal_weights = np.ones(len(self.assets)) / len(self.assets) if len(self.assets) > 0 else []
             return pd.Series(equal_weights, index=self.assets)
