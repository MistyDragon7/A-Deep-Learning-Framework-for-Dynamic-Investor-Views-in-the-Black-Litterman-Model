import numpy as numpy
import pandas as pd
class TechnicalIndicators:
    """Calculate technical indicators for stocks"""

    @staticmethod
    def moving_average(data, window=10):
        return data.rolling(window=window).mean()

    @staticmethod
    def exponential_moving_average(data, span=10):
        return data.ewm(span=span).mean()

    @staticmethod
    def rsi(data, window=14):
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    @staticmethod
    def macd(data, fast=12, slow=26, signal=9):
        ema_fast = data.ewm(span=fast).mean()
        ema_slow = data.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        signal_line = macd.ewm(span=signal).mean()
        return macd, signal_line

    @staticmethod
    def bollinger_bands(data, window=20, num_std=2):
        rolling_mean = data.rolling(window=window).mean()
        rolling_std = data.rolling(window=window).std()
        upper_band = rolling_mean + (rolling_std * num_std)
        lower_band = rolling_mean - (rolling_std * num_std)
        return upper_band, rolling_mean, lower_band

    @staticmethod
    def calculate_returns(prices):
        return prices.pct_change()

    @staticmethod
    def calculate_volatility(returns, window=20):
        return returns.rolling(window=window).std()
