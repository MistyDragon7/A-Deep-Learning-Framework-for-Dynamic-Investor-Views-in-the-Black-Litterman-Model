import yfinance as yf
import pandas as pd
import numpy as np
from technical_indicators import TechnicalIndicators
class StockDataFetcher:
    """Fetch and preprocess stock data"""

    def __init__(self, stock_list=None, period="2y", interval="1d"):
        self.stock_list = stock_list
        self.period = period
        self.interval = interval
        self.stock_data = {}
        self.returns_matrix = None
        self.market_caps = {}

    def fetch_all_stocks(self):
        """Fetch data for all stocks"""
        print(f"Fetching data for {len(self.stock_list)} stocks...")

        for i, ticker in enumerate(self.stock_list):
            try:
                print(f"Fetching {ticker} ({i+1}/{len(self.stock_list)})")
                stock = yf.Ticker(ticker)
                df = stock.history(period=self.period, interval=self.interval)

                if not df.empty and len(df) > 100:
                    df = df.dropna()
                    self.stock_data[ticker] = df

                    # Get market cap info
                    info = stock.info
                    self.market_caps[ticker] = info.get('marketCap', 1e12)  # Default if not available

                    print(f"✓ {ticker}: {df.shape[0]} records")
                else:
                    print(f"✗ {ticker}: Insufficient data")

            except Exception as e:
                print(f"✗ Error fetching {ticker}: {str(e)}")

        print(f"Successfully fetched {len(self.stock_data)} stocks")
        return self.stock_data

    def add_technical_indicators(self):
        """Add technical indicators to all stocks"""
        print("Adding technical indicators...")
    
        for ticker in self.stock_data.keys():
            df = self.stock_data[ticker].copy()
    
            # ⚠️ NEW: Skip if already computed
            if 'RSI' in df.columns:
                continue
            
            # Basic indicators
            df['Returns'] = TechnicalIndicators.calculate_returns(df['Close'])
            df['Volatility'] = TechnicalIndicators.calculate_volatility(df['Returns'])
            df['MA_10'] = TechnicalIndicators.moving_average(df['Close'], 10)
            df['MA_20'] = TechnicalIndicators.moving_average(df['Close'], 20)
            df['EMA_12'] = TechnicalIndicators.exponential_moving_average(df['Close'], 12)
            df['RSI'] = TechnicalIndicators.rsi(df['Close'])
    
            # MACD
            macd, signal = TechnicalIndicators.macd(df['Close'])
            df['MACD'] = macd
            df['MACD_Signal'] = signal
    
            # Bollinger Bands
            bb_upper, bb_middle, bb_lower = TechnicalIndicators.bollinger_bands(df['Close'])
            df['BB_Upper'] = bb_upper
            df['BB_Lower'] = bb_lower
            df['BB_Position'] = (df['Close'] - bb_lower) / (bb_upper - bb_lower)
    
            # Price momentum and volume
            df['Price_Change'] = df['Close'].pct_change()
            df['Volume_Change'] = df['Volume'].pct_change()
            df['Price_Momentum'] = df['Close'].pct_change(5)
    
            self.stock_data[ticker] = df

    def create_returns_matrix(self):
        """Create returns matrix for all stocks"""
        returns_data = {}
        common_dates = None

        # Find common dates
        for ticker, df in self.stock_data.items():
            if common_dates is None:
                common_dates = set(df.index)
            else:
                common_dates = common_dates.intersection(set(df.index))

        common_dates = sorted(list(common_dates))

        # Create returns matrix
        for ticker, df in self.stock_data.items():
            returns_data[ticker] = []
            for date in common_dates:
                if date in df.index and 'Returns' in df.columns:
                    ret = df.loc[date, 'Returns']
                    returns_data[ticker].append(ret if not pd.isna(ret) else 0)
                else:
                    returns_data[ticker].append(0)

        self.returns_matrix = pd.DataFrame(returns_data, index=common_dates)
        return self.returns_matrix
