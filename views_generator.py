import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Conv1D, MaxPooling1D,
                                   Bidirectional, LSTM, Dense, Dropout,
                                   BatchNormalization, Concatenate)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
def mc_dropout(x, rate):
    return Dropout(rate)(x, training=True)  # Keeps dropout active during inference
class CNNBiLSTMViewsGenerator:
    """CNN-BiLSTM model to generate investor views for Black-Litterman"""

    def __init__(self, n_stocks, sequence_length=30):
        self.n_stocks = n_stocks
        self.sequence_length = sequence_length
        self.feature_columns = [
            'Open', 'High', 'Low', 'Close', 'Volume', 'Returns', 'Volatility',
            'MA_10', 'MA_20', 'EMA_12', 'RSI', 'MACD', 'MACD_Signal',
            'BB_Upper', 'BB_Lower', 'BB_Position', 'Price_Change', 'Volume_Change',
            'Price_Momentum'
        ]
        self.n_features = len(self.feature_columns)
        self.models = {}  # Individual models for each stock
        self.scalers = {}

    def prepare_data_for_stock(self, stock_data, ticker):
        """Prepare training data for individual stock"""
        df = stock_data[ticker].copy()

        features_list = []
        for col in self.feature_columns:
            if col in df.columns:
                feature_data = df[col].values.astype(np.float64)
                feature_data[np.isinf(feature_data)] = np.nan
                features_list.append(feature_data)
            else:
                features_list.append(np.zeros(len(df), dtype=np.float64))

        features = np.array(features_list).T  # Shape: (n_samples, n_features)

        features = pd.DataFrame(features).fillna(method='ffill').fillna(method='bfill').fillna(0).values

        X, y = [], []
        for i in range(self.sequence_length, len(features) - 1):
            X.append(features[i-self.sequence_length:i])
            if 'Returns' in df.columns and i + 1 < len(df): # Ensure i+1 is a valid index
                 next_return = df['Returns'].iloc[i+1]
                 y.append(float(next_return) if not pd.isna(next_return) else 0.0)
            else:
                 y.append(0.0)


        return np.array(X), np.array(y)

    def build_stock_model(self, stock_ticker):
        """Build CNN-BiLSTM model for individual stock"""
        print(f"Building model for {stock_ticker}")

        inputs = Input(shape=(self.sequence_length, self.n_features))

        # CNN layers for feature extraction
        conv1 = Conv1D(filters=64, kernel_size=3, activation='relu', padding='same')(inputs)
        conv1 = BatchNormalization()(conv1)
        pool1 = MaxPooling1D(pool_size=2)(conv1)

        conv2 = Conv1D(filters=32, kernel_size=3, activation='relu', padding='same')(pool1)
        conv2 = BatchNormalization()(conv2)

        # BiLSTM layers for temporal dependencies
        bilstm1 = Bidirectional(LSTM(50, return_sequences=True))(conv2)
        bilstm2 = Bidirectional(LSTM(25, return_sequences=False))(bilstm1)

        # Dense layers for prediction
        dense1 = mc_dropout(Dense(50, activation='relu')(bilstm2), rate=0.2)
        dense2 = mc_dropout(Dense(25, activation='relu')(dense1), rate=0.1)

        # Output: predicted return for next period
        output = Dense(1, activation='linear')(dense2)

        model = Model(inputs=inputs, outputs=output)
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])

        return model

    def train_all_models(self, stock_data, epochs=50, batch_size=32):
        """Train CNN-BiLSTM models for all stocks"""
        print(f"Training models for {len(stock_data)} stocks...")

        for ticker in stock_data.keys():
            print(f"\nTraining model for {ticker}")

            X, y = self.prepare_data_for_stock(stock_data, ticker)

            if len(X) < 100:  # Skip if insufficient data after preparing sequences
                print(f"Insufficient data for {ticker}")
                continue

            # Scale features
            # Apply scaler to the reshaped data for MinMaxScaler, then reshape back
            scaler = MinMaxScaler()
            # Reshape X from (n_sequences, sequence_length, n_features) to (n_sequences * sequence_length, n_features)
            X_reshaped = X.reshape(-1, self.n_features)
            X_scaled_reshaped = scaler.fit_transform(X_reshaped)
            # Reshape back to the original sequence shape
            X_scaled = X_scaled_reshaped.reshape(X.shape)
            self.scalers[ticker] = scaler

            # Split data
            split_idx = int(len(X_scaled) * 0.8)
            X_train, X_val = X_scaled[:split_idx], X_scaled[split_idx:]
            y_train, y_val = y[:split_idx], y[split_idx:]

            # Build and train model
            model = self.build_stock_model(ticker)

            callbacks = [
                EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
                ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)
            ]

            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=epochs,
                batch_size=batch_size,
                callbacks=callbacks,
                verbose=0
            )

            self.models[ticker] = model

            # Print training results
            # Ensure there is history data before trying to find min val_loss
            if history.history:
                val_loss = min(history.history.get('val_loss', [float('inf')]))
                if val_loss == float('inf'):
                    print(f"✓ {ticker} trained - No validation loss recorded")
                else:
                    print(f"✓ {ticker} trained - Best validation loss: {val_loss:.6f}")
            else:
                 print(f"✓ {ticker} trained - No training history recorded")


    def generate_investor_views(self, stock_data, prediction_horizon=5):
        """Generate investor views using trained models"""
        print(f"\nGenerating investor views for {prediction_horizon} days ahead...")
        def predict_mc(model, x_input, n_samples=20):
            preds = np.array([model(x_input, training=True).numpy().squeeze() for _ in range(n_samples)])
            return preds
        views = {}
        view_uncertainties = {}

        for ticker in self.models.keys():
            # Get latest data for prediction
            # Make sure to get the latest sequence correctly
            X_latest, _ = self.prepare_data_for_stock(stock_data, ticker)

            if len(X_latest) < 1: # Need at least one sequence
                print(f"Insufficient data to generate views for {ticker}")
                continue

            # Scale the latest sequence
            # Ensure the scaler is available for this ticker
            if ticker not in self.scalers:
                print(f"Scaler not found for {ticker}, skipping view generation")
                continue

            # Get the very last sequence from X_latest
            latest_sequence = X_latest[-1:] # Shape (1, sequence_length, n_features)

            # Reshape for scaler (2D)
            latest_sequence_reshaped = latest_sequence.reshape(-1, self.n_features)

            # Apply scaler
            X_latest_scaled_reshaped = self.scalers[ticker].transform(latest_sequence_reshaped)

            # Reshape back to the sequence shape
            X_latest_scaled = X_latest_scaled_reshaped.reshape(1, self.sequence_length, self.n_features)


            # Predict multiple horizons (simplified approach)
            # A more robust approach would be a recursive prediction
            mc_preds = predict_mc(self.models[ticker], X_latest_scaled, n_samples=20)
            expected_return = np.mean(mc_preds)
            view_uncertainty = max(np.std(mc_preds), 0.001)
            views[ticker] = expected_return
            view_uncertainties[ticker] = max(view_uncertainty, 0.001)  # Minimum uncertainty

            print(f"{ticker}: Expected return = {expected_return:.4f}, Uncertainty = {view_uncertainties[ticker]:.4f}")

        return views, view_uncertainties
