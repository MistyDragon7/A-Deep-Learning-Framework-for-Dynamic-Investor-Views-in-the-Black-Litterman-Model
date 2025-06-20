import pandas as pd
import os
from tensorflow.keras.models import load_model
import joblib
import numpy as np
import tensorflow as tf
from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy('mixed_float16')
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Conv1D, MaxPooling1D,
                                   Bidirectional, LSTM, Dense, Dropout,
                                   BatchNormalization, Concatenate, SpatialDropout1D,
                                   GaussianNoise)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

from tensorflow.keras import mixed_precision

SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
tf.keras.utils.set_random_seed(SEED)
tf.config.experimental.enable_op_determinism()  # Ensure full determinism
mixed_precision.set_global_policy('mixed_float16')
@tf.function(reduce_retracing=True)
def predict_mc_tf(model, x_repeated):
    return tf.squeeze(model(x_repeated, training=True), axis=-1)
def mc_dropout(x, rate):
    return Dropout(rate)(x)  # Keeps dropout active during inference

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
        df = stock_data[ticker].copy()

        features_list = []
        for col in self.feature_columns:
            if col in df.columns:
                feature_data = df[col].values.astype(np.float64)
                feature_data[np.isinf(feature_data)] = np.nan
                features_list.append(feature_data)
            else:
                features_list.append(np.zeros(len(df), dtype=np.float64))

        features = np.array(features_list).T
        features = pd.DataFrame(features).fillna(method='ffill').fillna(method='bfill').fillna(0).values

        X, y = [], []
        for i in range(self.sequence_length, len(features) - 5):
            X.append(features[i - self.sequence_length:i])
            if 'Returns' in df.columns and i + 5 < len(df):
                cum_return = df['Returns'].iloc[i+1:i+6].sum()
                y.append(float(cum_return) if not pd.isna(cum_return) else 0.0)
            else:
                y.append(0.0)

        return np.array(X), np.array(y)

    def build_stock_model(self, stock_ticker):
        print(f"Building model for {stock_ticker}")

        inputs = Input(shape=(self.sequence_length, self.n_features))
        x = GaussianNoise(0.01)(inputs)

        x = Conv1D(filters=64, kernel_size=3, activation='relu', padding='same')(x)
        x = SpatialDropout1D(0.1)(x, training=True)
        x = BatchNormalization()(x)
        x = MaxPooling1D(pool_size=2)(x)

        x = Conv1D(filters=32, kernel_size=3, activation='relu', padding='same')(x)
        x = SpatialDropout1D(0.1)(x, training=True)
        x = BatchNormalization()(x)

        x = Bidirectional(LSTM(50, return_sequences=True))(x)
        x = Dropout(0.2)(x, training=True)
        x = Bidirectional(LSTM(25, return_sequences=False))(x)

        x = mc_dropout(Dense(50, activation='relu')(x), rate=0.2)
        x = mc_dropout(Dense(25, activation='relu')(x), rate=0.1)

        output = Dense(1, activation='linear')(x)

        model = Model(inputs=inputs, outputs=output)
        from tensorflow.keras.losses import MeanSquaredError
        from tensorflow.keras.metrics import MeanAbsoluteError

        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss=MeanSquaredError(),
            metrics=[MeanAbsoluteError()]
        )
        

        return model

    def train_all_models(self, stock_data, epochs=50, batch_size=32, model_dir="saved_models"):
        print(f"Training models for {len(stock_data)} stocks...")
        os.makedirs(model_dir, exist_ok=True)

        for ticker in stock_data.keys():
            print(f"\nProcessing model for {ticker}")

            model_path = os.path.join(model_dir, f"{ticker}.h5")
            scaler_path = os.path.join(model_dir, f"{ticker}_scaler.pkl")

            # Try loading model
            if os.path.exists(model_path) and os.path.exists(scaler_path):
                try:
                    self.models[ticker] = load_model(model_path)
                    self.scalers[ticker] = joblib.load(scaler_path)
                    print(f"✓ Loaded saved model and scaler for {ticker}")
                    continue
                except Exception as e:
                    print(f"⚠️ Failed to load saved model for {ticker}: {e}. Retraining...")

            # Train from scratch
            X, y = self.prepare_data_for_stock(stock_data, ticker)

            if len(X) < 100:
                print(f"⚠️ Insufficient data for {ticker} — Skipping")
                continue

            # Scale features
            scaler = MinMaxScaler()
            X_reshaped = X.reshape(-1, self.n_features)
            X_scaled_reshaped = scaler.fit_transform(X_reshaped)
            X_scaled = X_scaled_reshaped.reshape(X.shape)
            self.scalers[ticker] = scaler

            split_idx = int(len(X_scaled) * 0.8)
            X_train, X_val = X_scaled[:split_idx], X_scaled[split_idx:]
            y_train, y_val = y[:split_idx], y[split_idx:]

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

            # Save model and scaler
            model.save(model_path)
            joblib.dump(scaler, scaler_path)

            if history.history:
                val_loss = min(history.history.get('val_loss', [float('inf')]))
                print(f"✓ Trained and saved model for {ticker} — Best val_loss: {val_loss:.6f}")
            else:
                print(f"✓ Trained and saved model for {ticker}")


    def generate_investor_views(self, stock_data, prediction_horizon=5):
        print(f"\nGenerating investor views for {prediction_horizon} days ahead...")

        def predict_mc(model, x_input, n_samples=10):
            x_repeated = tf.repeat(x_input, repeats=n_samples, axis=0)
            preds = model(x_repeated, training=True)  # Dropout off
            return tf.squeeze(preds, axis=-1).numpy()
        views = {}
        view_uncertainties = {}

        for ticker in self.models.keys():
            X_latest, _ = self.prepare_data_for_stock(stock_data, ticker)

            if len(X_latest) < 1:
                print(f"Insufficient data to generate views for {ticker}")
                continue

            if ticker not in self.scalers:
                print(f"Scaler not found for {ticker}, skipping view generation")
                continue

            latest_sequence = X_latest[-1:]
            latest_sequence_reshaped = latest_sequence.reshape(-1, self.n_features)
            X_latest_scaled_reshaped = self.scalers[ticker].transform(latest_sequence_reshaped)
            X_latest_scaled = X_latest_scaled_reshaped.reshape(1, self.sequence_length, self.n_features)

            mc_preds = predict_mc(self.models[ticker], X_latest_scaled, n_samples=50)
            expected_return = np.mean(mc_preds)
            view_uncertainty = max(np.std(mc_preds), 0.001)
            views[ticker] = expected_return
            view_uncertainties[ticker] = view_uncertainty

            print(f"{ticker}: Expected return = {expected_return:.4f}, Uncertainty = {view_uncertainty:.4f}")

        return views, view_uncertainties
