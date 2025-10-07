import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

class EnhancedForexPredictor:
    def __init__(self):
        self.model = None
        self.feature_columns = []
        self.is_trained = False
    
    def calculate_confidence_tier(self, confidence):
        if confidence >= 0.7:
            return "high", 1, 50
        elif confidence >= 0.6:
            return "medium", 3, 30
        elif confidence >= 0.5:
            return "low", 5, 20
        else:
            return "very_low", 0, 0
    
    def create_features(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        
        df['price_range'] = (df['high'] - df['low']) / df['open']
        df['price_change'] = (df['close'] - df['open']) / df['open']
        df['body_size'] = abs(df['close'] - df['open']) / (df['high'] - df['low']).replace(0, 0.0001)
        
        for window in [3, 5, 8]:
            df[f'sma_{window}'] = df['close'].rolling(window=window, min_periods=1).mean()
            df[f'price_vs_sma{window}'] = df['close'] / df[f'sma_{window}'].replace(0, 1)
        
        df['volatility'] = df['close'].pct_change().rolling(window=5, min_periods=1).std().fillna(0)
        
        if 'volume' in df.columns:
            df['volume_sma'] = df['volume'].rolling(window=5, min_periods=1).mean()
            df['volume_ratio'] = df['volume'] / df['volume_sma'].replace(0, 1)
            df['volume_ratio'] = df['volume_ratio'].replace([np.inf, -np.inf], 1.0)
        
        df = df.fillna(0)
        return df
    
    def prepare_target(self, data: pd.DataFrame, prediction_horizon: int = 1) -> pd.Series:
        future_return = (data['close'].shift(-prediction_horizon) - data['close']) / data['close']
        target = (future_return > 0).astype(int)
        target = target.fillna(0)
        return target
    
    def train_model(self, data: pd.DataFrame, timeframe: str = "1h"):
        df_with_features = self.create_features(data)
        target = self.prepare_target(df_with_features, 1)
        
        self.feature_columns = [
            'open', 'high', 'low', 'close', 'price_range', 'price_change', 
            'body_size', 'sma_3', 'sma_5', 'sma_8', 'price_vs_sma3',
            'price_vs_sma5', 'price_vs_sma8', 'volatility'
        ]
        
        if 'volume' in df_with_features.columns:
            self.feature_columns.extend(['volume', 'volume_ratio'])
        
        available_features = [col for col in self.feature_columns if col in df_with_features.columns]
        X = df_with_features[available_features]
        y = target
        
        valid_indices = ~(X.isna().any(axis=1) | y.isna())
        X_clean = X[valid_indices]
        y_clean = y[valid_indices]
        
        if len(X_clean) < 10:
            raise ValueError("Insufficient data for training")
        
        self.model = RandomForestClassifier(
            n_estimators=30,
            max_depth=8,
            min_samples_split=3,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=1
        )
        
        self.model.fit(X_clean, y_clean)
        self.is_trained = True
        
        return self.model
    
    def predict_with_confidence(self, latest_data: pd.DataFrame):
        if not self.is_trained:
            raise ValueError("Model must be trained first")
        
        latest_with_features = self.create_features(latest_data)
        
        available_features = [col for col in self.feature_columns if col in latest_with_features.columns]
        X_new = latest_with_features[available_features].iloc[-1:]
        
        if X_new.isna().any().any():
            X_new = X_new.fillna(0)
        
        prediction = self.model.predict(X_new)[0]
        probabilities = self.model.predict_proba(X_new)[0]
        confidence = max(probabilities)
        
        return prediction, min(confidence, 0.95), probabilities