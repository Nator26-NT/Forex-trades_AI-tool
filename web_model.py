import requests
import pandas as pd
import time
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from config import API_KEY, BASE_URL, TIMEFRAME_INTERVALS, RATE_LIMIT_DELAY

class ForexDataFetcher:
    def __init__(self):
        self.api_key = API_KEY
        self.base_url = BASE_URL
    
    def fetch_forex_data(self, symbol: str, timeframe: str = "1day"):
        """Fetch forex data from Twelve Data API with specified timeframe"""
        if timeframe not in TIMEFRAME_INTERVALS:
            raise ValueError(f"Unsupported timeframe: {timeframe}")
        
        # Format symbol for API (e.g., EUR/USD)
        symbol_td = f"{symbol[:3]}/{symbol[3:]}"
        interval = TIMEFRAME_INTERVALS[timeframe]
        
        # Adjust outputsize based on timeframe for optimal performance
        outputsize = self._get_optimal_outputsize(timeframe)
        
        url = f"{self.base_url}?symbol={symbol_td}&interval={interval}&outputsize={outputsize}&apikey={self.api_key}"

        try:
            response = requests.get(url)
            data = response.json()
            
            if "values" not in data:
                if "message" in data:
                    raise ValueError(f"API Error: {data['message']}")
                raise ValueError(f"Error fetching data: {data}")

            df = pd.DataFrame(data["values"])
            df = self._process_dataframe(df)
            
            time.sleep(RATE_LIMIT_DELAY)  # Rate limiting
            return df
            
        except requests.exceptions.RequestException as e:
            raise ValueError(f"Network error: {str(e)}")
    
    def _get_optimal_outputsize(self, timeframe: str) -> int:
        """Get optimal data points based on timeframe for performance"""
        outputsize_map = {
            "1min": 1440,    # 24 hours
            "5min": 576,     # 2 days
            "15min": 384,    # 4 days
            "1h": 168,       # 1 week
            "4h": 126,       # 3 weeks
            "1day": 500,     # ~1.5 years
            "1week": 260,    # 5 years
            "1month": 120    # 10 years
        }
        return outputsize_map.get(timeframe, 100)
    
    def _process_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process and clean the dataframe"""
        df = df.rename(columns={
            "datetime": "date",
            "open": "open",
            "high": "high",
            "low": "low",
            "close": "close",
            "volume": "volume"
        })

        # Convert to numeric types
        numeric_columns = ["open", "high", "low", "close"]
        for col in numeric_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # Handle volume
        if "volume" in df.columns:
            df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0)
        else:
            df["volume"] = 0

        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)
        
        # Drop any rows with NaN values in critical columns
        df = df.dropna(subset=numeric_columns)
        
        return df


class ForexPredictor:
    def __init__(self):
        self.model = None
        self.feature_columns = []
        self.is_trained = False
    
    def create_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create simplified technical indicators and features"""
        df = data.copy()
        
        # Basic price features (no NaN creation)
        df['price_range'] = (df['high'] - df['low']) / df['open']
        df['price_change'] = (df['close'] - df['open']) / df['open']
        
        # Simple moving averages with smaller windows to reduce NaN
        df['sma_3'] = df['close'].rolling(window=3, min_periods=1).mean()
        df['sma_5'] = df['close'].rolling(window=5, min_periods=1).mean()
        
        # Price relative to moving averages
        df['price_vs_sma3'] = df['close'] / df['sma_3']
        df['price_vs_sma5'] = df['close'] / df['sma_5']
        
        # Simple volatility (using smaller window)
        df['volatility'] = df['close'].pct_change().rolling(window=5, min_periods=1).std()
        
        # Volume features if available
        if 'volume' in df.columns:
            df['volume_sma'] = df['volume'].rolling(window=5, min_periods=1).mean()
            df['volume_ratio'] = df['volume'] / df['volume_sma']
            # Replace infinite values with 1 (when volume_sma is 0)
            df['volume_ratio'] = df['volume_ratio'].replace([np.inf, -np.inf], 1.0)
        
        # Fill any remaining NaN values
        df = df.fillna(method='bfill').fillna(method='ffill').fillna(0)
        
        return df
    
    def prepare_target(self, data: pd.DataFrame, prediction_horizon: int = 1) -> pd.Series:
        """Create target variable for prediction"""
        # Target: 1 if price will go up in next period, else 0
        future_return = (data['close'].shift(-prediction_horizon) - data['close']) / data['close']
        target = (future_return > 0).astype(int)
        # Fill the last NaN value with 0 (no prediction for last row)
        target = target.fillna(0)
        return target
    
    def train_model(self, data: pd.DataFrame, timeframe: str = "1day"):
        """Train Random Forest model with simplified features"""
        print(f"Original data shape: {data.shape}")
        
        # Create features
        df_with_features = self.create_features(data)
        print(f"After feature creation: {df_with_features.shape}")
        
        # Use shorter prediction horizon to reduce NaN
        prediction_horizon = 1  # Always predict next period
        
        target = self.prepare_target(df_with_features, prediction_horizon)
        print(f"Target shape: {target.shape}")
        print(f"Target value counts:\n{target.value_counts()}")
        
        # Define simple feature columns
        self.feature_columns = [
            'open', 'high', 'low', 'close',
            'price_range', 'price_change', 
            'sma_3', 'sma_5', 'price_vs_sma3', 'price_vs_sma5',
            'volatility'
        ]
        
        # Add volume features if available
        if 'volume' in df_with_features.columns:
            self.feature_columns.extend(['volume', 'volume_ratio'])
        
        print(f"Using features: {self.feature_columns}")
        
        # Prepare features and target
        X = df_with_features[self.feature_columns]
        y = target
        
        # Final check for NaN values
        print(f"NaN in features: {X.isna().sum().sum()}")
        print(f"NaN in target: {y.isna().sum()}")
        
        # Remove any remaining NaN rows (should be very few now)
        valid_indices = ~(X.isna().any(axis=1) | y.isna())
        X_clean = X[valid_indices]
        y_clean = y[valid_indices]
        
        print(f"After final cleaning: {len(X_clean)} samples")
        
        if len(X_clean) < 20:
            raise ValueError(f"Not enough valid data for training (only {len(X_clean)} samples available, minimum 20 required)")
        
        # Use 80-20 split
        split_idx = int(len(X_clean) * 0.8)
        X_train, X_val = X_clean.iloc[:split_idx], X_clean.iloc[split_idx:]
        y_train, y_val = y_clean.iloc[:split_idx], y_clean.iloc[split_idx:]
        
        print(f"Training set: {len(X_train)} samples")
        print(f"Validation set: {len(X_val)} samples")
        
        # Simple model parameters
        self.model = RandomForestClassifier(
            n_estimators=50,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        
        self.model.fit(X_train, y_train)
        
        # Validate model
        if len(X_val) > 0:
            y_pred = self.model.predict(X_val)
            accuracy = accuracy_score(y_val, y_pred)
            print(f"Model trained! Validation accuracy: {accuracy:.3f}")
            
            # Show feature importance
            feature_importance = pd.DataFrame({
                'feature': self.feature_columns,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            print("Feature importance:")
            print(feature_importance)
        
        self.is_trained = True
        return self.model
    
    def predict(self, latest_data: pd.DataFrame):
        """Make prediction using trained model"""
        if not self.is_trained or self.model is None:
            raise ValueError("Model must be trained before making predictions")
        
        # Create features for latest data
        latest_with_features = self.create_features(latest_data)
        
        # Get the most recent row
        X_new = latest_with_features[self.feature_columns].iloc[-1:]
        
        # Ensure no NaN values
        if X_new.isna().any().any():
            X_new = X_new.fillna(0)
            print("Warning: NaN values found in prediction data, filled with 0")
        
        prediction = self.model.predict(X_new)[0]
        probabilities = self.model.predict_proba(X_new)[0]
        confidence = max(probabilities)
        
        print(f"Prediction: {prediction}, Confidence: {confidence:.3f}")
        print(f"Probabilities: {probabilities}")
        
        return prediction, confidence, probabilities


class WebForexPredictor:
    def __init__(self):
        self.data_fetcher = ForexDataFetcher()
        self.model = ForexPredictor()

    def perform_web_analysis(self, pair: str, timeframe: str):
        """Adapted analysis for web output"""
        try:
            # Fetch data using your existing fetcher
            df = self.data_fetcher.fetch_forex_data(pair, timeframe)
            
            if len(df) < 20:
                return {'error': 'Not enough historical data for analysis (minimum 20 data points required)'}
            
            # Train model using your existing predictor
            self.model.train_model(df, timeframe)
            
            # Make prediction using your existing predictor
            prediction, confidence, probabilities = self.model.predict(df)
            
            # Get current price info
            current_price = df.iloc[-1]['close']
            previous_price = df.iloc[-2]['close']
            price_change = current_price - previous_price
            change_percent = (price_change / previous_price) * 100
            
            # Generate recommendation (your existing logic)
            if prediction == 1 and confidence > 0.7:
                recommendation = "STRONG BUY"
                risk_level = "Low Risk"
                color = "green"
            elif prediction == 1 and confidence > 0.6:
                recommendation = "MODERATE BUY" 
                risk_level = "Medium Risk"
                color = "orange"
            elif prediction == 1:
                recommendation = "WEAK BUY"
                risk_level = "High Risk"
                color = "red"
            elif prediction == 0 and confidence > 0.7:
                recommendation = "STRONG SELL"
                risk_level = "Low Risk"
                color = "green"
            elif prediction == 0 and confidence > 0.6:
                recommendation = "MODERATE SELL"
                risk_level = "Medium Risk"
                color = "orange"
            else:
                recommendation = "WEAK SELL"
                risk_level = "High Risk"
                color = "red"
            
            # Timeframe advice (your existing logic)
            timeframe_advice = self._get_timeframe_advice(timeframe)
            
            return {
                'success': True,
                'symbol': pair,
                'timeframe': timeframe,
                'recommendation': recommendation,
                'confidence': round(confidence * 100, 1),
                'risk_level': risk_level,
                'color': color,
                'current_price': round(current_price, 4),
                'price_change': round(price_change, 4),
                'change_percent': round(change_percent, 2),
                'data_points': len(df),
                'latest_date': df.iloc[-1]['date'].strftime('%Y-%m-%d %H:%M'),
                'timeframe_advice': timeframe_advice,
                'probabilities': {
                    'buy_probability': round(probabilities[1] * 100, 1),
                    'sell_probability': round(probabilities[0] * 100, 1)
                }
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def _get_timeframe_advice(self, timeframe: str) -> str:
        """Your existing method - unchanged"""
        advice_map = {
            "1min": "Scalping strategy. Very short-term movements. High frequency trading.",
            "5min": "Day trading strategy. Short-term price actions.",
            "15min": "Swing trading. Medium-term trends.",
            "1h": "Swing to position trading. Good for daily trends.",
            "4h": "Position trading. Captures multi-day trends.",
            "1day": "Long-term investing. Weekly/Monthly trends.",
            "1week": "Investment horizon. Long-term positions.",
            "1month": "Strategic investing. Very long-term views."
        }
        return advice_map.get(timeframe, "Consider your trading strategy and risk tolerance.")