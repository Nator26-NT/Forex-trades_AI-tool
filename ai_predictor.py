# ai_predictor.py
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
import talib
from config import API_KEY, BASE_URL
import os

class PatternAnalyzer:
    def __init__(self):
        self.patterns_df = None
        self.meta_df = None
        self.segmentation_df = None
        self.load_pattern_data()
    
    def load_pattern_data(self):
        """Load pattern data from CSV files"""
        try:
            # Load pattern metadata
            if os.path.exists('Meta.csv'):
                self.meta_df = pd.read_csv('Meta.csv')
            
            # Load pattern details
            if os.path.exists('Patterns.csv'):
                self.patterns_df = pd.read_csv('Patterns.csv')
            
            # Load segmentation data
            if os.path.exists('Segmentation.csv'):
                self.segmentation_df = pd.read_csv('Segmentation.csv')
                
        except Exception as e:
            print(f"Warning: Could not load pattern data: {e}")
    
    def classify_market_condition(self, df):
        """
        Classify market condition based on pattern data and current price action
        """
        if self.patterns_df is None:
            return "unknown"
        
        try:
            # Analyze recent price action
            recent_data = df.tail(20)
            volatility = recent_data['high'].std() / recent_data['close'].mean()
            trend_strength = self.calculate_trend_strength(recent_data)
            
            # Count pattern occurrences
            double_top_count = len(self.patterns_df[self.patterns_df['ClassName'] == 'Double top'])
            double_bottom_count = len(self.patterns_df[self.patterns_df['ClassName'] == 'Double bottom'])
            
            # Determine market condition
            if volatility < 0.001 and trend_strength < 0.3:
                return "consolidation"
            elif trend_strength > 0.7:
                return "strong_trend"
            elif double_top_count > double_bottom_count:
                return "potential_reversal_bearish"
            elif double_bottom_count > double_top_count:
                return "potential_reversal_bullish"
            else:
                return "mixed"
                
        except Exception as e:
            print(f"Error classifying market condition: {e}")
            return "unknown"
    
    def calculate_trend_strength(self, df):
        """Calculate trend strength using linear regression"""
        if len(df) < 5:
            return 0
            
        x = np.arange(len(df))
        y = df['close'].values
        
        # Linear regression for trend
        slope = np.polyfit(x, y, 1)[0]
        avg_price = y.mean()
        trend_strength = abs(slope / avg_price) if avg_price != 0 else 0
        
        return min(trend_strength * 1000, 1.0)  # Normalize to 0-1

class AdvancedForexPredictor:
    def __init__(self, lookback_period=50, regression_window=20):
        self.model = None
        self.feature_columns = []
        self.is_trained = False
        self.lookback_period = lookback_period
        self.regression_window = regression_window
        self.pattern_analyzer = PatternAnalyzer()
        
    def calculate_confidence_tier(self, confidence):
        if confidence >= 0.7:
            return "high", 1, 50
        elif confidence >= 0.6:
            return "medium", 3, 30
        elif confidence >= 0.5:
            return "low", 5, 20
        else:
            return "very_low", 0, 0
    
    def detect_three_line_strike(self, df):
        """
        Detect Three Line Strike candlestick pattern
        Returns: 1 for bullish, -1 for bearish, 0 for no pattern
        """
        if len(df) < 4:
            return 0
            
        # Get last 4 candles
        opens = df['open'].values[-4:]
        highs = df['high'].values[-4:]
        lows = df['low'].values[-4:]
        closes = df['close'].values[-4:]
        
        # Bullish Three Line Strike
        if (closes[0] < opens[0] and  # First candle bearish
            closes[1] < opens[1] and  # Second candle bearish  
            closes[2] < opens[2] and  # Third candle bearish
            closes[3] > opens[3] and  # Fourth candle bullish
            closes[3] > opens[0] and  # Closes above first open
            lows[3] <= lows[0:3].min()):  # Tests previous lows
            return 1
            
        # Bearish Three Line Strike
        elif (closes[0] > opens[0] and  # First candle bullish
              closes[1] > opens[1] and  # Second candle bullish
              closes[2] > opens[2] and  # Third candle bullish
              closes[3] < opens[3] and  # Fourth candle bearish
              closes[3] < opens[0] and  # Closes below first open
              highs[3] >= highs[0:3].max()):  # Tests previous highs
            return -1
        else:
            return 0
    
    def detect_double_patterns(self, df):
        """
        Detect double top and double bottom patterns using pattern database
        Returns: dict with pattern information
        """
        if len(df) < 10:
            return {'double_top': False, 'double_bottom': False, 'pattern_confidence': 0}
        
        # Use pattern analyzer to detect double patterns
        market_condition = self.pattern_analyzer.classify_market_condition(df)
        
        double_top = "potential_reversal_bearish" in market_condition
        double_bottom = "potential_reversal_bullish" in market_condition
        
        # Calculate pattern confidence based on recent volatility and trend
        recent_volatility = df['close'].pct_change().std()
        pattern_confidence = min(recent_volatility * 1000, 0.8)  # Normalize confidence
        
        return {
            'double_top': double_top,
            'double_bottom': double_bottom,
            'pattern_confidence': pattern_confidence,
            'market_condition': market_condition
        }
    
    def calculate_support_resistance(self, df, window=20):
        """
        Identify dynamic support and resistance levels
        """
        if len(df) < window:
            return {'support': 0, 'resistance': 0, 'pivot': 0}
            
        recent_highs = df['high'].rolling(window=window).max()
        recent_lows = df['low'].rolling(window=window).min()
        
        resistance = recent_highs.iloc[-1]
        support = recent_lows.iloc[-1]
        
        # Pivot point levels
        pivot = (df['high'].iloc[-1] + df['low'].iloc[-1] + df['close'].iloc[-1]) / 3
        r1 = 2 * pivot - df['low'].iloc[-1]
        s1 = 2 * pivot - df['high'].iloc[-1]
        
        return {
            'support': min(support, s1),
            'resistance': max(resistance, r1),
            'pivot': pivot
        }
    
    def linear_regression_trend(self, prices):
        """
        Calculate linear regression slope and strength for trend confirmation
        """
        if len(prices) < self.regression_window:
            return 0, 0
            
        x = np.arange(len(prices)).reshape(-1, 1)
        y = prices.values
        
        model = LinearRegression()
        model.fit(x, y)
        
        slope = model.coef_[0]
        y_pred = model.predict(x)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        return slope, r_squared
    
    def calculate_regression_channel(self, df, window=20):
        """
        Calculate regression channel for trend confirmation
        """
        if len(df) < window:
            return None
            
        closes = df['close'].iloc[-window:]
        x = np.arange(len(closes)).reshape(-1, 1)
        
        model = LinearRegression()
        model.fit(x, closes)
        
        y_pred = model.predict(x)
        residuals = closes - y_pred
        std_dev = np.std(residuals)
        
        return {
            'slope': model.coef_[0],
            'intercept': model.intercept_,
            'upper_channel': y_pred[-1] + 2 * std_dev,
            'lower_channel': y_pred[-1] - 2 * std_dev,
            'middle_line': y_pred[-1],
            'volatility': std_dev
        }
    
    def enhanced_linear_regression_analysis(self, df):
        """
        Enhanced regression analysis with multiple timeframes
        """
        timeframes = [5, 10, 20, 50]
        results = {}
        
        for tf in timeframes:
            if len(df) >= tf:
                prices = df['close'].iloc[-tf:]
                slope, r_squared = self.linear_regression_trend(prices)
                results[f'regression_tf_{tf}'] = {
                    'slope': slope,
                    'r_squared': r_squared,
                    'trend_strength': r_squared,
                    'direction': 'up' if slope > 0 else 'down'
                }
        
        return results
    
    def calculate_signal_score(self, df, current_price):
        """
        Calculate comprehensive signal score based on multiple factors
        """
        score = 0
        max_score = 12  # Increased max score for additional factors

        # 1. Three Line Strike Pattern (20% weight)
        strike_pattern = self.detect_three_line_strike(df)
        if strike_pattern != 0:
            score += 2.4
        
        # 2. Double Pattern Detection (20% weight)
        double_patterns = self.detect_double_patterns(df)
        if double_patterns['double_top'] or double_patterns['double_bottom']:
            score += 2.4 * double_patterns['pattern_confidence']
        
        # 3. Support/Resistance Alignment (20% weight)
        levels = self.calculate_support_resistance(df)
        if levels:
            distance_to_support = abs(current_price - levels['support'])
            distance_to_resistance = abs(current_price - levels['resistance'])
            price_range = levels['resistance'] - levels['support']
            
            if price_range > 0:
                support_proximity = 1 - (distance_to_support / price_range)
                resistance_proximity = 1 - (distance_to_resistance / price_range)
                score += 2.4 * max(support_proximity, resistance_proximity)
        
        # 4. Regression Trend Strength (30% weight)
        closes = df['close'].iloc[-self.regression_window:]
        slope, r_squared = self.linear_regression_trend(closes)
        
        trend_strength = min(r_squared * 4, 1.0)
        score += 3.6 * trend_strength
        
        # 5. Enhanced Regression Analysis (10% weight)
        enhanced_regression = self.enhanced_linear_regression_analysis(df)
        multi_timeframe_score = 0
        for tf_result in enhanced_regression.values():
            multi_timeframe_score += tf_result['trend_strength']
        
        if enhanced_regression:
            multi_timeframe_score /= len(enhanced_regression)
            score += 1.2 * multi_timeframe_score
        
        return min(score, max_score), max_score, double_patterns
    
    def create_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create enhanced features including regression and pattern features"""
        df = data.copy()
        
        # Basic price features
        df['price_range'] = (df['high'] - df['low']) / df['open']
        df['price_change'] = (df['close'] - df['open']) / df['open']
        df['body_size'] = abs(df['close'] - df['open']) / (df['high'] - df['low']).replace(0, 0.0001)
        
        # Moving averages
        for window in [3, 5, 8, 20]:
            df[f'sma_{window}'] = df['close'].rolling(window=window, min_periods=1).mean()
            df[f'price_vs_sma{window}'] = df['close'] / df[f'sma_{window}'].replace(0, 1)
        
        # Volatility
        df['volatility'] = df['close'].pct_change().rolling(window=5, min_periods=1).std().fillna(0)
        
        # Volume features
        if 'volume' in df.columns:
            df['volume_sma'] = df['volume'].rolling(window=5, min_periods=1).mean()
            df['volume_ratio'] = df['volume'] / df['volume_sma'].replace(0, 1)
            df['volume_ratio'] = df['volume_ratio'].replace([np.inf, -np.inf], 1.0)
        
        # Linear regression features
        regression_features = []
        for i in range(len(df)):
            if i >= self.regression_window:
                window_data = df['close'].iloc[i-self.regression_window:i]
                slope, r_squared = self.linear_regression_trend(window_data)
                regression_features.append({
                    'regression_slope': slope,
                    'trend_strength': r_squared
                })
            else:
                regression_features.append({'regression_slope': 0, 'trend_strength': 0})
        
        regression_df = pd.DataFrame(regression_features, index=df.index)
        df = pd.concat([df, regression_df], axis=1)
        
        # Pattern detection
        pattern_features = []
        for i in range(len(df)):
            if i >= 4:
                window_data = df.iloc[i-4:i+1]
                three_line_strike = self.detect_three_line_strike(window_data)
                pattern_features.append({
                    'three_line_strike': three_line_strike
                })
            else:
                pattern_features.append({'three_line_strike': 0})
        
        pattern_df = pd.DataFrame(pattern_features, index=df.index)
        df = pd.concat([df, pattern_df], axis=1)
        
        # Market condition features
        market_condition_features = []
        for i in range(len(df)):
            if i >= 20:
                window_data = df.iloc[:i+1]  # All data up to current point
                double_patterns = self.detect_double_patterns(window_data)
                market_condition_features.append({
                    'market_condition_score': 1 if 'strong_trend' in double_patterns['market_condition'] else 0.5,
                    'pattern_based_signal': 1 if double_patterns['double_bottom'] else -1 if double_patterns['double_top'] else 0
                })
            else:
                market_condition_features.append({'market_condition_score': 0, 'pattern_based_signal': 0})
        
        market_condition_df = pd.DataFrame(market_condition_features, index=df.index)
        df = pd.concat([df, market_condition_df], axis=1)
        
        return df.fillna(0)
    
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
            'body_size', 'sma_3', 'sma_5', 'sma_8', 'sma_20',
            'price_vs_sma3', 'price_vs_sma5', 'price_vs_sma8', 'price_vs_sma20',
            'volatility', 'regression_slope', 'trend_strength', 'three_line_strike',
            'market_condition_score', 'pattern_based_signal'
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
            n_estimators=50,
            max_depth=10,
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
        
        # Get additional signal information
        current_price = latest_data['close'].iloc[-1]
        signal_score, max_score, double_patterns = self.calculate_signal_score(latest_data, current_price)
        levels = self.calculate_support_resistance(latest_data)
        channel = self.calculate_regression_channel(latest_data)
        strike_pattern = self.detect_three_line_strike(latest_data)
        enhanced_regression = self.enhanced_linear_regression_analysis(latest_data)
        
        return {
            'prediction': prediction,
            'confidence': min(confidence, 0.95),
            'probabilities': probabilities,
            'signal_score': signal_score,
            'max_score': max_score,
            'support_level': levels['support'],
            'resistance_level': levels['resistance'],
            'regression_slope': channel['slope'] if channel else 0,
            'three_line_strike': strike_pattern,
            'double_patterns': double_patterns,
            'enhanced_regression': enhanced_regression,
            'current_price': current_price,
            'market_condition': double_patterns['market_condition']
        }