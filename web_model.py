# web_model.py
import requests
import pandas as pd
import time
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from config import API_KEY, BASE_URL, TIMEFRAME_INTERVALS, RATE_LIMIT_DELAY
from ai_predictor import AdvancedForexPredictor

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
    
    def fetch_forex_data_without_api(self, symbol: str, timeframe: str = "1day"):
        """
        Simulate forex data without using API (for demo/testing)
        This generates synthetic data based on pattern analysis
        """
        # Generate synthetic data based on patterns
        days = 100
        base_price = 1.1000  # Base price for EURUSD
        
        dates = pd.date_range(end=pd.Timestamp.now(), periods=days, freq='D')
        
        # Create price data with trends and patterns
        np.random.seed(42)  # For reproducible results
        returns = np.random.normal(0.0001, 0.005, days)
        prices = base_price * (1 + returns).cumprod()
        
        # Add some patterns based on the pattern database
        patterns = self._apply_pattern_effects(prices, days)
        
        df = pd.DataFrame({
            'date': dates,
            'open': patterns['open'],
            'high': patterns['high'],
            'low': patterns['low'],
            'close': patterns['close'],
            'volume': np.random.randint(1000000, 5000000, days)
        })
        
        return df
    
    def _apply_pattern_effects(self, prices, days):
        """Apply pattern-based effects to synthetic data"""
        opens = []
        highs = []
        lows = []
        closes = []
        
        for i in range(days):
            base_price = prices[i]
            volatility = 0.002  # 0.2% daily volatility
            
            # Simulate OHLC prices
            open_price = base_price
            close_price = base_price * (1 + np.random.normal(0, volatility))
            
            # Calculate high and low with pattern influences
            price_range = abs(close_price - open_price)
            high = max(open_price, close_price) + price_range * 0.5
            low = min(open_price, close_price) - price_range * 0.5
            
            # Apply double top/bottom patterns occasionally
            if i % 20 == 15:  # Simulate double top
                high = open_price * 1.01
                close_price = open_price * 0.995
                low = open_price * 0.99
            elif i % 20 == 5:  # Simulate double bottom
                low = open_price * 0.99
                close_price = open_price * 1.005
                high = open_price * 1.01
            
            opens.append(open_price)
            highs.append(high)
            lows.append(low)
            closes.append(close_price)
        
        return {
            'open': opens,
            'high': highs,
            'low': lows,
            'close': closes
        }
    
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

class WebForexPredictor:
    def __init__(self):
        self.data_fetcher = ForexDataFetcher()
        self.predictor = AdvancedForexPredictor()

    def perform_web_analysis(self, pair: str, timeframe: str, use_api: bool = True):
        """Enhanced analysis using advanced AI predictor with pattern recognition"""
        try:
            # Fetch data
            if use_api:
                df = self.data_fetcher.fetch_forex_data(pair, timeframe)
            else:
                df = self.data_fetcher.fetch_forex_data_without_api(pair, timeframe)
            
            if len(df) < 50:
                return {'error': 'Not enough historical data for advanced analysis (minimum 50 data points required)'}
            
            # Train model using advanced predictor
            self.predictor.train_model(df, timeframe)
            
            # Make prediction with enhanced features
            result = self.predictor.predict_with_confidence(df)
            
            # Get additional price info
            current_price = df.iloc[-1]['close']
            previous_price = df.iloc[-2]['close']
            price_change = current_price - previous_price
            change_percent = (price_change / previous_price) * 100
            
            # Enhanced recommendation based on multiple factors
            prediction = result['prediction']
            confidence = result['confidence']
            signal_score = result['signal_score']
            max_score = result['max_score']
            market_condition = result['market_condition']
            
            # Generate comprehensive recommendation
            recommendation, risk_level, color = self._generate_recommendation(
                prediction, confidence, signal_score, market_condition
            )
            
            # Enhanced timeframe advice
            timeframe_advice = self._get_enhanced_timeframe_advice(timeframe, result, market_condition)
            
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
                    'buy_probability': round(result['probabilities'][1] * 100, 1),
                    'sell_probability': round(result['probabilities'][0] * 100, 1)
                },
                # Enhanced features
                'signal_score': signal_score,
                'max_score': max_score,
                'support_level': round(result['support_level'], 4),
                'resistance_level': round(result['resistance_level'], 4),
                'regression_slope': round(result['regression_slope'], 6),
                'three_line_strike': result['three_line_strike'],
                'double_patterns': result['double_patterns'],
                'enhanced_regression': result['enhanced_regression'],
                'market_condition': market_condition,
                'pattern_alignment': self._get_pattern_alignment(
                    result['three_line_strike'], 
                    result['double_patterns'], 
                    prediction
                ),
                'data_source': 'API' if use_api else 'Synthetic'
            }
            
        except Exception as e:
            return {'error': f'Advanced analysis failed: {str(e)}'}
    
    def _generate_recommendation(self, prediction, confidence, signal_score, market_condition):
        """Generate trading recommendation based on multiple factors"""
        # Base recommendation
        if prediction == 1 and confidence > 0.7 and signal_score > 8:
            base_rec = "STRONG BUY"
            risk_level = "Low Risk"
            color = "green"
        elif prediction == 1 and confidence > 0.6 and signal_score > 6:
            base_rec = "MODERATE BUY" 
            risk_level = "Medium Risk"
            color = "orange"
        elif prediction == 1:
            base_rec = "WEAK BUY"
            risk_level = "High Risk"
            color = "red"
        elif prediction == 0 and confidence > 0.7 and signal_score > 8:
            base_rec = "STRONG SELL"
            risk_level = "Low Risk"
            color = "green"
        elif prediction == 0 and confidence > 0.6 and signal_score > 6:
            base_rec = "MODERATE SELL"
            risk_level = "Medium Risk"
            color = "orange"
        else:
            base_rec = "WEAK SELL"
            risk_level = "High Risk"
            color = "red"
        
        # Adjust based on market condition
        if "strong_trend" in market_condition:
            base_rec += " | STRONG TREND"
        elif "consolidation" in market_condition:
            base_rec += " | CONSOLIDATION"
            risk_level = "Higher Risk"
        elif "reversal" in market_condition:
            base_rec += " | POTENTIAL REVERSAL"
        
        return base_rec, risk_level, color
    
    def _get_enhanced_timeframe_advice(self, timeframe: str, result: dict, market_condition: str) -> str:
        """Get enhanced timeframe advice including regression analysis and patterns"""
        base_advice = {
            "1min": "Scalping strategy. Very short-term movements. High frequency trading.",
            "5min": "Day trading strategy. Short-term price actions.",
            "15min": "Swing trading. Medium-term trends.",
            "1h": "Swing to position trading. Good for daily trends.",
            "4h": "Position trading. Captures multi-day trends.",
            "1day": "Long-term investing. Weekly/Monthly trends.",
            "1week": "Investment horizon. Long-term positions.",
            "1month": "Strategic investing. Very long-term views."
        }.get(timeframe, "Consider your trading strategy and risk tolerance.")
        
        # Add regression trend info
        slope = result.get('regression_slope', 0)
        if slope > 0.001:
            trend_info = " 📈 Strong uptrend confirmed by regression analysis."
        elif slope < -0.001:
            trend_info = " 📉 Strong downtrend confirmed by regression analysis."
        else:
            trend_info = " ➡️ Sideways market - exercise caution."
            
        # Add pattern info
        pattern_info = ""
        three_line_strike = result.get('three_line_strike', 0)
        if three_line_strike == 1:
            pattern_info += " 🎯 Bullish Three-Line Strike pattern detected."
        elif three_line_strike == -1:
            pattern_info += " 🎯 Bearish Three-Line Strike pattern detected."
            
        double_patterns = result.get('double_patterns', {})
        if double_patterns.get('double_top'):
            pattern_info += " ⚠️ Double Top pattern detected (Bearish)."
        if double_patterns.get('double_bottom'):
            pattern_info += " ⚠️ Double Bottom pattern detected (Bullish)."
            
        # Add market condition info
        market_info = f" 📊 Market Condition: {market_condition.replace('_', ' ').title()}"
            
        return base_advice + trend_info + pattern_info + market_info
    
    def _get_pattern_alignment(self, three_line_strike: int, double_patterns: dict, prediction: int) -> str:
        """Check if pattern aligns with prediction"""
        patterns = []
        
        if three_line_strike == 1:
            patterns.append("Three-Line Strike Bullish")
        elif three_line_strike == -1:
            patterns.append("Three-Line Strike Bearish")
            
        if double_patterns.get('double_top'):
            patterns.append("Double Top Bearish")
        if double_patterns.get('double_bottom'):
            patterns.append("Double Bottom Bullish")
        
        if not patterns:
            return "No significant pattern detected"
        
        # Check if patterns align with prediction
        bullish_patterns = [p for p in patterns if 'Bullish' in p]
        bearish_patterns = [p for p in patterns if 'Bearish' in p]
        
        if prediction == 1 and bullish_patterns and not bearish_patterns:
            return "All patterns align with BUY signal"
        elif prediction == 0 and bearish_patterns and not bullish_patterns:
            return "All patterns align with SELL signal"
        elif prediction == 1 and bearish_patterns:
            return "Warning: Some bearish patterns contradict BUY signal"
        elif prediction == 0 and bullish_patterns:
            return "Warning: Some bullish patterns contradict SELL signal"
        else:
            return "Mixed pattern signals"