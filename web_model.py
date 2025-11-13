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

    def perform_web_analysis(self, pair: str, timeframe: str):
        """Enhanced analysis using advanced AI predictor with regression confirmation"""
        try:
            # Fetch data using your existing fetcher
            df = self.data_fetcher.fetch_forex_data(pair, timeframe)
            
            if len(df) < 50:  # Increased minimum for regression analysis
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
            
            # Generate comprehensive recommendation
            if prediction == 1 and confidence > 0.7 and signal_score > 7:
                recommendation = "STRONG BUY"
                risk_level = "Low Risk"
                color = "green"
            elif prediction == 1 and confidence > 0.6 and signal_score > 5:
                recommendation = "MODERATE BUY" 
                risk_level = "Medium Risk"
                color = "orange"
            elif prediction == 1:
                recommendation = "WEAK BUY"
                risk_level = "High Risk"
                color = "red"
            elif prediction == 0 and confidence > 0.7 and signal_score > 7:
                recommendation = "STRONG SELL"
                risk_level = "Low Risk"
                color = "green"
            elif prediction == 0 and confidence > 0.6 and signal_score > 5:
                recommendation = "MODERATE SELL"
                risk_level = "Medium Risk"
                color = "orange"
            else:
                recommendation = "WEAK SELL"
                risk_level = "High Risk"
                color = "red"
            
            # Enhanced timeframe advice
            timeframe_advice = self._get_enhanced_timeframe_advice(timeframe, result)
            
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
                'pattern_alignment': self._get_pattern_alignment(result['three_line_strike'], prediction)
            }
            
        except Exception as e:
            return {'error': f'Advanced analysis failed: {str(e)}'}
    
    def _get_enhanced_timeframe_advice(self, timeframe: str, result: dict) -> str:
        """Get enhanced timeframe advice including regression analysis"""
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
            trend_info = " Strong uptrend confirmed by regression analysis."
        elif slope < -0.001:
            trend_info = " Strong downtrend confirmed by regression analysis."
        else:
            trend_info = " Sideways market - exercise caution."
            
        # Add pattern info
        pattern = result.get('three_line_strike', 0)
        if pattern == 1:
            pattern_info = " Bullish Three-Line Strike pattern detected."
        elif pattern == -1:
            pattern_info = " Bearish Three-Line Strike pattern detected."
        else:
            pattern_info = ""
            
        return base_advice + trend_info + pattern_info
    
    def _get_pattern_alignment(self, pattern: int, prediction: int) -> str:
        """Check if pattern aligns with prediction"""
        if pattern == 1 and prediction == 1:
            return "Pattern aligns with BUY signal"
        elif pattern == -1 and prediction == 0:
            return "Pattern aligns with SELL signal"
        elif pattern == 0:
            return "No significant pattern detected"
        else:
            return "Pattern contradicts signal - exercise caution"