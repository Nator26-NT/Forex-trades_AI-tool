# data_fetcher.py
import pandas as pd
import requests
import time
import json
import numpy as np
from datetime import datetime, timedelta
from config import API_KEY, BASE_URL, TIMEFRAME_INTERVALS, OUTPUTSIZE_MAP, RATE_LIMIT_DELAY

class AdvancedForexDataFetcher:
    def __init__(self):
        self.api_key = API_KEY
        self.base_url = BASE_URL
        self.pattern_cache = {}
    
    def fetch_forex_data(self, symbol: str, timeframe: str = "1h"):
        """Fetch forex data from Twelve Data API with enhanced error handling"""
        if timeframe not in TIMEFRAME_INTERVALS:
            raise ValueError(f"Unsupported timeframe: {timeframe}")
        
        symbol_td = f"{symbol[:3]}/{symbol[3:]}"
        interval = TIMEFRAME_INTERVALS[timeframe]
        outputsize = OUTPUTSIZE_MAP.get(timeframe, 100)
        
        url = f"{self.base_url}?symbol={symbol_td}&interval={interval}&outputsize={outputsize}&apikey={self.api_key}"

        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            if "values" not in data:
                if "message" in data:
                    error_msg = data["message"]
                    # Check for specific API errors
                    if "invalid API key" in error_msg.lower():
                        raise ValueError("Invalid API key. Please check your Twelve Data API configuration.")
                    elif "limit" in error_msg.lower():
                        raise ValueError("API rate limit exceeded. Please try again later.")
                    else:
                        raise ValueError(f"API Error: {error_msg}")
                raise ValueError(f"Unexpected API response: {data}")

            df = pd.DataFrame(data["values"])
            df = self._process_dataframe(df)
            
            # Enhance data with technical features
            df = self._add_technical_features(df)
            
            time.sleep(RATE_LIMIT_DELAY)
            return df
            
        except requests.exceptions.Timeout:
            raise ValueError("API request timed out. Please try again.")
        except requests.exceptions.ConnectionError:
            raise ValueError("Network connection error. Please check your internet connection.")
        except requests.exceptions.HTTPError as e:
            raise ValueError(f"HTTP error: {str(e)}")
        except json.JSONDecodeError:
            raise ValueError("Invalid JSON response from API.")
        except Exception as e:
            raise ValueError(f"Unexpected error fetching data: {str(e)}")
    
    def fetch_forex_data_without_api(self, symbol: str, timeframe: str = "1h", days: int = 100):
        """
        Generate sophisticated synthetic forex data with realistic patterns
        """
        # Set seed for reproducibility
        np.random.seed(42)
        
        # Generate date range
        end_date = datetime.now()
        if timeframe in ["1min", "5min", "15min", "30min"]:
            freq = '1H'  # Hourly for intraday
            start_date = end_date - timedelta(days=30)
        else:
            freq = '1D'  # Daily for longer timeframes
            start_date = end_date - timedelta(days=days)
        
        dates = pd.date_range(start=start_date, end=end_date, freq=freq)
        
        # Base parameters for different symbols
        symbol_params = {
            "EURUSD": {"base_price": 1.1000, "volatility": 0.0008, "trend_bias": 0.0001},
            "GBPUSD": {"base_price": 1.2500, "volatility": 0.0010, "trend_bias": 0.0002},
            "USDJPY": {"base_price": 150.00, "volatility": 0.0012, "trend_bias": -0.0001},
            "AUDUSD": {"base_price": 0.6500, "volatility": 0.0015, "trend_bias": 0.0003},
            "USDCAD": {"base_price": 1.3500, "volatility": 0.0009, "trend_bias": -0.0002},
            "USDCHF": {"base_price": 0.8800, "volatility": 0.0007, "trend_bias": 0.0001},
            "NZDUSD": {"base_price": 0.6000, "volatility": 0.0016, "trend_bias": 0.0004},
        }
        
        params = symbol_params.get(symbol, {"base_price": 1.0000, "volatility": 0.0010, "trend_bias": 0.0001})
        
        # Generate price series with realistic patterns
        prices = self._generate_realistic_price_series(
            len(dates), 
            params["base_price"], 
            params["volatility"], 
            params["trend_bias"]
        )
        
        # Apply technical patterns
        ohlc_data = self._apply_technical_patterns(prices, dates, symbol)
        
        df = pd.DataFrame({
            'date': dates,
            'open': ohlc_data['open'],
            'high': ohlc_data['high'],
            'low': ohlc_data['low'],
            'close': ohlc_data['close'],
            'volume': np.random.randint(1000000, 5000000, len(dates))
        })
        
        # Add technical features
        df = self._add_technical_features(df)
        
        return df
    
    def _generate_realistic_price_series(self, n_points, base_price, volatility, trend_bias):
        """Generate realistic price series with trends and mean reversion"""
        prices = [base_price]
        
        for i in range(1, n_points):
            # Basic random walk with trend
            change = np.random.normal(trend_bias, volatility)
            
            # Add mean reversion component
            mean_reversion = (base_price - prices[-1]) * 0.01
            change += mean_reversion
            
            # Add momentum effect
            if i > 5:
                recent_trend = np.mean(np.diff(prices[-5:]))
                change += recent_trend * 0.1
            
            new_price = prices[-1] * (1 + change)
            prices.append(new_price)
        
        return prices
    
    def _apply_technical_patterns(self, prices, dates, symbol):
        """Apply realistic technical patterns to price data"""
        n = len(prices)
        opens = []
        highs = []
        lows = []
        closes = []
        
        for i in range(n):
            base_price = prices[i]
            
            # Determine pattern type based on position in series
            pattern_type = self._get_pattern_type(i, n)
            
            if pattern_type == "double_top" and i > n//3 and i < 2*n//3:
                # Create double top pattern
                if i % 20 == 15:
                    open_price = base_price
                    high = base_price * 1.008
                    close_price = base_price * 0.998
                    low = base_price * 0.992
                elif i % 20 == 16:
                    open_price = base_price * 0.998
                    high = base_price * 1.006
                    close_price = base_price * 0.995
                    low = base_price * 0.990
                else:
                    open_price, high, low, close_price = self._generate_normal_candle(base_price)
            
            elif pattern_type == "double_bottom" and i > n//3 and i < 2*n//3:
                # Create double bottom pattern
                if i % 25 == 10:
                    open_price = base_price
                    low = base_price * 0.992
                    close_price = base_price * 1.002
                    high = base_price * 1.008
                elif i % 25 == 11:
                    open_price = base_price * 1.002
                    low = base_price * 0.994
                    close_price = base_price * 1.005
                    high = base_price * 1.010
                else:
                    open_price, high, low, close_price = self._generate_normal_candle(base_price)
            
            elif pattern_type == "three_line_strike" and i > 10:
                # Create three line strike pattern
                if i % 30 == 20:
                    open_price = base_price
                    close_price = base_price * 0.995
                    high = open_price * 1.003
                    low = close_price * 0.997
                elif i % 30 == 21:
                    open_price = base_price * 0.995
                    close_price = base_price * 0.990
                    high = open_price * 1.002
                    low = close_price * 0.998
                elif i % 30 == 22:
                    open_price = base_price * 0.990
                    close_price = base_price * 0.985
                    high = open_price * 1.001
                    low = close_price * 0.999
                elif i % 30 == 23:  # Strike candle
                    open_price = base_price * 0.985
                    close_price = base_price * 1.000
                    high = close_price * 1.005
                    low = open_price * 0.995
                else:
                    open_price, high, low, close_price = self._generate_normal_candle(base_price)
            
            else:
                open_price, high, low, close_price = self._generate_normal_candle(base_price)
            
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
    
    def _get_pattern_type(self, index, total_length):
        """Determine which pattern to apply based on position"""
        patterns = ["double_top", "double_bottom", "three_line_strike", "normal"]
        pattern_weights = [0.1, 0.1, 0.1, 0.7]  # 70% normal, 10% each pattern
        
        # Increase pattern probability in middle of series
        if total_length // 3 < index < 2 * total_length // 3:
            pattern_weights = [0.2, 0.2, 0.2, 0.4]
        
        return np.random.choice(patterns, p=pattern_weights)
    
    def _generate_normal_candle(self, base_price):
        """Generate a normal candlestick with realistic properties"""
        volatility = 0.002  # 0.2% volatility
        
        open_price = base_price
        close_price = base_price * (1 + np.random.normal(0, volatility))
        
        # Ensure high is highest, low is lowest
        price_range = abs(close_price - open_price)
        high = max(open_price, close_price) + price_range * np.random.uniform(0.3, 0.7)
        low = min(open_price, close_price) - price_range * np.random.uniform(0.3, 0.7)
        
        return open_price, high, low, close_price
    
    def _add_technical_features(self, df):
        """Add technical indicators to the dataframe"""
        # Moving averages
        for window in [5, 10, 20, 50]:
            df[f'sma_{window}'] = df['close'].rolling(window=window).mean()
            df[f'ema_{window}'] = df['close'].ewm(span=window).mean()
        
        # RSI
        df['rsi'] = self._calculate_rsi(df['close'])
        
        # MACD
        macd, signal, histogram = self._calculate_macd(df['close'])
        df['macd'] = macd
        df['macd_signal'] = signal
        df['macd_histogram'] = histogram
        
        # Bollinger Bands
        bb_upper, bb_lower, bb_middle = self._calculate_bollinger_bands(df['close'])
        df['bb_upper'] = bb_upper
        df['bb_lower'] = bb_lower
        df['bb_middle'] = bb_middle
        
        # ATR
        df['atr'] = self._calculate_atr(df)
        
        # Volume indicators
        if 'volume' in df.columns:
            df['volume_sma'] = df['volume'].rolling(window=20).mean()
            df['volume_ratio'] = df['volume'] / df['volume_sma']
        
        return df.fillna(method='bfill')
    
    def _calculate_rsi(self, prices, window=14):
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_macd(self, prices, fast=12, slow=26, signal=9):
        """Calculate MACD indicator"""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=signal).mean()
        macd_histogram = macd - macd_signal
        return macd, macd_signal, macd_histogram
    
    def _calculate_bollinger_bands(self, prices, window=20, num_std=2):
        """Calculate Bollinger Bands"""
        middle = prices.rolling(window=window).mean()
        std = prices.rolling(window=window).std()
        upper = middle + (std * num_std)
        lower = middle - (std * num_std)
        return upper, lower, middle
    
    def _calculate_atr(self, df, window=14):
        """Calculate Average True Range"""
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        true_range = np.maximum(np.maximum(high_low, high_close), low_close)
        atr = true_range.rolling(window=window).mean()
        return atr
    
    def _process_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process and clean the dataframe with enhanced features"""
        column_map = {
            "datetime": "date",
            "open": "open", 
            "high": "high",
            "low": "low", 
            "close": "close", 
            "volume": "volume"
        }
        
        df = df.rename(columns={k: v for k, v in column_map.items() if k in df.columns})

        # Convert to numeric types
        numeric_columns = ["open", "high", "low", "close"]
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # Handle volume
        if "volume" in df.columns:
            df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0)
        else:
            df["volume"] = 0

        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
            df = df.sort_values("date").reset_index(drop=True)
        
        # Enhanced data cleaning
        df = df.fillna(method='ffill').fillna(method='bfill')
        
        # Remove any remaining outliers
        for col in numeric_columns:
            if col in df.columns:
                q_low = df[col].quantile(0.01)
                q_high = df[col].quantile(0.99)
                df[col] = df[col].clip(lower=q_low, upper=q_high)
        
        return df

# Legacy class for backward compatibility
class ForexDataFetcher(AdvancedForexDataFetcher):
    def __init__(self):
        super().__init__()