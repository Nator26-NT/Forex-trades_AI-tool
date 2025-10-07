import pandas as pd
import requests
import time
import json
from config import API_KEY, BASE_URL, TIMEFRAME_INTERVALS, OUTPUTSIZE_MAP, RATE_LIMIT_DELAY

class ForexDataFetcher:
    def __init__(self):
        self.api_key = API_KEY
        self.base_url = BASE_URL
    
    def fetch_forex_data(self, symbol: str, timeframe: str = "1h"):
        if timeframe not in TIMEFRAME_INTERVALS:
            raise ValueError(f"Unsupported timeframe: {timeframe}")
        
        symbol_td = f"{symbol[:3]}/{symbol[3:]}"
        interval = TIMEFRAME_INTERVALS[timeframe]
        outputsize = OUTPUTSIZE_MAP.get(timeframe, 30)
        
        url = f"{self.base_url}?symbol={symbol_td}&interval={interval}&outputsize={outputsize}&apikey={self.api_key}"

        try:
            response = requests.get(url, timeout=15)
            
            if response.status_code != 200:
                return None
            
            try:
                data = response.json()
            except json.JSONDecodeError:
                return None
        
            if "values" not in data:
                return None

            df = pd.DataFrame(data["values"])
            df = self._process_dataframe(df)
            
            if len(df) < 10:
                return None
            
            time.sleep(RATE_LIMIT_DELAY)
            return df
            
        except requests.exceptions.Timeout:
            return None
        except requests.exceptions.RequestException:
            return None
        except Exception:
            return None
    
    def _process_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        column_map = {
            "datetime": "date", "open": "open", "high": "high", 
            "low": "low", "close": "close", "volume": "volume"
        }
        
        df = df.rename(columns={k: v for k, v in column_map.items() if k in df.columns})

        numeric_columns = ["open", "high", "low", "close"]
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        if "volume" in df.columns:
            df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0)
        else:
            df["volume"] = 0

        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
            df = df.sort_values("date").reset_index(drop=True)
        
        df = df.fillna(method='ffill').fillna(method='bfill')
        return df