# =============================================================================
# CONFIGURATION & CONSTANTS
# =============================================================================

API_KEY = "fe6aec0e85244251ab5cb28263f98bd6"
BASE_URL = "https://api.twelvedata.com/time_series"
RATE_LIMIT_DELAY = 0.5

# Timeframe mapping for Twelve Data API
TIMEFRAME_INTERVALS = {
    "1min": "1min", 
    "5min": "5min", 
    "15min": "15min", 
    "30min": "30min",
    "1h": "1h", 
    "4h": "4h", 
    "1day": "1day", 
    "1week": "1week", 
    "1month": "1month"
}

# All available timeframes
ALL_TIMEFRAMES = ["1min", "5min", "15min", "30min", "1h", "4h", "1day", "1week", "1month"]

# PROFESSIONAL TACTICAL PIP STRUCTURE - Volatility Adjusted
PIP_TARGETS = {
    # Scalping Timeframes - Tighter stops for quick entries
    "1min": {
        "sl_pips": 3, 
        "tp_pips": 6, 
        "description": "Ultra Scalping", 
        "hold_period": "1-5 min",
        "risk_reward": "1:2",
        "volatility_factor": 0.8
    },
    "5min": {
        "sl_pips": 4, 
        "tp_pips": 8, 
        "description": "Momentum Scalping", 
        "hold_period": "10-30 min",
        "risk_reward": "1:2", 
        "volatility_factor": 0.9
    },
    
    # Intraday Timeframes - Balanced risk for day trading
    "15min": {
        "sl_pips": 6, 
        "tp_pips": 12, 
        "description": "Intraday Momentum", 
        "hold_period": "1-4 hours",
        "risk_reward": "1:2",
        "volatility_factor": 1.0
    },
    "30min": {
        "sl_pips": 8, 
        "tp_pips": 16, 
        "description": "Intraday Swing", 
        "hold_period": "2-8 hours",
        "risk_reward": "1:2",
        "volatility_factor": 1.1
    },
    
    # Swing Timeframes - Wider stops for trend following
    "1h": {
        "sl_pips": 10, 
        "tp_pips": 20, 
        "description": "Swing Setup", 
        "hold_period": "6-24 hours",
        "risk_reward": "1:2",
        "volatility_factor": 1.2
    },
    "4h": {
        "sl_pips": 15, 
        "tp_pips": 30, 
        "description": "Swing Trade", 
        "hold_period": "1-3 days",
        "risk_reward": "1:2", 
        "volatility_factor": 1.3
    },
    
    # Position Trading - Maximum room for volatility
    "1day": {
        "sl_pips": 20, 
        "tp_pips": 40, 
        "description": "Position Trade", 
        "hold_period": "3-7 days",
        "risk_reward": "1:2",
        "volatility_factor": 1.5
    },
    "1week": {
        "sl_pips": 30, 
        "tp_pips": 60, 
        "description": "Weekly Position", 
        "hold_period": "1-3 weeks",
        "risk_reward": "1:2",
        "volatility_factor": 1.8
    },
    "1month": {
        "sl_pips": 50, 
        "tp_pips": 100, 
        "description": "Monthly Investment", 
        "hold_period": "1-3 months",
        "risk_reward": "1:2",
        "volatility_factor": 2.0
    }
}

# Output size mapping for optimal data fetching
OUTPUTSIZE_MAP = {
    "1min": 1440,    # 24 hours
    "5min": 576,     # 2 days
    "15min": 384,    # 4 days
    "30min": 288,    # 6 days
    "1h": 168,       # 1 week
    "4h": 126,       # 3 weeks
    "1day": 500,     # ~1.5 years
    "1week": 260,    # 5 years
    "1month": 120    # 10 years
}

# Market Session Multipliers
SESSION_MULTIPLIERS = {
    "asia": 0.8,        # Lower volatility
    "london": 1.2,      # High volatility opening
    "new_york": 1.3,    # Highest volatility
    "overlap": 1.5,     # London/NY overlap - maximum volatility
}

# Currency Pair Volatility Adjustments
PAIR_VOLATILITY = {
    "EURUSD": 1.0,      # Baseline
    "GBPUSD": 1.2,      # 20% more volatile
    "USDJPY": 1.1,      # 10% more volatile
    "AUDUSD": 1.3,      # 30% more volatile
    "USDCAD": 1.1,      # 10% more volatile
    "USDCHF": 0.9,      # 10% less volatile
    "NZDUSD": 1.4,      # 40% more volatile
    "EURGBP": 1.1,      # 10% more volatile
    "default": 1.0
}

# Economic Event Multipliers
EVENT_MULTIPLIERS = {
    "normal": 1.0,
    "high_impact": 1.8,
    "medium_impact": 1.3,
    "low_impact": 1.1
}

# CSS Styles for mobile optimization
MOBILE_CSS = """
<style>
    @media (max-width: 768px) {
        .main-header { 
            font-size: 1.8rem !important; 
            text-align: center; 
            margin-bottom: 1rem; 
            padding: 0.5rem; 
        }
        .row-widget.stColumns { 
            flex-direction: column !important; 
        }
        .row-widget.stColumns > div { 
            width: 100% !important; 
            margin-bottom: 1rem; 
        }
        .stButton button { 
            width: 100% !important; 
            height: 3rem !important; 
            font-size: 1.1rem !important; 
            margin: 0.25rem 0 !important; 
        }
        .stTextInput input, .stSelectbox select, .stNumberInput input { 
            font-size: 1.1rem !important; 
            height: 3rem !important; 
        }
        .stMetric { 
            padding: 0.5rem !important; 
            margin: 0.25rem !important; 
        }
        .element-container { 
            margin-bottom: 1rem !important; 
        }
    }
    
    .buy-signal { 
        border-left: 5px solid #28a745; 
        background-color: rgba(40, 167, 69, 0.1); 
        padding: 1rem; 
        border-radius: 10px; 
        margin: 0.5rem 0; 
        font-size: 0.9rem; 
    }
    
    .sell-signal { 
        border-left: 5px solid #dc3545; 
        background-color: rgba(220, 53, 69, 0.1); 
        padding: 1rem; 
        border-radius: 10px; 
        margin: 0.5rem 0; 
        font-size: 0.9rem; 
    }
    
    .pro-tactical { 
        border-left: 5px solid #007bff; 
        background-color: rgba(0, 123, 255, 0.1); 
        padding: 1rem; 
        border-radius: 10px; 
        margin: 0.5rem 0; 
    }
    
    .analysis-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        margin-bottom: 1rem;
    }
</style>
"""

# Quick pairs for easy access
QUICK_PAIRS = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD"]

# Trading hours information
TRADING_HOURS = {
    "asia": {"open": "00:00", "close": "09:00", "description": "Asian Session"},
    "london": {"open": "07:00", "close": "16:00", "description": "London Session"},
    "new_york": {"open": "13:00", "close": "22:00", "description": "New York Session"},
    "overlap": {"open": "13:00", "close": "16:00", "description": "London/NY Overlap"}
}

# Risk management settings
RISK_SETTINGS = {
    "max_daily_risk": 5.0,  # Maximum daily risk percentage
    "max_trade_risk": 2.0,  # Maximum risk per trade percentage
    "min_confidence": 0.6,  # Minimum confidence for trading
    "max_open_trades": 3,   # Maximum simultaneous trades
}

# Model training settings
MODEL_SETTINGS = {
    "min_data_points": 20,
    "training_split": 0.8,
    "validation_split": 0.2,
    "prediction_horizon": 1,
}

# Feature engineering settings
FEATURE_SETTINGS = {
    "sma_windows": [3, 5, 8],
    "volatility_window": 5,
    "volume_window": 5,
}

# Performance metrics thresholds
PERFORMANCE_THRESHOLDS = {
    "min_accuracy": 0.55,
    "good_accuracy": 0.65,
    "excellent_accuracy": 0.75,
    "min_confidence": 0.6,
    "good_confidence": 0.7,
    "excellent_confidence": 0.8,
}

# API rate limiting settings
RATE_LIMIT_SETTINGS = {
    "requests_per_minute": 60,
    "requests_per_hour": 1000,
    "retry_attempts": 3,
    "timeout_seconds": 30,
}

# Display settings
DISPLAY_SETTINGS = {
    "price_decimals": 5,
    "percentage_decimals": 2,
    "confidence_decimals": 1,
    "pip_decimals": 1,
}

# Error messages
ERROR_MESSAGES = {
    "api_error": "❌ API Error: Failed to fetch market data",
    "network_error": "❌ Network Error: Please check your internet connection",
    "data_error": "❌ Data Error: Insufficient data for analysis",
    "model_error": "❌ Model Error: Failed to train prediction model",
    "symbol_error": "❌ Symbol Error: Invalid forex pair symbol",
    "timeframe_error": "❌ Timeframe Error: Unsupported timeframe",
}

# Success messages
SUCCESS_MESSAGES = {
    "analysis_complete": "✅ Analysis completed successfully",
    "data_fetched": "✅ Market data fetched successfully",
    "model_trained": "✅ AI model trained successfully",
    "prediction_made": "✅ Prediction generated successfully",
}