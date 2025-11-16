# config.py
# =============================================================================
# ENHANCED CONFIGURATION & CONSTANTS WITH PATTERN RECOGNITION
# =============================================================================

# API Configuration
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

# Enhanced AI Model Settings with Pattern Recognition
AI_MODEL_SETTINGS = {
    "lookback_period": 50,
    "regression_window": 20,
    "min_data_points": 50,
    "prediction_horizon": 3,
    "confidence_threshold_high": 0.7,
    "confidence_threshold_medium": 0.6,
    "confidence_threshold_low": 0.5,
    "signal_score_threshold_high": 8,
    "signal_score_threshold_medium": 6,
    "pattern_confidence_threshold": 0.7,
    "regression_r_squared_threshold": 0.6
}

# Advanced Pattern Recognition Settings
PATTERN_SETTINGS = {
    "three_line_strike": True,
    "double_top_bottom": True,
    "support_resistance_window": 20,
    "regression_channel_window": 20,
    "volatility_adjustment": True,
    "pattern_confidence_weight": 0.3,
    "market_condition_analysis": True,
    "multi_timeframe_regression": True,
    "pattern_alignment_check": True
}

# ENHANCED TACTICAL PIP STRUCTURE - Pattern & Volatility Adjusted
PIP_TARGETS = {
    # Scalping Timeframes - Tighter stops for quick entries
    "1min": {
        "sl_pips": 3, 
        "tp_pips": 6, 
        "description": "Ultra Scalping", 
        "hold_period": "1-5 min",
        "risk_reward": "1:2",
        "volatility_factor": 0.7,
        "max_adjustment": 2.0
    },
    "5min": {
        "sl_pips": 4, 
        "tp_pips": 8, 
        "description": "Momentum Scalping", 
        "hold_period": "10-30 min",
        "risk_reward": "1:2", 
        "volatility_factor": 0.8,
        "max_adjustment": 2.2
    },
    
    # Intraday Timeframes - Balanced risk for day trading
    "15min": {
        "sl_pips": 6, 
        "tp_pips": 12, 
        "description": "Intraday Momentum", 
        "hold_period": "1-4 hours",
        "risk_reward": "1:2",
        "volatility_factor": 0.9,
        "max_adjustment": 2.5
    },
    "30min": {
        "sl_pips": 8, 
        "tp_pips": 16, 
        "description": "Intraday Swing", 
        "hold_period": "2-8 hours",
        "risk_reward": "1:2",
        "volatility_factor": 1.0,
        "max_adjustment": 2.5
    },
    
    # Swing Timeframes - Wider stops for trend following
    "1h": {
        "sl_pips": 10, 
        "tp_pips": 20, 
        "description": "Swing Setup", 
        "hold_period": "6-24 hours",
        "risk_reward": "1:2",
        "volatility_factor": 1.1,
        "max_adjustment": 3.0
    },
    "4h": {
        "sl_pips": 15, 
        "tp_pips": 30, 
        "description": "Swing Trade", 
        "hold_period": "1-3 days",
        "risk_reward": "1:2", 
        "volatility_factor": 1.2,
        "max_adjustment": 3.0
    },
    
    # Position Trading - Maximum room for volatility
    "1day": {
        "sl_pips": 20, 
        "tp_pips": 40, 
        "description": "Position Trade", 
        "hold_period": "3-7 days",
        "risk_reward": "1:2",
        "volatility_factor": 1.3,
        "max_adjustment": 3.5
    },
    "1week": {
        "sl_pips": 30, 
        "tp_pips": 60, 
        "description": "Weekly Position", 
        "hold_period": "1-3 weeks",
        "risk_reward": "1:2",
        "volatility_factor": 1.5,
        "max_adjustment": 4.0
    },
    "1month": {
        "sl_pips": 50, 
        "tp_pips": 100, 
        "description": "Monthly Investment", 
        "hold_period": "1-3 months",
        "risk_reward": "1:2",
        "volatility_factor": 2.0,
        "max_adjustment": 5.0
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

# Enhanced Market Session Multipliers
SESSION_MULTIPLIERS = {
    "asia": 0.8,        # Lower volatility
    "london": 1.2,      # High volatility opening
    "new_york": 1.3,    # Highest volatility
    "overlap": 1.8,     # London/NY overlap - maximum volatility
    "other": 0.9        # Outside main sessions
}

# Enhanced Currency Pair Volatility Adjustments
PAIR_VOLATILITY = {
    "EURUSD": 1.0,      # Baseline
    "GBPUSD": 1.3,      # 30% more volatile
    "USDJPY": 1.2,      # 20% more volatile
    "AUDUSD": 1.4,      # 40% more volatile
    "USDCAD": 1.1,      # 10% more volatile
    "USDCHF": 0.8,      # 20% less volatile
    "NZDUSD": 1.5,      # 50% more volatile
    "EURGBP": 1.2,      # 20% more volatile
    "EURJPY": 1.4,      # 40% more volatile
    "GBPJPY": 1.7,      # 70% more volatile (very volatile)
    "AUDJPY": 1.6,      # 60% more volatile
    "CADJPY": 1.5,      # 50% more volatile
    "default": 1.0
}

# Enhanced Economic Event Multipliers
EVENT_MULTIPLIERS = {
    "normal": 1.0,
    "high_impact": 2.0,    # Increased for major events
    "medium_impact": 1.5,  # Increased for medium events
    "low_impact": 1.2      # Increased for minor events
}

# Advanced Linear Regression Settings
REGRESSION_SETTINGS = {
    "trend_strength_strong": 0.001,
    "trend_strength_weak": 0.0001,
    "r_squared_threshold_high": 0.7,
    "r_squared_threshold_medium": 0.5,
    "r_squared_threshold_low": 0.3,
    "channel_std_dev": 2.0,
    "multi_timeframe_periods": [5, 10, 20, 50],
    "slope_significance": 0.0005
}

# Enhanced Signal Scoring Weights with Pattern Recognition
SIGNAL_WEIGHTS = {
    "three_line_strike": 0.15,          # 15%
    "double_patterns": 0.20,           # 20% for double top/bottom
    "support_resistance": 0.15,         # 15%
    "regression_trend": 0.25,           # 25%
    "market_condition": 0.10,           # 10%
    "volume_confirmation": 0.05,        # 5%
    "pattern_alignment_bonus": 0.10,    # 10% bonus for alignment
    "max_score": 12
}

# Pattern Recognition Settings
PATTERN_RECOGNITION = {
    "double_top_confidence_threshold": 0.6,
    "double_bottom_confidence_threshold": 0.6,
    "three_line_strike_confidence": 0.7,
    "pattern_persistence_periods": 5,
    "min_pattern_formation_bars": 10,
    "pattern_confirmation_required": True
}

# Market Condition Classification
MARKET_CONDITIONS = {
    "strong_trend_threshold": 0.7,
    "consolidation_threshold": 0.3,
    "reversal_confidence_threshold": 0.6,
    "volatility_breakout_threshold": 1.5
}

# CSS Styles for mobile optimization with enhanced pattern classes
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
        .stProgress > div > div { 
            height: 1.5rem !important; 
        }
        .pattern-dashboard { 
            flex-direction: column !important; 
        }
    }
    
    /* Enhanced Signal Display Classes */
    .buy-signal { 
        border-left: 5px solid #28a745; 
        background: linear-gradient(135deg, rgba(40, 167, 69, 0.1) 0%, rgba(40, 167, 69, 0.05) 100%); 
        padding: 1.5rem; 
        border-radius: 15px; 
        margin: 0.5rem 0; 
        font-size: 0.9rem; 
        box-shadow: 0 8px 25px rgba(40, 167, 69, 0.15);
        border: 1px solid rgba(40, 167, 69, 0.3);
    }
    
    .sell-signal { 
        border-left: 5px solid #dc3545; 
        background: linear-gradient(135deg, rgba(220, 53, 69, 0.1) 0%, rgba(220, 53, 69, 0.05) 100%); 
        padding: 1.5rem; 
        border-radius: 15px; 
        margin: 0.5rem 0; 
        font-size: 0.9rem; 
        box-shadow: 0 8px 25px rgba(220, 53, 69, 0.15);
        border: 1px solid rgba(220, 53, 69, 0.3);
    }
    
    .neutral-signal { 
        border-left: 5px solid #6c757d; 
        background: linear-gradient(135deg, rgba(108, 117, 125, 0.1) 0%, rgba(108, 117, 125, 0.05) 100%); 
        padding: 1.5rem; 
        border-radius: 15px; 
        margin: 0.5rem 0; 
        font-size: 0.9rem; 
        box-shadow: 0 8px 25px rgba(108, 117, 125, 0.15);
        border: 1px solid rgba(108, 117, 125, 0.3);
    }
    
    .pro-tactical { 
        border-left: 5px solid #007bff; 
        background: linear-gradient(135deg, rgba(0, 123, 255, 0.1) 0%, rgba(0, 123, 255, 0.05) 100%); 
        padding: 1.5rem; 
        border-radius: 15px; 
        margin: 0.5rem 0; 
        box-shadow: 0 8px 25px rgba(0, 123, 255, 0.15);
        border: 1px solid rgba(0, 123, 255, 0.3);
    }
    
    .pattern-analysis { 
        border-left: 5px solid #ffc107; 
        background: linear-gradient(135deg, rgba(255, 193, 7, 0.1) 0%, rgba(255, 193, 7, 0.05) 100%); 
        padding: 1.5rem; 
        border-radius: 15px; 
        margin: 0.5rem 0; 
        box-shadow: 0 8px 25px rgba(255, 193, 7, 0.15);
        border: 1px solid rgba(255, 193, 7, 0.3);
    }
    
    .analysis-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        margin-bottom: 1rem;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.15);
    }
    
    /* Enhanced Pattern Indicators */
    .pattern-bullish {
        color: #28a745;
        font-weight: bold;
        background: rgba(40, 167, 69, 0.1);
        padding: 0.5rem;
        border-radius: 5px;
        border-left: 3px solid #28a745;
    }
    
    .pattern-bearish {
        color: #dc3545;
        font-weight: bold;
        background: rgba(220, 53, 69, 0.1);
        padding: 0.5rem;
        border-radius: 5px;
        border-left: 3px solid #dc3545;
    }
    
    .pattern-neutral {
        color: #6c757d;
        font-weight: bold;
        background: rgba(108, 117, 125, 0.1);
        padding: 0.5rem;
        border-radius: 5px;
        border-left: 3px solid #6c757d;
    }
    
    /* Enhanced Trend Indicators */
    .trend-strong-up {
        color: #28a745;
        font-weight: bold;
        background: linear-gradient(135deg, rgba(40, 167, 69, 0.2) 0%, rgba(40, 167, 69, 0.1) 100%);
        padding: 0.5rem;
        border-radius: 5px;
    }
    
    .trend-strong-down {
        color: #dc3545;
        font-weight: bold;
        background: linear-gradient(135deg, rgba(220, 53, 69, 0.2) 0%, rgba(220, 53, 69, 0.1) 100%);
        padding: 0.5rem;
        border-radius: 5px;
    }
    
    .trend-sideways {
        color: #6c757d;
        font-weight: bold;
        background: linear-gradient(135deg, rgba(108, 117, 125, 0.2) 0%, rgba(108, 117, 125, 0.1) 100%);
        padding: 0.5rem;
        border-radius: 5px;
    }
    
    /* Enhanced Progress Bars */
    .stProgress > div > div {
        background: linear-gradient(90deg, #4CAF50, #8BC34A);
        border-radius: 10px;
    }
    
    /* Custom Metric Cards */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 15px;
        padding: 1.5rem;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    .pattern-card {
        background: linear-gradient(135deg, #ffc107 0%, #ff8c00 100%);
        border-radius: 15px;
        padding: 1rem;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
    }
    
    .regression-card {
        background: linear-gradient(135deg, #17a2b8 0%, #138496 100%);
        border-radius: 15px;
        padding: 1rem;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
    }
    
    /* Mobile-specific enhancements */
    @media (max-width: 480px) {
        .main-header {
            font-size: 1.5rem !important;
            padding: 0.25rem !important;
        }
        
        .buy-signal, .sell-signal, .neutral-signal {
            padding: 1rem !important;
            margin: 0.25rem 0 !important;
        }
        
        .stMetric {
            padding: 0.25rem !important;
            margin: 0.1rem !important;
        }
        
        .pattern-dashboard {
            grid-template-columns: 1fr !important;
        }
    }
    
    /* Pattern Dashboard Grid */
    .pattern-dashboard {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1rem;
        margin: 1rem 0;
    }
</style>
"""

# Quick pairs for easy access
QUICK_PAIRS = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "USDCHF", "NZDUSD", "EURJPY", "GBPJPY"]

# Enhanced Trading hours information
TRADING_HOURS = {
    "asia": {"open": "00:00", "close": "09:00", "description": "Asian Session - Lower Volatility"},
    "london": {"open": "07:00", "close": "16:00", "description": "London Session - High Volatility"},
    "new_york": {"open": "13:00", "close": "22:00", "description": "New York Session - Highest Volatility"},
    "overlap": {"open": "13:00", "close": "16:00", "description": "London/NY Overlap - Maximum Volatility"},
    "other": {"open": "16:00", "close": "24:00", "description": "Evening Session - Lower Volatility"}
}

# Enhanced Risk management settings with pattern consideration
RISK_SETTINGS = {
    "max_daily_risk": 5.0,           # Maximum daily risk percentage
    "max_trade_risk": 3.0,           # Maximum risk per trade percentage
    "min_confidence": 0.6,           # Minimum confidence for trading
    "min_signal_score": 6,           # Minimum signal score for trading
    "max_open_trades": 3,            # Maximum simultaneous trades
    "leverage_multiplier": 1.0,      # Default leverage
    "emergency_stop_loss": 0.10,     # Emergency stop loss (10%)
    "pattern_confidence_threshold": 0.7,  # Minimum pattern confidence
    "volatility_adjustment_enabled": True,
    "pattern_based_risk_adjustment": True
}

# Enhanced Model training settings
MODEL_SETTINGS = {
    "min_data_points": 50,
    "training_split": 0.8,
    "validation_split": 0.15,
    "test_split": 0.05,
    "prediction_horizon": 1,
    "random_state": 42,
    "pattern_feature_weight": 0.3,
    "regression_feature_weight": 0.4
}

# Enhanced Feature engineering settings
FEATURE_SETTINGS = {
    "sma_windows": [3, 5, 8, 13, 21, 34],
    "ema_windows": [5, 10, 20, 50],
    "volatility_window": [5, 10, 20],
    "volume_window": [5, 10, 20],
    "rsi_period": [7, 14, 21],
    "atr_period": [7, 14],
    "macd_fast": 12,
    "macd_slow": 26,
    "macd_signal": 9,
    "bollinger_period": 20,
    "bollinger_std": 2
}

# Enhanced Performance metrics thresholds
PERFORMANCE_THRESHOLDS = {
    "min_accuracy": 0.55,
    "good_accuracy": 0.65,
    "excellent_accuracy": 0.75,
    "min_confidence": 0.6,
    "good_confidence": 0.7,
    "excellent_confidence": 0.8,
    "min_signal_score": 6,
    "good_signal_score": 8,
    "excellent_signal_score": 10,
    "pattern_confidence_good": 0.7,
    "pattern_confidence_excellent": 0.8
}

# API rate limiting settings
RATE_LIMIT_SETTINGS = {
    "requests_per_minute": 60,
    "requests_per_hour": 1000,
    "retry_attempts": 3,
    "timeout_seconds": 30,
    "backoff_factor": 2.0,
    "concurrent_requests": 5
}

# Enhanced Display settings
DISPLAY_SETTINGS = {
    "price_decimals": 5,
    "percentage_decimals": 2,
    "confidence_decimals": 1,
    "pip_decimals": 1,
    "slope_decimals": 6,
    "volume_decimals": 0,
    "pattern_confidence_decimals": 3,
    "r_squared_decimals": 3
}

# Enhanced Error messages
ERROR_MESSAGES = {
    "api_error": "❌ API Error: Failed to fetch market data from Twelve Data",
    "network_error": "❌ Network Error: Please check your internet connection",
    "data_error": "❌ Data Error: Insufficient data for pattern analysis",
    "model_error": "❌ Model Error: Failed to train prediction model with patterns",
    "symbol_error": "❌ Symbol Error: Invalid forex pair symbol",
    "timeframe_error": "❌ Timeframe Error: Unsupported timeframe",
    "pattern_error": "❌ Pattern Error: Failed to analyze candlestick patterns",
    "regression_error": "❌ Regression Error: Failed to calculate trend analysis",
    "pattern_data_error": "❌ Pattern Data Error: Could not load pattern database"
}

# Enhanced Success messages
SUCCESS_MESSAGES = {
    "analysis_complete": "✅ Enhanced pattern analysis completed successfully",
    "data_fetched": "✅ Market data fetched successfully from Twelve Data API",
    "model_trained": "✅ AI model trained with pattern recognition features",
    "prediction_made": "✅ Enhanced prediction generated with pattern confirmation",
    "pattern_detected": "✅ Candlestick pattern analysis completed with database",
    "regression_analyzed": "✅ Multi-timeframe regression analysis completed",
    "market_condition_analyzed": "✅ Market condition classification completed"
}

# Enhanced Trading recommendation levels with pattern consideration
RECOMMENDATION_LEVELS = {
    "VERY_STRONG_BUY": {
        "min_confidence": 80,
        "min_signal_score": 10,
        "min_pattern_confidence": 0.8,
        "color": "green",
        "emoji": "🚀"
    },
    "STRONG_BUY": {
        "min_confidence": 70,
        "min_signal_score": 8,
        "min_pattern_confidence": 0.7,
        "color": "green",
        "emoji": "📈"
    },
    "MODERATE_BUY": {
        "min_confidence": 60,
        "min_signal_score": 6,
        "min_pattern_confidence": 0.6,
        "color": "orange",
        "emoji": "↗️"
    },
    "WEAK_BUY": {
        "min_confidence": 50,
        "min_signal_score": 4,
        "min_pattern_confidence": 0.5,
        "color": "yellow",
        "emoji": "⬆️"
    },
    "VERY_STRONG_SELL": {
        "min_confidence": 80,
        "min_signal_score": 10,
        "min_pattern_confidence": 0.8,
        "color": "red",
        "emoji": "📉"
    },
    "STRONG_SELL": {
        "min_confidence": 70,
        "min_signal_score": 8,
        "min_pattern_confidence": 0.7,
        "color": "red",
        "emoji": "🔻"
    },
    "MODERATE_SELL": {
        "min_confidence": 60,
        "min_signal_score": 6,
        "min_pattern_confidence": 0.6,
        "color": "orange",
        "emoji": "↘️"
    },
    "WEAK_SELL": {
        "min_confidence": 50,
        "min_signal_score": 4,
        "min_pattern_confidence": 0.5,
        "color": "yellow",
        "emoji": "⬇️"
    },
    "HOLD": {
        "max_confidence": 50,
        "color": "gray",
        "emoji": "⏸️"
    }
}

# Ad settings
AD_SETTINGS = {
    "ad_duration_seconds": 10,
    "ad_client_id": "ca-pub-9612311218546127",
    "ad_slot_main": "1234567890",
    "ad_slot_footer": "0987654321",
    "ad_refresh_minutes": 30,
    "ad_required_for_analysis": True
}

# MetaTrader Integration Settings
MT5_SETTINGS = {
    "enabled": True,
    "server": "",
    "login": 0,
    "password": "",
    "timeout": 10000,
    "portable": False,
    "auto_trading_enabled": False,
    "max_slippage": 20,
    "magic_number": 234000
}

# Enhanced Logging configuration
LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "file": "forex_ai_advanced.log",
    "max_size_mb": 50,
    "backup_count": 10,
    "pattern_analysis_logging": True,
    "regression_analysis_logging": True
}

# Enhanced Backup and recovery settings
BACKUP_SETTINGS = {
    "auto_backup": True,
    "backup_interval_hours": 24,
    "max_backup_files": 10,
    "backup_path": "backups/",
    "pattern_data_backup": True,
    "model_backup": True
}

# Enhanced Real-time update settings
REALTIME_SETTINGS = {
    "update_interval_minutes": 5,
    "market_hours_only": True,
    "auto_refresh": False,
    "pattern_reanalysis_interval": 60,
    "volatility_check_interval": 15
}

# Pattern Database Settings
PATTERN_DATABASE = {
    "enabled": True,
    "csv_files": ["Patterns.csv", "Segmentation.csv", "Meta.csv"],
    "min_pattern_confidence": 0.6,
    "pattern_matching_threshold": 0.7,
    "auto_update_patterns": False
}

# Machine Learning Model Settings
ML_MODEL_SETTINGS = {
    "random_forest_estimators": 100,
    "random_forest_depth": 15,
    "feature_importance_threshold": 0.01,
    "retrain_interval_hours": 24,
    "cross_validation_folds": 5
}