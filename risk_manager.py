import numpy as np
from datetime import datetime
from config import PIP_TARGETS, SESSION_MULTIPLIERS, PAIR_VOLATILITY, EVENT_MULTIPLIERS

class AdvancedRiskManager:
    @staticmethod
    def calculate_volatility_adjustment(data: dict, symbol: str, timeframe: str) -> float:
        """Calculate dynamic volatility adjustment based on recent price action"""
        base_config = PIP_TARGETS[timeframe]
        base_volatility = base_config["volatility_factor"]
        
        # Pair-specific volatility
        pair_multiplier = PAIR_VOLATILITY.get(symbol, PAIR_VOLATILITY["default"])
        
        # Session-based adjustment
        market_session = get_market_session()
        session_multiplier = SESSION_MULTIPLIERS.get(market_session, 1.0)
        
        # ATR-based adjustment (if data available)
        atr_multiplier = AdvancedRiskManager.calculate_atr_multiplier(data, timeframe)
        
        # Combine all adjustments
        final_adjustment = base_volatility * pair_multiplier * session_multiplier * atr_multiplier
        
        return max(0.5, min(2.0, final_adjustment))  # Cap between 0.5x and 2.0x
    
    @staticmethod
    def calculate_atr_multiplier(data: dict, timeframe: str) -> float:
        """Calculate Average True Range multiplier for volatility adjustment"""
        try:
            if 'high' in data and 'low' in data and 'close' in data:
                high = data['high']
                low = data['low']
                close = data['close']
                
                # Calculate True Range
                tr1 = high - low
                tr2 = abs(high - close.shift(1))
                tr3 = abs(low - close.shift(1))
                true_range = np.maximum(tr1, np.maximum(tr2, tr3))
                
                # Calculate ATR (14 period)
                atr = true_range.rolling(window=14).mean().iloc[-1]
                
                # Normalize ATR based on timeframe
                if timeframe in ["1min", "5min"]:
                    normal_atr = 0.0005  # 5 pips normal for scalping
                elif timeframe in ["15min", "30min"]:
                    normal_atr = 0.0010  # 10 pips normal for intraday
                elif timeframe in ["1h", "4h"]:
                    normal_atr = 0.0015  # 15 pips normal for swing
                else:
                    normal_atr = 0.0020  # 20 pips normal for position
                
                atr_ratio = atr / normal_atr
                return max(0.7, min(1.5, atr_ratio))  # Cap ATR adjustment
                
        except Exception:
            pass
        
        return 1.0  # Default no adjustment
    
    @staticmethod
    def get_tactical_pip_targets(timeframe: str, symbol: str, data: dict = None) -> dict:
        """Get dynamically adjusted pip targets for tactical trading"""
        if timeframe not in PIP_TARGETS:
            timeframe = "1h"
        
        base_config = PIP_TARGETS[timeframe]
        
        # Calculate dynamic adjustments
        volatility_adjustment = 1.0
        if data is not None:
            volatility_adjustment = AdvancedRiskManager.calculate_volatility_adjustment(data, symbol, timeframe)
        
        # Apply adjustments to base pip values
        adjusted_sl = base_config["sl_pips"] * volatility_adjustment
        adjusted_tp = base_config["tp_pips"] * volatility_adjustment
        
        return {
            "sl_pips": round(adjusted_sl, 1),
            "tp_pips": round(adjusted_tp, 1),
            "pip_target": round(adjusted_tp, 1),
            "hold_period": base_config["hold_period"],
            "description": base_config["description"],
            "risk_reward": base_config["risk_reward"],
            "volatility_adjustment": round(volatility_adjustment, 2),
            "base_sl": base_config["sl_pips"],
            "base_tp": base_config["tp_pips"]
        }
    
    @staticmethod
    def calculate_tactical_tp_sl_levels(current_price: float, direction: int, timeframe: str, symbol: str, data: dict = None) -> dict:
        """Calculate TP/SL levels with tactical adjustments"""
        pip_config = AdvancedRiskManager.get_tactical_pip_targets(timeframe, symbol, data)
        
        stop_loss_pips = pip_config["sl_pips"]
        take_profit_pips = pip_config["tp_pips"]
        pip_value = 0.0001
        
        if direction == 1:  # BUY
            take_profit_price = current_price + (take_profit_pips * pip_value)
            stop_loss_price = current_price - (stop_loss_pips * pip_value)
        else:  # SELL
            take_profit_price = current_price - (take_profit_pips * pip_value)
            stop_loss_price = current_price + (stop_loss_pips * pip_value)
        
        actual_ratio = take_profit_pips / stop_loss_pips if stop_loss_pips > 0 else 1.0
        
        return {
            'take_profit': round(take_profit_price, 5),
            'stop_loss': round(stop_loss_price, 5),
            'tp_pips': take_profit_pips,
            'sl_pips': stop_loss_pips,
            'reward_risk_ratio': round(actual_ratio, 2),
            'hold_period': pip_config["hold_period"],
            'pip_target': take_profit_pips,
            'description': pip_config["description"],
            'volatility_adjustment': pip_config["volatility_adjustment"],
            'risk_reward': pip_config["risk_reward"],
            'base_sl': pip_config["base_sl"],
            'base_tp': pip_config["base_tp"]
        }
    
    @staticmethod
    def calculate_tactical_position_size(account_balance: float, risk_per_trade: float, stop_loss_pips: float, confidence: float) -> int:
        """Calculate position size with confidence-based adjustments"""
        base_risk_amount = account_balance * (risk_per_trade / 100)
        
        # Confidence-based risk adjustment
        confidence_multiplier = {
            "high": 1.2,    # 20% more on high confidence
            "medium": 1.0,  # Standard risk
            "low": 0.5,     # 50% less on low confidence
            "very_low": 0.2 # 80% less on very low confidence
        }.get(confidence, 1.0)
        
        adjusted_risk = base_risk_amount * confidence_multiplier
        pip_value = 10
        position_size = adjusted_risk / (stop_loss_pips * pip_value)
        
        return int(position_size * 1000), confidence_multiplier

class RiskManager:
    @staticmethod
    def get_pip_targets(timeframe: str) -> dict:
        """Get pip targets based on timeframe"""
        if timeframe not in PIP_TARGETS:
            timeframe = "1h"
        
        config = PIP_TARGETS[timeframe]
        
        return {
            "sl_pips": config["sl_pips"],
            "tp_pips": config["tp_pips"],
            "pip_target": config["tp_pips"],
            "hold_period": config["hold_period"],
            "description": config["description"]
        }
    
    @staticmethod
    def calculate_tp_sl_levels(current_price: float, direction: int, timeframe: str) -> dict:
        """Calculate TP/SL levels with fixed pip values"""
        pip_config = RiskManager.get_pip_targets(timeframe)
        
        stop_loss_pips = pip_config["sl_pips"]
        take_profit_pips = pip_config["tp_pips"]
        pip_value = 0.0001
        
        if direction == 1:  # BUY
            take_profit_price = current_price + (take_profit_pips * pip_value)
            stop_loss_price = current_price - (stop_loss_pips * pip_value)
        else:  # SELL
            take_profit_price = current_price - (take_profit_pips * pip_value)
            stop_loss_price = current_price + (stop_loss_pips * pip_value)
        
        actual_ratio = take_profit_pips / stop_loss_pips if stop_loss_pips > 0 else 1.0
        
        return {
            'take_profit': round(take_profit_price, 5),
            'stop_loss': round(stop_loss_price, 5),
            'tp_pips': take_profit_pips,
            'sl_pips': stop_loss_pips,
            'reward_risk_ratio': round(actual_ratio, 2),
            'hold_period': pip_config["hold_period"],
            'pip_target': take_profit_pips,
            'description': pip_config["description"]
        }
    
    @staticmethod
    def calculate_position_size(account_balance: float, risk_per_trade: float, stop_loss_pips: int) -> int:
        """Calculate position size"""
        risk_amount = account_balance * (risk_per_trade / 100)
        pip_value = 10
        position_size = risk_amount / (stop_loss_pips * pip_value)
        return int(position_size * 1000)

def get_market_session():
    """Get current market session based on UTC time"""
    hour = datetime.utcnow().hour
    if 7 <= hour < 16: 
        return "london"
    elif 13 <= hour < 22: 
        return "new_york" 
    elif 0 <= hour < 9: 
        return "asia"
    else: 
        return "overlap"

def generate_trading_signals(prediction: int, patterns: dict, confidence: float) -> dict:
    """Generate trading signals based on prediction and confidence"""
    signals = []
    if prediction == 1 and confidence > 0.6:
        signals.append("🤖 AI BUY SIGNAL")
    elif prediction == 0 and confidence > 0.6:
        signals.append("🤖 AI SELL SIGNAL")
    
    if confidence >= 0.7:
        recommendation = "HIGH CONFIDENCE - AGGRESSIVE TRADING"
        risk_level = "AGGRESSIVE"
    elif confidence >= 0.6:
        recommendation = "MEDIUM CONFIDENCE - MODERATE TRADING"
        risk_level = "MODERATE"
    elif confidence >= 0.5:
        recommendation = "LOW CONFIDENCE - CONSERVATIVE TRADING"
        risk_level = "CONSERVATIVE"
    else:
        recommendation = "LOW CONFIDENCE - AVOID TRADING"
        risk_level = "AVOID"
    
    return {
        'active_signals': signals,
        'trading_recommendation': recommendation,
        'signal_strength': 'STRONG' if confidence >= 0.7 else 'MODERATE' if confidence >= 0.6 else 'WEAK',
        'risk_level': risk_level
    }