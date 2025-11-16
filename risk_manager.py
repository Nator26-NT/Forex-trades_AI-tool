# risk_manager.py
import numpy as np
from datetime import datetime
from config import PIP_TARGETS, SESSION_MULTIPLIERS, PAIR_VOLATILITY, EVENT_MULTIPLIERS, AI_MODEL_SETTINGS

class AdvancedRiskManager:
    @staticmethod
    def calculate_volatility_adjustment(data: dict, symbol: str, timeframe: str) -> float:
        """Calculate dynamic volatility adjustment based on recent price action and patterns"""
        base_config = PIP_TARGETS[timeframe]
        base_volatility = base_config["volatility_factor"]
        
        # Pair-specific volatility
        pair_multiplier = PAIR_VOLATILITY.get(symbol, PAIR_VOLATILITY["default"])
        
        # Session-based adjustment
        market_session = get_market_session()
        session_multiplier = SESSION_MULTIPLIERS.get(market_session, 1.0)
        
        # ATR-based adjustment (if data available)
        atr_multiplier = AdvancedRiskManager.calculate_atr_multiplier(data, timeframe)
        
        # Pattern-based adjustment
        pattern_multiplier = AdvancedRiskManager.calculate_pattern_multiplier(data)
        
        # Market condition adjustment
        market_condition_multiplier = AdvancedRiskManager.calculate_market_condition_multiplier(data)
        
        # Combine all adjustments
        final_adjustment = (base_volatility * pair_multiplier * session_multiplier * 
                          atr_multiplier * pattern_multiplier * market_condition_multiplier)
        
        return max(0.3, min(3.0, final_adjustment))  # Wider cap for more flexibility
    
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
                # More aggressive ATR adjustment
                if atr_ratio > 2.0:
                    return 1.8
                elif atr_ratio > 1.5:
                    return 1.5
                elif atr_ratio > 1.2:
                    return 1.2
                elif atr_ratio < 0.5:
                    return 0.7
                elif atr_ratio < 0.8:
                    return 0.8
                else:
                    return 1.0
                
        except Exception:
            pass
        
        return 1.0  # Default no adjustment
    
    @staticmethod
    def calculate_pattern_multiplier(data: dict) -> float:
        """Calculate risk adjustment based on detected patterns"""
        try:
            double_patterns = data.get('double_patterns', {})
            three_line_strike = data.get('three_line_strike', 0)
            market_condition = data.get('market_condition', 'unknown')
            
            pattern_multiplier = 1.0
            
            # Double pattern adjustments
            if double_patterns.get('double_top') or double_patterns.get('double_bottom'):
                pattern_confidence = double_patterns.get('pattern_confidence', 0.5)
                if pattern_confidence > 0.7:
                    pattern_multiplier *= 0.8  # Reduce risk on high-confidence patterns
                else:
                    pattern_multiplier *= 1.2  # Increase risk on low-confidence patterns
            
            # Three-line strike adjustments
            if three_line_strike != 0:
                pattern_multiplier *= 0.9  # Slightly reduce risk on confirmed patterns
            
            # Market condition adjustments
            if 'strong_trend' in market_condition:
                pattern_multiplier *= 0.8  # Reduce risk in strong trends
            elif 'consolidation' in market_condition:
                pattern_multiplier *= 1.3  # Increase risk in consolidation
            elif 'reversal' in market_condition:
                pattern_multiplier *= 1.5  # Significantly increase risk in reversals
            
            return max(0.5, min(2.0, pattern_multiplier))
            
        except Exception:
            return 1.0
    
    @staticmethod
    def calculate_market_condition_multiplier(data: dict) -> float:
        """Calculate risk adjustment based on overall market condition"""
        try:
            market_condition = data.get('market_condition', 'unknown')
            regression_slope = data.get('regression_slope', 0)
            
            condition_multiplier = 1.0
            
            if 'strong_trend' in market_condition:
                condition_multiplier = 0.7  # Lower risk in strong trends
            elif 'consolidation' in market_condition:
                condition_multiplier = 1.4  # Higher risk in choppy markets
            elif 'reversal' in market_condition:
                condition_multiplier = 1.6  # Highest risk in potential reversals
            
            # Additional adjustment based on regression slope
            if abs(regression_slope) > 0.002:
                condition_multiplier *= 0.8  # Strong trend confirmed
            elif abs(regression_slope) < 0.0001:
                condition_multiplier *= 1.3  # Very weak trend
            
            return max(0.5, min(2.0, condition_multiplier))
            
        except Exception:
            return 1.0
    
    @staticmethod
    def get_tactical_pip_targets(timeframe: str, symbol: str, data: dict = None) -> dict:
        """Get dynamically adjusted pip targets for tactical trading with pattern consideration"""
        if timeframe not in PIP_TARGETS:
            timeframe = "1h"
        
        base_config = PIP_TARGETS[timeframe]
        
        # Calculate dynamic adjustments
        volatility_adjustment = 1.0
        pattern_adjustment = 1.0
        if data is not None:
            volatility_adjustment = AdvancedRiskManager.calculate_volatility_adjustment(data, symbol, timeframe)
            pattern_adjustment = AdvancedRiskManager.calculate_pattern_multiplier(data)
        
        # Apply adjustments to base pip values
        combined_adjustment = volatility_adjustment * pattern_adjustment
        adjusted_sl = base_config["sl_pips"] * combined_adjustment
        adjusted_tp = base_config["tp_pips"] * combined_adjustment
        
        # Ensure minimum pip values
        min_sl = 2 if timeframe in ["1min", "5min"] else 5
        adjusted_sl = max(min_sl, adjusted_sl)
        adjusted_tp = max(adjusted_sl * 1.5, adjusted_tp)  # Ensure minimum 1.5:1 ratio
        
        return {
            "sl_pips": round(adjusted_sl, 1),
            "tp_pips": round(adjusted_tp, 1),
            "pip_target": round(adjusted_tp, 1),
            "hold_period": base_config["hold_period"],
            "description": base_config["description"],
            "risk_reward": base_config["risk_reward"],
            "volatility_adjustment": round(volatility_adjustment, 2),
            "pattern_adjustment": round(pattern_adjustment, 2),
            "combined_adjustment": round(combined_adjustment, 2),
            "base_sl": base_config["sl_pips"],
            "base_tp": base_config["tp_pips"]
        }
    
    @staticmethod
    def calculate_tactical_tp_sl_levels(current_price: float, direction: int, timeframe: str, symbol: str, data: dict = None) -> dict:
        """Calculate TP/SL levels with tactical adjustments including pattern analysis"""
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
            'pattern_adjustment': pip_config["pattern_adjustment"],
            'combined_adjustment': pip_config["combined_adjustment"],
            'risk_reward': pip_config["risk_reward"],
            'base_sl': pip_config["base_sl"],
            'base_tp': pip_config["base_tp"]
        }
    
    @staticmethod
    def calculate_tactical_position_size(account_balance: float, risk_per_trade: float, stop_loss_pips: float, confidence: str) -> tuple:
        """Calculate position size with confidence-based and pattern-based adjustments"""
        base_risk_amount = account_balance * (risk_per_trade / 100)
        
        # Confidence-based risk adjustment
        confidence_multiplier = {
            "high": 1.5,    # 50% more on high confidence
            "medium": 1.0,  # Standard risk
            "low": 0.6,     # 40% less on low confidence
            "very_low": 0.3  # 70% less on very low confidence
        }.get(confidence, 1.0)
        
        # Market condition adjustment
        market_condition_multiplier = 1.0
        # This would typically come from the analysis data
        
        adjusted_risk = base_risk_amount * confidence_multiplier * market_condition_multiplier
        # Enhanced pip value calculation
        pip_value = 10  # Standard lot pip value
        
        # Calculate position size (in units)
        if stop_loss_pips > 0:
            position_size = adjusted_risk / (stop_loss_pips * pip_value * 0.0001)
        else:
            position_size = adjusted_risk / (10 * pip_value * 0.0001)  # Default 10 pip stop
        
        # Convert to standard units (1000 units = 0.01 lot)
        position_units = int(position_size * 1000)
        
        return position_units, confidence_multiplier

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
    """Get current market session based on UTC time with enhanced logic"""
    hour = datetime.utcnow().hour
    minute = datetime.utcnow().minute
    
    # Enhanced session detection
    if 7 <= hour < 16: 
        return "london"
    elif 13 <= hour < 22: 
        return "new_york" 
    elif 0 <= hour < 9: 
        return "asia"
    elif (hour == 12 and minute >= 30) or (hour == 13 and minute <= 30):
        return "overlap"  # London/NY overlap
    else: 
        return "other"

def generate_trading_signals(prediction: int, patterns: dict, confidence: float) -> dict:
    """Generate enhanced trading signals based on prediction, patterns and confidence"""
    signals = []
    
    # Base signals
    if prediction == 1 and confidence > 0.6:
        signals.append("🤖 AI BUY SIGNAL")
    elif prediction == 0 and confidence > 0.6:
        signals.append("🤖 AI SELL SIGNAL")
    
    # Confidence-based recommendations
    if confidence >= 0.8:
        recommendation = "VERY HIGH CONFIDENCE - AGGRESSIVE TRADING"
        risk_level = "VERY AGGRESSIVE"
    elif confidence >= 0.7:
        recommendation = "HIGH CONFIDENCE - AGGRESSIVE TRADING"
        risk_level = "AGGRESSIVE"
    elif confidence >= 0.6:
        recommendation = "MEDIUM CONFIDENCE - MODERATE TRADING"
        risk_level = "MODERATE"
    elif confidence >= 0.5:
        recommendation = "LOW CONFIDENCE - CONSERVATIVE TRADING"
        risk_level = "CONSERVATIVE"
    else:
        recommendation = "VERY LOW CONFIDENCE - AVOID TRADING"
        risk_level = "AVOID"
    
    # Pattern-based signals
    if patterns:
        if patterns.get('double_top'):
            signals.append("⚠️ DOUBLE TOP PATTERN - Bearish Bias")
        if patterns.get('double_bottom'):
            signals.append("⚠️ DOUBLE BOTTOM PATTERN - Bullish Bias")
        if patterns.get('three_line_strike') == 1:
            signals.append("🎯 BULLISH THREE-LINE STRIKE")
        elif patterns.get('three_line_strike') == -1:
            signals.append("🎯 BEARISH THREE-LINE STRIKE")
    
    return {
        'active_signals': signals,
        'trading_recommendation': recommendation,
        'signal_strength': 'VERY STRONG' if confidence >= 0.8 else 'STRONG' if confidence >= 0.7 else 'MODERATE' if confidence >= 0.6 else 'WEAK',
        'risk_level': risk_level
    }