# ui_components.py
import streamlit as st
from risk_manager import RiskManager

def display_analysis_result(result: dict, account_balance: float, risk_per_trade: float):
    """Display the analysis results with proper error handling - Enhanced version"""
    
    try:
        # Safely get values with defaults
        prediction = result.get('prediction', 'UNKNOWN')
        symbol = result.get('symbol', 'UNKNOWN')
        timeframe = result.get('timeframe', 'UNKNOWN')
        confidence = result.get('confidence', 0)
        confidence_tier = result.get('confidence_tier', 'unknown')
        market_session = result.get('market_session', 'unknown')
        
        # Main result box
        signal_class = "buy-signal" if prediction == 'BUY' else "sell-signal" if prediction == 'SELL' else "neutral-signal"
        
        with st.container():
            st.markdown(f"""
            <div class="{signal_class}">
                <h3>🎯 {symbol} Analysis ({timeframe})</h3>
                <h4>🤖 AI Recommendation: {prediction}</h4>
                <p>🎯 Confidence: {confidence}% | Tier: {confidence_tier.upper()}</p>
                <p>📊 <strong>ENHANCED AI ANALYSIS</strong> | Session: {market_session.title()}</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Enhanced Technical Analysis Section
        st.subheader("🔍 Enhanced Technical Analysis")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            signal_score = result.get('signal_score', 0)
            max_score = result.get('max_score', 10)
            st.metric("Signal Score", f"{signal_score}/{max_score}")
        with col2:
            regression_slope = result.get('regression_slope', 0)
            st.metric("Regression Slope", f"{regression_slope:.6f}")
        with col3:
            support_level = result.get('support_level', 'N/A')
            st.metric("Support Level", f"{support_level}")
        with col4:
            resistance_level = result.get('resistance_level', 'N/A')
            st.metric("Resistance Level", f"{resistance_level}")
        
        # Pattern Analysis
        three_line_strike = result.get('three_line_strike', 0)
        if three_line_strike == 1:
            st.success("🎯 Three-Line Strike: BULLISH Pattern Detected")
        elif three_line_strike == -1:
            st.error("🎯 Three-Line Strike: BEARISH Pattern Detected")
        else:
            st.info("🎯 Three-Line Strike: No Pattern Detected")
        
        # Pattern Alignment
        pattern_alignment = result.get('pattern_alignment', 'No alignment data')
        if "aligns" in pattern_alignment.lower():
            st.success(f"✅ {pattern_alignment}")
        elif "contradicts" in pattern_alignment.lower():
            st.warning(f"⚠️ {pattern_alignment}")
        else:
            st.info(f"ℹ️ {pattern_alignment}")
        
        # Key metrics
        st.subheader("📊 Live Market Stats")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Confidence", f"{confidence}%")
        with col2:
            current_price = result.get('current_price', 'N/A')
            st.metric("Current Price", f"{current_price}")
        with col3:
            price_change = result.get('price_change', 'N/A')
            st.metric("Price Change", f"{price_change}")
        with col4:
            st.metric("Market Session", f"{market_session.title()}")
        
        # Trading Plan with Pip Scaler
        st.subheader("💼 Trading Plan")
        
        trading_plan = result.get('trading_plan', {})
        with st.expander("Trading Details", expanded=True):
            col1, col2 = st.columns(2)
            
            with col1:
                st.info(f"""
                **Position Sizing**
                - Trades: {trading_plan.get('recommended_trades', 'N/A')}
                - Stop Loss: {trading_plan.get('sl_pips', 'N/A')} pips
                - Take Profit: {trading_plan.get('tp_pips', 'N/A')} pips
                - R:R Ratio: {trading_plan.get('reward_risk_ratio', 'N/A')}:1
                - Hold Period: {trading_plan.get('hold_period', 'N/A')}
                - Type: {trading_plan.get('description', 'N/A')}
                """)
            
            with col2:
                st.info(f"""
                **Entry & Exit**
                - Take Profit: {trading_plan.get('take_profit', 'N/A')}
                - Stop Loss: {trading_plan.get('stop_loss', 'N/A')}
                - TP Pips: {trading_plan.get('tp_pips', 'N/A')}
                - SL Pips: {trading_plan.get('sl_pips', 'N/A')}
                """)
        
        # Probability Analysis
        st.subheader("📈 Probability Analysis")
        
        try:
            probabilities = result.get('probabilities', {})
            if 'buy' in probabilities and 'sell' in probabilities:
                buy_prob = probabilities['buy']
                sell_prob = probabilities['sell']
            elif 'buy_probability' in probabilities and 'sell_probability' in probabilities:
                buy_prob = probabilities['buy_probability']
                sell_prob = probabilities['sell_probability']
            else:
                buy_prob = confidence if prediction == 'BUY' else 100 - confidence
                sell_prob = 100 - confidence if prediction == 'BUY' else confidence
            
            buy_prob = max(0, min(100, float(buy_prob)))
            sell_prob = max(0, min(100, float(sell_prob)))
            
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**BUY: {buy_prob:.1f}%**")
                st.progress(buy_prob / 100)
            with col2:
                st.write(f"**SELL: {sell_prob:.1f}%**")
                st.progress(sell_prob / 100)
                
        except Exception as e:
            st.warning(f"⚠️ Probability display issue: {str(e)}")
            col1, col2 = st.columns(2)
            with col1:
                st.write("**BUY: N/A**")
                st.progress(0.5)
            with col2:
                st.write("**SELL: N/A**")
                st.progress(0.5)
        
        # Trading Signals
        st.subheader("📢 Trading Signals")
        
        signals = result.get('signals', {})
        active_signals = signals.get('active_signals', [])
        
        if active_signals:
            for signal in active_signals:
                if "BUY" in signal.upper():
                    st.success(signal)
                elif "SELL" in signal.upper():
                    st.error(signal)
                elif "UPTREND" in signal.upper() or "BULLISH" in signal.upper():
                    st.success(signal)
                elif "DOWNTREND" in signal.upper() or "BEARISH" in signal.upper():
                    st.error(signal)
                else:
                    st.info(signal)
            
            st.info(f"**Recommendation:** {signals.get('trading_recommendation', 'No specific recommendation')}")
        else:
            st.info("No trading signals generated")
        
        # Enhanced Strategy Components
        with st.expander("🎯 Advanced Strategy Components", expanded=True):
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Linear Regression Trend**")
                slope = regression_slope
                if slope > 0.001:
                    st.success(f"📈 Strong Uptrend: {slope:.6f}")
                elif slope < -0.001:
                    st.error(f"📉 Strong Downtrend: {slope:.6f}")
                else:
                    st.info(f"➡️ Sideways: {slope:.6f}")
                    
                st.write("**Support/Resistance**")
                st.write(f"Support: {support_level}")
                st.write(f"Resistance: {resistance_level}")
                
            with col2:
                st.write("**Pattern Analysis**")
                pattern = three_line_strike
                if pattern == 1:
                    st.success("✅ Bullish Three-Line Strike")
                elif pattern == -1:
                    st.error("✅ Bearish Three-Line Strike")
                else:
                    st.info("➖ No Significant Pattern")
                    
                st.write("**Signal Quality**")
                score_ratio = signal_score / max_score if max_score > 0 else 0
                st.progress(score_ratio)
                st.write(f"Score: {signal_score}/{max_score}")
        
        # Auto-Trading Setup
        with st.expander("🤖 Auto-Trading Setup"):
            try:
                sl_pips = trading_plan.get('sl_pips', 10)
                position_size = RiskManager.calculate_position_size(
                    account_balance, risk_per_trade, sl_pips
                )
                
                st.code(f"""
# Trading Execution
order = {{
    'instrument': '{symbol}',
    'units': {position_size},
    'type': 'MARKET',
    'takeProfit': {trading_plan.get('take_profit', 'N/A')},
    'stopLoss': {trading_plan.get('stop_loss', 'N/A')}
}}
                """, language='python')
            except Exception as e:
                st.error(f"Error generating auto-trading setup: {str(e)}")
        
        # Additional Information
        if 'timeframe_advice' in result and result['timeframe_advice']:
            with st.expander("📋 Timeframe Strategy Advice"):
                st.info(result['timeframe_advice'])
        
        if 'data_points' in result:
            with st.expander("🔍 Analysis Details"):
                st.write(f"**Data Points Used:** {result['data_points']}")
                if 'risk_level' in result:
                    st.write(f"**Risk Level:** {result['risk_level']}")
                if 'analysis_timestamp' in result:
                    st.write(f"**Analysis Time:** {result['analysis_timestamp']}")
    
    except Exception as e:
        st.error(f"Error displaying analysis results: {str(e)}")
        st.info("Please try the analysis again or check the input parameters.")
    
    # Disclaimer
    st.warning("""
    **⚠️ EDUCATIONAL PURPOSE ONLY**
    This analysis uses real market data from Twelve Data API with advanced AI algorithms.
    Forex trading involves substantial risk of loss.
    Past performance does not guarantee future results.
    """)