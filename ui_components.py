import streamlit as st
from risk_manager import RiskManager

def display_analysis_result(result: dict, account_balance: float, risk_per_trade: float):
    """Display the analysis results with proper error handling"""
    
    # Main result box
    signal_class = "buy-signal" if result['prediction'] == 'BUY' else "sell-signal"
    
    with st.container():
        st.markdown(f"""
        <div class="{signal_class}">
            <h3>🎯 {result['symbol']} Analysis ({result['timeframe']})</h3>
            <h4>🤖 AI Recommendation: {result['prediction']}</h4>
            <p>🎯 Confidence: {result['confidence']}% | Tier: {result['confidence_tier'].upper()}</p>
            <p>📊 <strong>LIVE MARKET DATA</strong> | Session: {result['market_session'].title()}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Key metrics
    st.subheader("📊 Live Market Stats")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Confidence", f"{result['confidence']}%")
    with col2:
        st.metric("Current Price", f"{result['current_price']}")
    with col3:
        st.metric("Price Change", f"{result['price_change']}")
    with col4:
        st.metric("Market Session", f"{result['market_session'].title()}")
    
    # Trading Plan with Pip Scaler
    st.subheader("💼 Trading Plan")
    
    with st.expander("Trading Details", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            st.info(f"""
            **Position Sizing**
            - Trades: {result['trading_plan']['recommended_trades']}
            - Stop Loss: {result['trading_plan']['sl_pips']} pips
            - Take Profit: {result['trading_plan']['tp_pips']} pips
            - R:R Ratio: {result['trading_plan']['reward_risk_ratio']}:1
            - Hold Period: {result['trading_plan']['hold_period']}
            - Type: {result['trading_plan']['description']}
            """)
        
        with col2:
            st.info(f"""
            **Entry & Exit**
            - Take Profit: {result['trading_plan']['take_profit']}
            - Stop Loss: {result['trading_plan']['stop_loss']}
            - TP Pips: {result['trading_plan']['tp_pips']}
            - SL Pips: {result['trading_plan']['sl_pips']}
            """)
    
    # Probability Analysis with error handling
    st.subheader("📈 Probability Analysis")
    
    try:
        # Handle different probability structures
        if 'probabilities' in result:
            if 'buy' in result['probabilities'] and 'sell' in result['probabilities']:
                # Standard structure
                buy_prob = result['probabilities']['buy']
                sell_prob = result['probabilities']['sell']
            elif 'buy_probability' in result['probabilities'] and 'sell_probability' in result['probabilities']:
                # Alternative structure from web_model
                buy_prob = result['probabilities']['buy_probability']
                sell_prob = result['probabilities']['sell_probability']
            else:
                # Fallback: calculate from confidence
                buy_prob = result['confidence'] if result['prediction'] == 'BUY' else 100 - result['confidence']
                sell_prob = 100 - buy_prob
        else:
            # No probabilities provided, calculate from prediction and confidence
            if result['prediction'] == 'BUY':
                buy_prob = result['confidence']
                sell_prob = 100 - result['confidence']
            else:
                sell_prob = result['confidence']
                buy_prob = 100 - result['confidence']
        
        # Ensure probabilities are valid
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
        # Fallback display
        col1, col2 = st.columns(2)
        with col1:
            st.write("**BUY: N/A**")
            st.progress(0.5)
        with col2:
            st.write("**SELL: N/A**")
            st.progress(0.5)
    
    # Trading Signals
    st.subheader("📢 Trading Signals")
    
    if 'signals' in result:
        for signal in result['signals'].get('active_signals', []):
            if "BUY" in signal:
                st.success(signal)
            elif "SELL" in signal:
                st.error(signal)
            else:
                st.info(signal)
        
        st.info(f"**Recommendation:** {result['signals'].get('trading_recommendation', 'No specific recommendation')}")
    else:
        st.info("No trading signals generated")
    
    # Auto-Trading Setup
    with st.expander("🤖 Auto-Trading Setup"):
        try:
            position_size = RiskManager.calculate_position_size(
                account_balance, risk_per_trade, result['trading_plan']['sl_pips']
            )
            
            st.code(f"""
# OANDA API Execution
order = {{
    'instrument': '{result['symbol']}',
    'units': {position_size},
    'type': 'MARKET',
    'takeProfit': {result['trading_plan']['take_profit']},
    'stopLoss': {result['trading_plan']['stop_loss']}
}}
            """, language='python')
        except Exception as e:
            st.error(f"Error generating auto-trading setup: {str(e)}")
    
    # Pip Scaler Guide
    with st.expander("🎯 Tactical Pip Targets Guide"):
        st.write("""
        **Tactical Pip Targets by Timeframe (1:2 Risk-Reward):**
        • 1min: SL 3 pips / TP 6 pips (1:2) - Ultra Scalping - Hold 1-5 min
        • 5min: SL 4 pips / TP 8 pips (1:2) - Momentum Scalping - Hold 10-30 min
        • 15min: SL 6 pips / TP 12 pips (1:2) - Intraday Momentum - Hold 1-4 hours  
        • 30min: SL 8 pips / TP 16 pips (1:2) - Intraday Swing - Hold 2-8 hours
        • 1h: SL 10 pips / TP 20 pips (1:2) - Swing Setup - Hold 6-24 hours
        • 4h: SL 15 pips / TP 30 pips (1:2) - Swing Trade - Hold 1-3 days
        • 1day: SL 20 pips / TP 40 pips (1:2) - Position Trade - Hold 3-7 days
        • 1week: SL 30 pips / TP 60 pips (1:2) - Weekly Position - Hold 1-3 weeks
        • 1month: SL 50 pips / TP 100 pips (1:2) - Monthly Investment - Hold 1-3 months
        
        **Risk Management:** Professional 1:2 risk-reward ratio with volatility adjustments
        """)
    
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
    
    # Disclaimer
    st.warning("""
    **⚠️ EDUCATIONAL PURPOSE ONLY**
    This analysis uses real market data from Twelve Data API.
    Forex trading involves substantial risk of loss.
    Past performance does not guarantee future results.
    """)