# ui_components.py
import streamlit as st
from risk_manager import RiskManager, AdvancedRiskManager
import plotly.graph_objects as go
import plotly.express as px

def display_analysis_result(result: dict, account_balance: float, risk_per_trade: float):
    """Display the analysis results with enhanced pattern recognition and regression analysis"""
    
    try:
        # Safely get values with defaults
        prediction = result.get('prediction', 'UNKNOWN')
        symbol = result.get('symbol', 'UNKNOWN')
        timeframe = result.get('timeframe', 'UNKNOWN')
        confidence = result.get('confidence', 0)
        confidence_tier = result.get('confidence_tier', 'unknown')
        market_session = result.get('market_session', 'unknown')
        data_source = result.get('data_source', 'API')
        
        # Main result box
        signal_class = "buy-signal" if prediction == 'BUY' else "sell-signal" if prediction == 'SELL' else "neutral-signal"
        
        with st.container():
            st.markdown(f"""
            <div class="{signal_class}">
                <h3>🎯 {symbol} Analysis ({timeframe}) | {data_source} Data</h3>
                <h4>🤖 AI Recommendation: {prediction}</h4>
                <p>🎯 Confidence: {confidence}% | Tier: {confidence_tier.upper()}</p>
                <p>📊 <strong>ENHANCED PATTERN RECOGNITION</strong> | Session: {market_session.title()}</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Enhanced Technical Analysis Section
        st.subheader("🔍 Advanced Pattern & Regression Analysis")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            signal_score = result.get('signal_score', 0)
            max_score = result.get('max_score', 12)
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
        
        # Pattern Recognition Dashboard
        st.subheader("🎯 Pattern Recognition Dashboard")
        
        pattern_col1, pattern_col2, pattern_col3 = st.columns(3)
        
        with pattern_col1:
            # Three Line Strike
            three_line_strike = result.get('three_line_strike', 0)
            if three_line_strike == 1:
                st.success("✅ Three-Line Strike: BULLISH")
            elif three_line_strike == -1:
                st.error("✅ Three-Line Strike: BEARISH")
            else:
                st.info("➖ Three-Line Strike: No Pattern")
        
        with pattern_col2:
            # Double Patterns
            double_patterns = result.get('double_patterns', {})
            if double_patterns.get('double_top'):
                st.error("⚠️ Double Top Pattern")
            elif double_patterns.get('double_bottom'):
                st.success("⚠️ Double Bottom Pattern")
            else:
                st.info("➖ No Double Pattern")
        
        with pattern_col3:
            # Market Condition
            market_condition = result.get('market_condition', 'unknown')
            condition_display = market_condition.replace('_', ' ').title()
            if 'strong' in market_condition:
                st.success(f"📊 {condition_display}")
            elif 'reversal' in market_condition:
                st.warning(f"📊 {condition_display}")
            else:
                st.info(f"📊 {condition_display}")
        
        # Enhanced Regression Analysis
        st.subheader("📈 Multi-Timeframe Regression Analysis")
        
        enhanced_regression = result.get('enhanced_regression', {})
        if enhanced_regression:
            regression_cols = st.columns(len(enhanced_regression))
            
            for i, (tf_name, reg_data) in enumerate(enhanced_regression.items()):
                with regression_cols[i]:
                    direction = reg_data.get('direction', 'neutral')
                    r_squared = reg_data.get('r_squared', 0)
                    slope = reg_data.get('slope', 0)
                    
                    if direction == 'up':
                        st.success(f"**TF {tf_name.split('_')[-1]}**")
                        st.metric("Trend", "BULLISH", f"R²: {r_squared:.3f}")
                    elif direction == 'down':
                        st.error(f"**TF {tf_name.split('_')[-1]}**")
                        st.metric("Trend", "BEARISH", f"R²: {r_squared:.3f}")
                    else:
                        st.info(f"**TF {tf_name.split('_')[-1]}**")
                        st.metric("Trend", "NEUTRAL", f"R²: {r_squared:.3f}")
        
        # Pattern Alignment
        pattern_alignment = result.get('pattern_alignment', 'No pattern alignment data')
        if "aligns" in pattern_alignment.lower() and "all" in pattern_alignment.lower():
            st.success(f"✅ {pattern_alignment}")
        elif "aligns" in pattern_alignment.lower():
            st.success(f"✅ {pattern_alignment}")
        elif "warning" in pattern_alignment.lower() or "contradicts" in pattern_alignment.lower():
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
        
        # Trading Plan with Advanced Risk Management
        st.subheader("💼 Advanced Trading Plan")
        
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
                - Pattern Confidence: {double_patterns.get('pattern_confidence', 0):.2f}
                """)
            
            with col2:
                st.info(f"""
                **Entry & Exit**
                - Take Profit: {trading_plan.get('take_profit', 'N/A')}
                - Stop Loss: {trading_plan.get('stop_loss', 'N/A')}
                - TP Pips: {trading_plan.get('tp_pips', 'N/A')}
                - SL Pips: {trading_plan.get('sl_pips', 'N/A')}
                - Market Condition: {market_condition.replace('_', ' ').title()}
                - Data Source: {data_source}
                """)
        
        # Advanced Position Sizing Calculator
        with st.expander("🎯 Advanced Position Sizing Calculator"):
            st.subheader("Risk-Adjusted Position Sizing")
            
            # Calculate advanced position size
            sl_pips = trading_plan.get('sl_pips', 10)
            try:
                # Use advanced risk manager with pattern confidence
                pattern_confidence = double_patterns.get('pattern_confidence', 0.5)
                confidence_tier_for_risk = "high" if pattern_confidence > 0.7 else "medium" if pattern_confidence > 0.5 else "low"
                
                position_size, confidence_multiplier = AdvancedRiskManager.calculate_tactical_position_size(
                    account_balance, risk_per_trade, sl_pips, confidence_tier_for_risk
                )
                
                st.info(f"""
                **Advanced Risk Calculation**
                - Account Balance: ${account_balance:,.2f}
                - Risk per Trade: {risk_per_trade}%
                - Stop Loss: {sl_pips} pips
                - Pattern Confidence: {pattern_confidence:.2f}
                - Confidence Multiplier: {confidence_multiplier}x
                - Recommended Position: {position_size:,} units
                - Risk Amount: ${account_balance * (risk_per_trade / 100) * confidence_multiplier:,.2f}
                """)
                
            except Exception as e:
                st.error(f"Error in position calculation: {str(e)}")
        
        # Probability Analysis
        st.subheader("📈 Advanced Probability Analysis")
        
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
            
            # Create probability chart
            fig = go.Figure(data=[
                go.Bar(name='BUY Probability', x=['BUY'], y=[buy_prob], marker_color='green'),
                go.Bar(name='SELL Probability', x=['SELL'], y=[sell_prob], marker_color='red')
            ])
            
            fig.update_layout(
                title='Trading Probability Distribution',
                yaxis_title='Probability (%)',
                showlegend=True,
                height=300
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
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
        st.subheader("📢 Enhanced Trading Signals")
        
        signals = result.get('signals', {})
        active_signals = signals.get('active_signals', [])
        
        if active_signals:
            for signal in active_signals:
                if "BUY" in signal.upper() and "STRONG" in signal.upper():
                    st.success(f"🚀 {signal}")
                elif "SELL" in signal.upper() and "STRONG" in signal.upper():
                    st.error(f"📉 {signal}")
                elif "BUY" in signal.upper():
                    st.success(signal)
                elif "SELL" in signal.upper():
                    st.error(signal)
                elif "UPTREND" in signal.upper() or "BULLISH" in signal.upper():
                    st.success(signal)
                elif "DOWNTREND" in signal.upper() or "BEARISH" in signal.upper():
                    st.error(signal)
                elif "WARNING" in signal.upper() or "CAUTION" in signal.upper():
                    st.warning(signal)
                else:
                    st.info(signal)
            
            st.info(f"**Recommendation:** {signals.get('trading_recommendation', 'No specific recommendation')}")
            st.info(f"**Risk Level:** {signals.get('risk_level', 'Medium Risk')}")
        else:
            st.info("No trading signals generated")
        
        # Enhanced Strategy Components
        with st.expander("🎯 Advanced Strategy Components", expanded=True):
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Linear Regression Trend Analysis**")
                slope = regression_slope
                if slope > 0.001:
                    st.success(f"📈 Strong Uptrend: {slope:.6f}")
                elif slope < -0.001:
                    st.error(f"📉 Strong Downtrend: {slope:.6f}")
                else:
                    st.info(f"➡️ Sideways: {slope:.6f}")
                    
                st.write("**Support/Resistance Levels**")
                st.write(f"�️ Support: {support_level}")
                st.write(f"🏔️ Resistance: {resistance_level}")
                
                # Pattern Summary
                st.write("**Pattern Summary**")
                pattern_count = 0
                if three_line_strike != 0:
                    pattern_count += 1
                if double_patterns.get('double_top') or double_patterns.get('double_bottom'):
                    pattern_count += 1
                st.write(f"🎯 Patterns Detected: {pattern_count}")
                
            with col2:
                st.write("**Pattern Analysis**")
                pattern = three_line_strike
                if pattern == 1:
                    st.success("✅ Bullish Three-Line Strike")
                elif pattern == -1:
                    st.error("✅ Bearish Three-Line Strike")
                else:
                    st.info("➖ No Significant Pattern")
                
                # Double Pattern Details
                if double_patterns:
                    st.write("**Double Pattern Analysis**")
                    if double_patterns.get('double_top'):
                        st.error("🔴 Double Top - Bearish Reversal")
                    if double_patterns.get('double_bottom'):
                        st.success("🟢 Double Bottom - Bullish Reversal")
                    st.write(f"Pattern Confidence: {double_patterns.get('pattern_confidence', 0):.2f}")
                    
                st.write("**Signal Quality**")
                score_ratio = signal_score / max_score if max_score > 0 else 0
                st.progress(score_ratio)
                st.write(f"Score: {signal_score}/{max_score} ({score_ratio:.1%})")
        
        # Auto-Trading Setup
        with st.expander("🤖 Advanced Auto-Trading Setup"):
            try:
                sl_pips = trading_plan.get('sl_pips', 10)
                position_size = RiskManager.calculate_position_size(
                    account_balance, risk_per_trade, sl_pips
                )
                
                # Advanced trading code with pattern conditions
                st.code(f"""
# Advanced Trading Execution with Pattern Recognition
order = {{
    'instrument': '{symbol}',
    'units': {position_size},
    'type': 'MARKET',
    'takeProfit': {trading_plan.get('take_profit', 'N/A')},
    'stopLoss': {trading_plan.get('stop_loss', 'N/A')},
    'conditions': {{
        'pattern_alignment': '{pattern_alignment}',
        'market_condition': '{market_condition}',
        'regression_trend': '{'bullish' if regression_slope > 0 else 'bearish'}',
        'confidence_tier': '{confidence_tier}'
    }}
}}

# Pattern-based Risk Management
risk_adjustment = {{
    'base_risk': {risk_per_trade}%,
    'pattern_multiplier': {double_patterns.get('pattern_confidence', 0.5):.2f},
    'adjusted_risk': {risk_per_trade * double_patterns.get('pattern_confidence', 0.5):.2f}%
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
                st.write(f"**Risk Level:** {result['risk_level']}")
                st.write(f"**Analysis Time:** {result['analysis_timestamp']}")
                st.write(f"**Data Source:** {data_source}")
                st.write(f"**Market Condition:** {market_condition.replace('_', ' ').title()}")
    
    except Exception as e:
        st.error(f"Error displaying analysis results: {str(e)}")
        st.info("Please try the analysis again or check the input parameters.")
    
    # Enhanced Disclaimer
    st.warning("""
    **⚠️ ADVANCED AI TRADING TOOL - EDUCATIONAL PURPOSE ONLY**
    This analysis uses advanced pattern recognition and linear regression algorithms.
    Features include: Double Top/Bottom detection, Three-Line Strike patterns, 
    Multi-timeframe regression analysis, and Market condition classification.
    
    Forex trading involves substantial risk of loss. Past performance does not guarantee future results.
    Pattern recognition and AI predictions are probabilistic in nature.
    """)

def display_pattern_insights(result: dict):
    """Display additional pattern insights and visualizations"""
    try:
        st.subheader("🔍 Pattern Insights & Visualizations")
        
        double_patterns = result.get('double_patterns', {})
        enhanced_regression = result.get('enhanced_regression', {})
        
        # Create pattern strength visualization
        if double_patterns:
            col1, col2 = st.columns(2)
            
            with col1:
                # Pattern confidence gauge
                pattern_confidence = double_patterns.get('pattern_confidence', 0)
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number+delta",
                    value = pattern_confidence * 100,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Pattern Confidence"},
                    gauge = {
                        'axis': {'range': [None, 100]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, 50], 'color': "lightgray"},
                            {'range': [50, 80], 'color': "yellow"},
                            {'range': [80, 100], 'color': "lightgreen"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 90
                        }
                    }
                ))
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Regression strength chart
                if enhanced_regression:
                    timeframes = []
                    r_squared_values = []
                    
                    for tf_name, reg_data in enhanced_regression.items():
                        timeframes.append(f"TF{tf_name.split('_')[-1]}")
                        r_squared_values.append(reg_data.get('r_squared', 0) * 100)
                    
                    fig = px.bar(
                        x=timeframes, 
                        y=r_squared_values,
                        title="Regression Strength by Timeframe",
                        labels={'x': 'Timeframe', 'y': 'R² Strength (%)'},
                        color=r_squared_values,
                        color_continuous_scale='Viridis'
                    )
                    fig.update_layout(height=300)
                    st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error displaying pattern insights: {str(e)}")

def display_market_condition_analysis(result: dict):
    """Display detailed market condition analysis"""
    try:
        st.subheader("🌐 Market Condition Analysis")
        
        market_condition = result.get('market_condition', 'unknown')
        double_patterns = result.get('double_patterns', {})
        
        # Market condition cards
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if 'strong_trend' in market_condition:
                st.success("""
                **📈 STRONG TREND**
                - High momentum
                - Clear direction
                - Lower risk entries
                """)
            else:
                st.info("""
                **📊 MODERATE TREND**
                - Mixed signals
                - Wait for confirmation
                - Medium risk
                """)
        
        with col2:
            if 'consolidation' in market_condition:
                st.warning("""
                **🔄 CONSOLIDATION**
                - Low volatility
                - Range-bound
                - Higher risk
                - Wait for breakout
                """)
            else:
                st.success("""
                **🎯 TRENDING MARKET**
                - Good for trend following
                - Clear support/resistance
                - Favorable for trading
                """)
        
        with col3:
            if 'reversal' in market_condition:
                st.error("""
                **⚠️ POTENTIAL REVERSAL**
                - Pattern signals reversal
                - High caution needed
                - Wait for confirmation
                - Consider smaller position
                """)
            else:
                st.success("""
                **✅ STABLE MARKET**
                - No reversal patterns
                - Continuation likely
                - Normal risk parameters
                """)
        
        # Trading recommendations based on market condition
        st.subheader("🎯 Condition-Based Recommendations")
        
        if 'strong_trend' in market_condition:
            st.success("""
            **RECOMMENDED ACTION:** Proceed with trading plan
            - Use normal position sizing
            - Follow trend direction
            - Set tight stop losses
            - Target trend continuation
            """)
        elif 'consolidation' in market_condition:
            st.warning("""
            **RECOMMENDED ACTION:** Exercise caution
            - Reduce position size by 50%
            - Wait for breakout confirmation
            - Use wider stop losses
            - Consider range trading strategies
            """)
        elif 'reversal' in market_condition:
            st.error("""
            **RECOMMENDED ACTION:** High caution
            - Reduce position size by 70%
            - Wait for strong confirmation
            - Use very tight stop losses
            - Consider waiting for clearer signals
            """)
        else:
            st.info("""
            **RECOMMENDED ACTION:** Standard trading
            - Use normal risk parameters
            - Follow AI signals
            - Monitor market closely
            - Be ready to adjust
            """)
            
    except Exception as e:
        st.error(f"Error displaying market condition analysis: {str(e)}")