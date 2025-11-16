# app.py
import streamlit as st
from datetime import datetime
import time
from config import MOBILE_CSS, ALL_TIMEFRAMES, QUICK_PAIRS, AI_MODEL_SETTINGS
from web_model import WebForexPredictor
from risk_manager import RiskManager, get_market_session, generate_trading_signals, AdvancedRiskManager
from ui_components import display_analysis_result, display_pattern_insights, display_market_condition_analysis
from mt5_integration import mt5_bridge

def initialize_session_state():
    """Initialize session state variables"""
    if 'menu_expanded' not in st.session_state:
        st.session_state.menu_expanded = False
    if 'symbol' not in st.session_state:
        st.session_state.symbol = "EURUSD"
    if 'timeframe' not in st.session_state:
        st.session_state.timeframe = "1h"
    if 'ad_watched' not in st.session_state:
        st.session_state.ad_watched = False
    if 'analysis_complete' not in st.session_state:
        st.session_state.analysis_complete = False
    if 'ad_timer_start' not in st.session_state:
        st.session_state.ad_timer_start = 0
    if 'mt5_connected' not in st.session_state:
        st.session_state.mt5_connected = False
    if 'last_analysis' not in st.session_state:
        st.session_state.last_analysis = None
    if 'use_api_data' not in st.session_state:
        st.session_state.use_api_data = True
    if 'advanced_analysis' not in st.session_state:
        st.session_state.advanced_analysis = True

def show_ads():
    """Display Google AdSense ads and timer"""
    st.markdown("""
    <script async src="https://pagead2.googlesyndication.com/pagead/js/adsbygoogle.js?client=ca-pub-9612311218546127"
        crossorigin="anonymous"></script>
    """, unsafe_allow_html=True)
    
    # Ad container
    st.markdown("""
    <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 2rem; border-radius: 15px; color: white; text-align: center; margin: 1rem 0;'>
        <h3>🎥 Advertisement</h3>
        <p>Please watch this brief ad to unlock AI predictions</p>
        <div style='background: #2d3748; padding: 1rem; border-radius: 10px; margin: 1rem 0;'>
            <p>📺 Ad Content Loading...</p>
            <!-- Google AdSense Ad Slot -->
            <ins class="adsbygoogle"
                style="display:block"
                data-ad-client="ca-pub-9612311218546127"
                data-ad-slot="1234567890"
                data-ad-format="auto"
                data-full-width-responsive="true"></ins>
            <script>
                (adsbygoogle = window.adsbygoogle || []).push({});
            </script>
        </div>
        <p>⏰ Please wait while the ad loads...</p>
    </div>
    """, unsafe_allow_html=True)
    
    return True

def show_ad_timer():
    """Show a countdown timer for ad viewing"""
    if 'ad_timer_start' not in st.session_state:
        st.session_state.ad_timer_start = time.time()
    
    elapsed = time.time() - st.session_state.ad_timer_start
    remaining = max(0, 10 - int(elapsed))  # 10 second ad view
    
    # Timer progress
    progress = elapsed / 10.0
    
    # Timer container
    with st.container():
        st.markdown(f"""
        <div style='background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%); padding: 2rem; border-radius: 15px; color: white; text-align: center; margin: 1rem 0;'>
            <h3>⏳ Ad in Progress</h3>
            <p>Please watch the advertisement to unlock AI predictions</p>
            <div style='background: #2d3748; padding: 1rem; border-radius: 10px; margin: 1rem 0;'>
                <h4>Time remaining: {remaining} seconds</h4>
                <div style='background: #4a5568; border-radius: 10px; height: 20px; margin: 1rem 0;'>
                    <div style='background: #48bb78; height: 100%; width: {progress * 100}%; border-radius: 10px; transition: width 0.3s;'></div>
                </div>
                <p>Thank you for supporting our service! 🙏</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Check if timer is complete
    if elapsed >= 10:
        st.session_state.ad_watched = True
        st.session_state.ad_timer_start = 0
        st.rerun()
    
    return remaining

def perform_enhanced_analysis(symbol: str, timeframe: str, use_api: bool = True) -> dict:
    """Perform enhanced AI analysis with pattern recognition and regression"""
    try:
        # Use the WebForexPredictor from web_model.py
        predictor = WebForexPredictor()
        
        data_source = "API" if use_api else "Synthetic"
        with st.spinner(f"🤖 AI is analyzing {data_source} market data with advanced pattern recognition..."):
            result = predictor.perform_web_analysis(symbol, timeframe, use_api)
        
        if 'error' in result:
            return {'error': result['error']}
        
        if not result.get('success', False):
            return {'error': 'Analysis failed - no results returned'}
        
        # Enhanced prediction handling with pattern confirmation
        prediction = 1 if "BUY" in result['recommendation'] else 0
        confidence = result['confidence']
        confidence_decimal = confidence / 100

        # Enhanced probability handling
        probabilities = result.get('probabilities', {})
        if 'buy_probability' in probabilities and 'sell_probability' in probabilities:
            probabilities = {
                'buy': probabilities['buy_probability'],
                'sell': probabilities['sell_probability']
            }
        else:
            probabilities = {
                'buy': result['confidence'] if prediction == 1 else 100 - result['confidence'],
                'sell': 100 - result['confidence'] if prediction == 1 else result['confidence']
            }
        
        # Enhanced confidence tier with signal score consideration
        signal_score = result.get('signal_score', 0)
        max_score = result.get('max_score', 12)
        score_ratio = signal_score / max_score if max_score > 0 else 0
        
        confidence_tier = "medium"
        num_trades = 3
        if confidence >= 70 and score_ratio >= 0.7:
            confidence_tier = "high"
            num_trades = 1
        elif confidence >= 60 and score_ratio >= 0.5:
            confidence_tier = "medium"
            num_trades = 3
        elif confidence >= 50:
            confidence_tier = "low"
            num_trades = 5
        else:
            confidence_tier = "very_low"
            num_trades = 0
        
        # Calculate TP/SL levels with advanced risk management
        current_price = result['current_price']
        
        # Use advanced risk manager for dynamic pip targets
        advanced_pip_targets = AdvancedRiskManager.get_tactical_pip_targets(
            timeframe, symbol, result
        )
        
        # Enhanced TP/SL calculation
        tp_sl_levels = AdvancedRiskManager.calculate_tactical_tp_sl_levels(
            current_price, prediction, timeframe, symbol, result
        )
        
        # Market context
        market_session = get_market_session()
        
        # Generate enhanced trading signals with pattern info
        signals = generate_trading_signals(prediction, {}, confidence_decimal)
        
        # Add enhanced signals based on pattern analysis
        double_patterns = result.get('double_patterns', {})
        if double_patterns.get('double_top'):
            signals['active_signals'].append("⚠️ DOUBLE TOP Pattern Detected - Potential Bearish Reversal")
        if double_patterns.get('double_bottom'):
            signals['active_signals'].append("⚠️ DOUBLE BOTTOM Pattern Detected - Potential Bullish Reversal")
        
        # Add regression signals
        regression_slope = result.get('regression_slope', 0)
        if regression_slope > 0.001:
            signals['active_signals'].append("📈 STRONG UPTREND - Regression Confirmed")
        elif regression_slope < -0.001:
            signals['active_signals'].append("📉 STRONG DOWNTREND - Regression Confirmed")
        
        # Add pattern signals
        three_line_strike = result.get('three_line_strike', 0)
        if three_line_strike == 1:
            signals['active_signals'].append("🎯 BULLISH Three-Line Strike Pattern")
        elif three_line_strike == -1:
            signals['active_signals'].append("🎯 BEARISH Three-Line Strike Pattern")
        
        # Enhanced regression analysis signals
        enhanced_regression = result.get('enhanced_regression', {})
        regression_signals = []
        for tf, reg_data in enhanced_regression.items():
            if reg_data['r_squared'] > 0.6:
                direction = "up" if reg_data['slope'] > 0 else "down"
                regression_signals.append(f"{tf}: Strong {direction} trend (R²={reg_data['r_squared']:.2f})")
        
        if regression_signals:
            signals['active_signals'].extend(regression_signals)
        
        # Market condition based adjustments
        market_condition = result.get('market_condition', 'unknown')
        if 'consolidation' in market_condition:
            signals['active_signals'].append("🔄 MARKET CONSOLIDATION - Higher Risk")
            signals['risk_level'] = "HIGH RISK"
        elif 'strong_trend' in market_condition:
            signals['active_signals'].append("📈 STRONG TREND - Lower Risk")
            signals['risk_level'] = "LOW RISK"
        
        # Build enhanced final result
        final_result = {
            'success': True,
            'symbol': symbol,
            'timeframe': timeframe,
            'analysis_timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'data_source': data_source,
            
            # AI Prediction
            'prediction': 'BUY' if prediction == 1 else 'SELL',
            'confidence': confidence,
            'confidence_tier': confidence_tier,
            'probabilities': probabilities,
            
            # Trading Plan
            'trading_plan': {
                'recommended_trades': num_trades,
                'pip_target': tp_sl_levels['pip_target'],
                'sl_pips': tp_sl_levels['sl_pips'],
                'tp_pips': tp_sl_levels['tp_pips'],
                'take_profit': tp_sl_levels['take_profit'],
                'stop_loss': tp_sl_levels['stop_loss'],
                'reward_risk_ratio': tp_sl_levels['reward_risk_ratio'],
                'hold_period': tp_sl_levels['hold_period'],
                'description': tp_sl_levels['description']
            },
            
            # Market Context
            'market_session': market_session,
            'current_price': current_price,
            'price_change': f"{result['change_percent']}%",
            
            # Trading Signals
            'signals': signals,
            
            # Enhanced Technical Analysis
            'signal_score': signal_score,
            'max_score': max_score,
            'support_level': result.get('support_level'),
            'resistance_level': result.get('resistance_level'),
            'regression_slope': regression_slope,
            'three_line_strike': three_line_strike,
            'double_patterns': double_patterns,
            'enhanced_regression': enhanced_regression,
            'market_condition': market_condition,
            'pattern_alignment': result.get('pattern_alignment', 'No pattern alignment data'),
            
            # Additional info from web model
            'risk_level': result.get('risk_level', 'Medium Risk'),
            'data_points': result.get('data_points', 0),
            'timeframe_advice': result.get('timeframe_advice', ''),
            'latest_date': result.get('latest_date', ''),
            'color': result.get('color', 'orange'),
            
            # Advanced features
            'advanced_pip_targets': advanced_pip_targets,
            'volatility_adjustment': tp_sl_levels.get('volatility_adjustment', 1.0)
        }
        
        return final_result
        
    except Exception as e:
        return {'error': f'Enhanced analysis failed: {str(e)}'}

def display_mt5_connection_panel():
    """Display MT5 connection panel in sidebar"""
    st.sidebar.markdown("---")
    st.sidebar.subheader("🤖 MT5 Connection")
    
    # Connection status
    if mt5_bridge.connected:
        st.sidebar.success("✅ MT5 Connected")
        
        # Account info
        account_info = mt5_bridge.get_account_info()
        if 'error' not in account_info:
            st.sidebar.info(f"**Account:** {account_info.get('login', 'N/A')}")
            st.sidebar.info(f"**Balance:** ${account_info.get('balance', 0):.2f}")
            st.sidebar.info(f"**Equity:** ${account_info.get('equity', 0):.2f}")
            st.sidebar.info(f"**Free Margin:** ${account_info.get('free_margin', 0):.2f}")
        
        # Open positions
        positions = mt5_bridge.get_open_positions()
        if isinstance(positions, list):
            st.sidebar.info(f"**Open Positions:** {len(positions)}")
            if positions:
                with st.sidebar.expander("View Positions"):
                    for pos in positions:
                        profit_color = "green" if pos['profit'] >= 0 else "red"
                        st.write(f"**{pos['symbol']} {pos['type']}**")
                        st.write(f"Lots: {pos['volume']} | P&L: <span style='color:{profit_color}'>${pos['profit']:.2f}</span>", 
                                unsafe_allow_html=True)
                        if st.button(f"Close {pos['ticket']}", key=f"close_{pos['ticket']}"):
                            close_result = mt5_bridge.close_position(pos['ticket'])
                            if 'error' in close_result:
                                st.error(f"Close failed: {close_result['error']}")
                            else:
                                st.success(f"Position closed! Profit: ${close_result.get('profit', 0):.2f}")
                                st.rerun()
        
        if st.sidebar.button("Disconnect MT5", key="sidebar_disconnect_mt5"):
            mt5_bridge.disconnect_mt5()
            st.session_state.mt5_connected = False
            st.rerun()
            
    else:
        st.sidebar.error("🔴 MT5 Not Connected")
        with st.sidebar.expander("Connect to MT5"):
            st.info("Enter your MT5 connection details:")
            
            server = st.text_input("Server", value="", key="mt5_server_sidebar")
            login = st.number_input("Login", value=0, min_value=0, key="mt5_login_sidebar")
            password = st.text_input("Password", type="password", key="mt5_password_sidebar")
            
            if st.button("Connect to MT5", key="connect_mt5_sidebar"):
                with st.spinner("Connecting to MT5..."):
                    if mt5_bridge.connect_mt5(server=server, login=login, password=password):
                        st.session_state.mt5_connected = True
                        st.success("✅ Successfully connected to MT5!")
                        st.rerun()
                    else:
                        st.error("❌ Failed to connect to MT5")

def main():
    """Main application function"""
    initialize_session_state()
    
    st.set_page_config(
        page_title="Forex Analyzer AI Pro - Advanced Pattern Recognition",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Add Google AdSense to head
    st.markdown("""
    <head>
        <script async src="https://pagead2.googlesyndication.com/pagead/js/adsbygoogle.js?client=ca-pub-9612311218546127"
            crossorigin="anonymous"></script>
    </head>
    """, unsafe_allow_html=True)
    
    st.markdown(MOBILE_CSS, unsafe_allow_html=True)
    st.markdown('<h1 class="main-header">🤖 Forex Analyzer AI Pro - Advanced Pattern Recognition</h1>', unsafe_allow_html=True)
    
    st.success("""
    **🚀 ENHANCED AI ANALYSIS**: Now with Double Top/Bottom Pattern Recognition & Multi-Timeframe Regression
    **📊 ADVANCED FEATURES**: Pattern Database Integration + Linear Regression + Market Condition Classification
    **🎯 SMART PATTERN FILTERING**: Avoids false signals using historical pattern data
    **🤖 MT5 INTEGRATION**: Auto-execution for high-confidence pattern signals
    **📺 Ad-Supported**: Free advanced analysis supported by advertisements
    """)
    
    # Menu toggle and main layout
    col1, col2 = st.columns([3, 1])
    with col2:
        if st.button("☰ Menu", key="menu_toggle", use_container_width=True):
            st.session_state.menu_expanded = not st.session_state.menu_expanded
    
    # Sidebar for expanded menu and MT5 connection
    with st.sidebar:
        st.header("🔧 Advanced Analysis Parameters")
        
        symbol = st.text_input(
            "Forex Pair",
            value=st.session_state.symbol,
            max_chars=6,
            help="Enter 6-character pair (e.g., EURUSD, GBPUSD, USDJPY)",
            key="sidebar_symbol_input"
        ).upper()
        
        timeframe = st.selectbox(
            "Timeframe", 
            options=ALL_TIMEFRAMES, 
            index=ALL_TIMEFRAMES.index(st.session_state.timeframe) if st.session_state.timeframe in ALL_TIMEFRAMES else 3,
            key="sidebar_timeframe_select"
        )
        
        # Data source selection
        use_api_data = st.radio(
            "Data Source",
            ["Live API Data", "Synthetic Data (Demo)"],
            index=0 if st.session_state.use_api_data else 1,
            key="data_source_radio"
        )
        st.session_state.use_api_data = (use_api_data == "Live API Data")
        
        st.subheader("Advanced Account Settings")
        account_balance = st.number_input("Balance ($)", value=10000, min_value=1000, step=1000, key="sidebar_balance_input")
        risk_per_trade = st.slider("Risk (%)", 0.5, 5.0, 2.0, 0.5, key="sidebar_risk_slider")
        
        # Advanced analysis options
        st.subheader("Analysis Options")
        advanced_analysis = st.checkbox(
            "Enable Advanced Pattern Recognition", 
            value=st.session_state.advanced_analysis,
            help="Use pattern database and multi-timeframe regression"
        )
        st.session_state.advanced_analysis = advanced_analysis
        
        st.markdown("---")
        st.success("📱 **Mobile Optimized**")
        st.info("**🎯 Advanced Pattern Recognition**")
        st.warning("**📊 Multi-Timeframe Regression**")
        st.info("**🤖 MT5 Auto-Execution**")
        
        analyze_clicked = st.button("🚀 Analyze Market with Patterns", type="primary", use_container_width=True, key="sidebar_analyze_button")
        
        # Update session state
        st.session_state.symbol = symbol
        st.session_state.timeframe = timeframe
        
        # MT5 Connection Panel
        display_mt5_connection_panel()
    
    # Main content area
    if not st.session_state.get('menu_expanded', False):
        # Quick pairs - Fixed dynamic columns with unique keys
        st.subheader("Quick Pairs")
        
        # Create dynamic columns based on number of pairs
        num_pairs = len(QUICK_PAIRS)
        quick_cols = st.columns(num_pairs)
        
        for i, pair in enumerate(QUICK_PAIRS):
            with quick_cols[i]:
                if st.button(pair, use_container_width=True, key=f"quick_{pair}_{i}"):
                    st.session_state.symbol = pair
                    analyze_clicked = True
        
        # Data source info
        data_source_info = "Live API Data" if st.session_state.use_api_data else "Synthetic Data (Demo)"
        st.info(f"**Current Data Source:** {data_source_info}")
        
        # Default account settings for mobile view
        account_balance = 10000
        risk_per_trade = 2.0
    
    # Perform analysis with ad viewing requirement
    if analyze_clicked:
        symbol = st.session_state.symbol
        timeframe = st.session_state.timeframe
        use_api = st.session_state.use_api_data
        
        if symbol and len(symbol) == 6:
            # Check if user has watched ad
            if not st.session_state.ad_watched:
                st.info("📺 **Ad Viewing Required**")
                st.warning("Please watch a brief advertisement to unlock AI predictions. This helps support our free service.")
                
                # Show ad timer
                remaining = show_ad_timer()
                
                if remaining > 0:
                    st.info(f"⏳ Please wait {remaining} seconds for the ad to complete...")
                    time.sleep(1)
                    st.rerun()
                else:
                    st.success("✅ Ad viewing complete! Generating AI predictions...")
                    st.session_state.ad_watched = True
                    st.rerun()
            else:
                # User has watched ad, proceed with analysis
                result = perform_enhanced_analysis(symbol, timeframe, use_api)
                
                if 'error' in result:
                    st.error(f"❌ {result['error']}")
                    st.info("💡 Please check your internet connection and try again.")
                    # Reset ad watched for next attempt
                    st.session_state.ad_watched = False
                else:
                    # Store last analysis for potential re-use
                    st.session_state.last_analysis = result
                    
                    # Display analysis results
                    display_analysis_result(result, account_balance, risk_per_trade)
                    
                    # Display additional pattern insights
                    display_pattern_insights(result)
                    
                    # Display market condition analysis
                    display_market_condition_analysis(result)
                    
                    # Show thank you message
                    st.success("""
                    **🙏 Thank you for watching the ad!** 
                    Your support helps us maintain and improve this advanced AI trading tool.
                    """)
                    
                    # Reset ad watched for next analysis
                    st.session_state.ad_watched = False
        else:
            st.error("⚠️ Please enter a valid 6-character Forex pair")
    
    # Display last analysis if available (for quick re-runs)
    elif st.session_state.get('last_analysis') and not analyze_clicked:
        st.info("💡 Displaying last analysis results. Click 'Analyze Market with Patterns' for fresh analysis.")
        result = st.session_state.last_analysis
        display_analysis_result(result, account_balance, risk_per_trade)
        display_pattern_insights(result)
        display_market_condition_analysis(result)
    
    # Additional AdSense ads in the footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; margin: 2rem 0;'>
        <h4>Support Our Service</h4>
        <!-- Google AdSense Horizontal Ad -->
        <ins class="adsbygoogle"
            style="display:block"
            data-ad-client="ca-pub-9612311218546127"
            data-ad-slot="0987654321"
            data-ad-format="auto"
            data-full-width-responsive="true"></ins>
        <script>
            (adsbygoogle = window.adsbygoogle || []).push({});
        </script>
    </div>
    """, unsafe_allow_html=True)
    
    # Mobile footer
    st.markdown(
        "<div style='text-align: center; color: #666; font-size: 0.8rem;'>"
        "📱 Mobile Optimized • 🎯 Pattern Recognition • 📊 Regression Analysis • 🤖 MT5 Integration • ⚠️ Educational Purpose Only"
        "</div>", 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"Application error: {str(e)}")
        st.info("Please refresh the page and try again.")
    finally:
        # Ensure MT5 connection is closed properly when the app stops
        if mt5_bridge.connected:
            mt5_bridge.disconnect_mt5()