import streamlit as st
from datetime import datetime
import time
from config import MOBILE_CSS, ALL_TIMEFRAMES, QUICK_PAIRS
from web_model import WebForexPredictor
from risk_manager import RiskManager, get_market_session, generate_trading_signals
from ui_components import display_analysis_result

def initialize_session_state():
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

def perform_enhanced_analysis(symbol: str, timeframe: str) -> dict:
    try:
        # Use the WebForexPredictor from web_model.py
        predictor = WebForexPredictor()
        
        with st.spinner("🤖 AI is analyzing LIVE market data..."):
            result = predictor.perform_web_analysis(symbol, timeframe)
        
        if 'error' in result:
            return {'error': result['error']}
        
        if not result.get('success', False):
            return {'error': 'Analysis failed - no results returned'}
        
        # Convert web model result to our format
        prediction = 1 if "BUY" in result['recommendation'] else 0
        confidence = result['confidence'] / 100

        # Ensure probabilities are properly structured
        probabilities = result.get('probabilities', {})
        if 'buy_probability' in probabilities and 'sell_probability' in probabilities:
            # Convert to standard format
            probabilities = {
                'buy': probabilities['buy_probability'],
                'sell': probabilities['sell_probability']
            }
        else:
            # Fallback probabilities
            probabilities = {
                'buy': result['confidence'] if prediction == 1 else 100 - result['confidence'],
                'sell': 100 - result['confidence'] if prediction == 1 else result['confidence']
            }
        
        # Get confidence tier
        confidence_tier = "medium"
        num_trades = 3
        if confidence >= 0.7:
            confidence_tier = "high"
            num_trades = 1
        elif confidence >= 0.6:
            confidence_tier = "medium"
            num_trades = 3
        elif confidence >= 0.5:
            confidence_tier = "low"
            num_trades = 5
        else:
            confidence_tier = "very_low"
            num_trades = 0
        
        # Calculate TP/SL levels
        current_price = result['current_price']
        tp_sl_levels = RiskManager.calculate_tp_sl_levels(current_price, prediction, timeframe)
        
        # Market context
        market_session = get_market_session()
        
        # Generate trading signals
        signals = generate_trading_signals(prediction, {}, confidence)
        
        # Build final result
        final_result = {
            'success': True,
            'symbol': symbol,
            'timeframe': timeframe,
            'analysis_timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            
            # AI Prediction
            'prediction': 'BUY' if prediction == 1 else 'SELL',
            'confidence': result['confidence'],
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
            
            # Additional info from web model
            'risk_level': result.get('risk_level', 'Medium Risk'),
            'data_points': result.get('data_points', 0),
            'timeframe_advice': result.get('timeframe_advice', ''),
            'latest_date': result.get('latest_date', ''),
            'color': result.get('color', 'orange')
        }
        
        return final_result
        
    except Exception as e:
        return {'error': f'Analysis failed: {str(e)}'}

def main():
    initialize_session_state()
    
    st.set_page_config(
        page_title="Forex Analyzer AI",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    
    # Add Google AdSense to head
    st.markdown("""
    <head>
        <script async src="https://pagead2.googlesyndication.com/pagead/js/adsbygoogle.js?client=ca-pub-9612311218546127"
            crossorigin="anonymous"></script>
    </head>
    """, unsafe_allow_html=True)
    
    st.markdown(MOBILE_CSS, unsafe_allow_html=True)
    st.markdown('<h1 class="main-header">🤖 Forex Analyzer AI</h1>', unsafe_allow_html=True)
    
    st.success("""
    **📊 LIVE MARKET DATA MODE**: Using enhanced AI model with robust data processing.
    **🎯 Professional Tactical Pip Targets**: Dynamic 1:2 risk-reward ratios.
    **🤖 IMPROVED AI**: Better error handling and feature engineering.
    **📺 Ad-Supported**: Free analysis supported by advertisements.
    """)
    
    # Menu toggle
    col1, col2 = st.columns([3, 1])
    with col2:
        if st.button("☰ Menu", key="menu_toggle", use_container_width=True):
            st.session_state.menu_expanded = not st.session_state.menu_expanded
    
    # Sidebar for expanded menu
    if st.session_state.get('menu_expanded', False):
        with st.sidebar:
            st.header("🔧 Analysis Parameters")
            
            symbol = st.text_input(
                "Forex Pair",
                value=st.session_state.symbol,
                max_chars=6,
                help="Enter 6-character pair (e.g., EURUSD, GBPUSD, USDJPY)"
            ).upper()
            
            timeframe = st.selectbox(
                "Timeframe", 
                options=ALL_TIMEFRAMES, 
                index=ALL_TIMEFRAMES.index(st.session_state.timeframe) if st.session_state.timeframe in ALL_TIMEFRAMES else 3
            )
            
            st.subheader("Account Settings")
            account_balance = st.number_input("Balance ($)", value=10000, min_value=1000, step=1000)
            risk_per_trade = st.slider("Risk (%)", 0.5, 5.0, 2.0, 0.5)
            
            st.markdown("---")
            st.success("📱 **Mobile Optimized**")
            st.info("**🎯 Tactical Pip Targets**")
            st.warning("**📺 Ad-Supported Service**")
            
            analyze_clicked = st.button("🚀 Analyze Market", type="primary", use_container_width=True)
            
            # Update session state
            st.session_state.symbol = symbol
            st.session_state.timeframe = timeframe
    else:
        # Main input form
        with st.form("main_analysis_form"):
            col1, col2, col3 = st.columns([2, 2, 1])
            
            with col1:
                symbol = st.text_input(
                    "Forex Pair", 
                    value=st.session_state.symbol, 
                    max_chars=6
                ).upper()
            
            with col2:
                timeframe = st.selectbox(
                    "Timeframe", 
                    options=ALL_TIMEFRAMES, 
                    index=ALL_TIMEFRAMES.index(st.session_state.timeframe) if st.session_state.timeframe in ALL_TIMEFRAMES else 3
                )
            
            with col3:
                st.write("")
                analyze_clicked = st.form_submit_button(
                    "🚀 Analyze", 
                    use_container_width=True
                )
            
            # Update session state
            st.session_state.symbol = symbol
            st.session_state.timeframe = timeframe
        
        # Quick pairs - Fixed dynamic columns
        st.subheader("Quick Pairs")
        
        # Create dynamic columns based on number of pairs
        num_pairs = len(QUICK_PAIRS)
        quick_cols = st.columns(num_pairs)
        
        for i, pair in enumerate(QUICK_PAIRS):
            with quick_cols[i]:
                if st.button(pair, use_container_width=True, key=f"quick_{pair}"):
                    st.session_state.symbol = pair
                    analyze_clicked = True
        
        # Default account settings for mobile view
        account_balance = 10000
        risk_per_trade = 2.0
    
    # Handle analysis trigger from sidebar
    if st.session_state.get('menu_expanded', False):
        if st.sidebar.button("🚀 Analyze Market", type="primary", use_container_width=True):
            analyze_clicked = True
    
    # Perform analysis with ad viewing requirement
    if analyze_clicked:
        symbol = st.session_state.symbol
        timeframe = st.session_state.timeframe
        
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
                result = perform_enhanced_analysis(symbol, timeframe)
                
                if 'error' in result:
                    st.error(f"❌ {result['error']}")
                    st.info("💡 Please check your internet connection and try again.")
                    # Reset ad watched for next attempt
                    st.session_state.ad_watched = False
                else:
                    display_analysis_result(result, account_balance, risk_per_trade)
                    # Show thank you message
                    st.success("""
                    **🙏 Thank you for watching the ad!** 
                    Your support helps us maintain and improve this free AI trading tool.
                    """)
                    
                    # Reset ad watched for next analysis (optional - remove this line if you want one ad per session)
                    st.session_state.ad_watched = False
        else:
            st.error("⚠️ Please enter a valid 6-character Forex pair")
    
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
        "📱 Mobile Optimized • 🎯 Tactical Pip Targets • 🤖 Enhanced AI • 📺 Ad-Supported • ⚠️ Educational Purpose Only"
        "</div>", 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()