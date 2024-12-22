import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from services.technical_analysis import TechnicalAnalysisService
from services.prediction import PredictionService
from plotly.subplots import make_subplots
import requests
from PIL import Image
from io import BytesIO

# Configure Streamlit theme
st.set_page_config(
    page_title="ItGuess - Smart Stock Analysis",
    page_icon="📈",
    layout="wide"
)

# Custom CSS for light/dark mode compatibility
st.markdown("""
<style>
    /* Light mode styles */
    [data-theme="light"] {
        --text-color: #333333;
        --background-color: #ffffff;
        --card-background: #f8f9fa;
        --border-color: #e0e0e0;
        --hover-color: #f0f0f0;
        --link-color: #1e88e5;
        --gradient-start: #2196f3;
        --gradient-mid: #00bcd4;
        --gradient-end: #1976d2;
    }
    
    /* Dark mode styles */
    [data-theme="dark"] {
        --text-color: #e0e0e0;
        --background-color: #1a1a1a;
        --card-background: #2d2d2d;
        --border-color: #404040;
        --hover-color: #3d3d3d;
        --link-color: #64b5f6;
        --gradient-start: #64b5f6;
        --gradient-mid: #4fc3f7;
        --gradient-end: #2196f3;
    }
    
    /* Common styles */
    .stApp {
        color: var(--text-color);
        background-color: var(--background-color);
    }
    
    .app-title {
        font-family: 'Segoe UI', sans-serif;
        font-size: 5em;
        font-weight: 900;
        text-align: center;
        margin: 40px 0 20px;
        color: var(--text-color);
        text-shadow: 2px 2px 4px rgba(0,0,0,0.15);
        letter-spacing: 4px;
    }
    
    .app-subtitle {
        font-family: 'Segoe UI', sans-serif;
        font-size: 1.4em;
        text-align: center;
        color: var(--text-color);
        margin-bottom: 40px;
        opacity: 0.9;
        font-weight: 500;
    }
    
    /* Typing animation */
    @keyframes typing {
        from { width: 0 }
        to { width: 100% }
    }
    
    @keyframes blink {
        50% { border-color: transparent }
    }
    
    .typing-container {
        display: flex;
        justify-content: center;
        margin: 30px 0;
    }
    
    .typing-text {
        font-family: 'Consolas', monospace;
        font-size: 1.3em;
        color: var(--text-color);
        border-right: 3px solid var(--text-color);
        white-space: nowrap;
        overflow: hidden;
        animation: 
            typing 3.5s steps(40, end),
            blink .75s step-end infinite;
        margin: 0 auto;
        max-width: fit-content;
    }
    
    /* Feature cards with improved styling */
    .features-container {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
        gap: 30px;
        padding: 20px;
        max-width: 1400px;
        margin: 30px auto;
    }
    
    .feature-card {
        background: var(--card-background);
        border-radius: 15px;
        padding: 30px;
        text-align: left;
        transition: all 0.3s ease;
        border: 2px solid var(--border-color);
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .feature-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 20px rgba(0, 0, 0, 0.15);
        border-color: var(--gradient-start);
    }
    
    .feature-icon {
        font-size: 2.5em;
        margin-bottom: 15px;
        display: inline-block;
    }
    
    .feature-title {
        font-size: 1.4em;
        font-weight: 600;
        margin: 10px 0;
        color: var(--text-color);
    }
    
    .feature-description {
        color: var(--text-color);
        line-height: 1.6;
        font-size: 1em;
        opacity: 0.9;
    }
    
    /* Search box with improved styling */
    .search-container {
        display: flex;
        justify-content: center;
        margin: 30px 0;
    }
    
    .search-text {
        font-size: 1.2em;
        color: var(--text-color);
        margin: 0;
    }
    
    .search-prompt {
        text-align: center;
        padding: 30px;
        margin: 20px 0;
        background: linear-gradient(145deg, var(--card-background), var(--hover-color));
        border-radius: 15px;
        border: 1px solid var(--border-color);
    }
    
    /* Hide Streamlit default elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display: none;}
    
    /* Hide any code elements */
    .element-container:has(pre) {display: none;}
    pre {display: none !important;}
    code {display: none !important;}
</style>
""", unsafe_allow_html=True)

# Initialize session state for symbol tracking
if 'last_symbol' not in st.session_state:
    st.session_state.last_symbol = ''
if 'last_update' not in st.session_state:
    st.session_state.last_update = datetime.now()

# Helper functions
@st.cache_data(ttl=3600)
def load_company_logo(symbol):
    try:
        stock = yf.Ticker(symbol)
        if 'logo_url' in stock.info:
            response = requests.get(stock.info['logo_url'])
            img = Image.open(BytesIO(response.content))
            return img
        return None
    except:
        return None

@st.cache_data(ttl=3600)
def get_company_info(symbol):
    try:
        stock = yf.Ticker(symbol)
        info = stock.info
        return {
            'name': info.get('longName', symbol),
            'sector': info.get('sector', 'N/A'),
            'industry': info.get('industry', 'N/A'),
            'website': info.get('website', 'N/A'),
            'description': info.get('longBusinessSummary', 'N/A')
        }
    except:
        return None

def display_company_logo(symbol):
    """Display company logo if available"""
    try:
        # Try to get logo from different sources
        logo_url = None
        
        # Try Wikipedia logo first
        try:
            response = requests.get(f"https://logo.clearbit.com/{symbol.lower()}.com")
            if response.status_code == 200:
                logo_url = f"https://logo.clearbit.com/{symbol.lower()}.com"
        except:
            pass
        
        # Try Clearbit as backup
        if not logo_url:
            try:
                ticker = yf.Ticker(symbol)
                info = ticker.info
                website = info.get('website', '')
                if website:
                    domain = website.replace('http://', '').replace('https://', '').split('/')[0]
                    logo_url = f"https://logo.clearbit.com/{domain}"
            except:
                pass
        
        # Display logo if found
        if logo_url:
            try:
                response = requests.get(logo_url)
                if response.status_code == 200:
                    image = Image.open(BytesIO(response.content))
                    st.image(image, width=100)
                    return True
            except:
                pass
        
        return False
    except Exception as e:
        print(f"Error displaying logo: {str(e)}")
        return False

# Initialize services
def get_services():
    """Initialize all services"""
    technical_analysis = TechnicalAnalysisService()
    prediction_service = PredictionService()
    
    return technical_analysis, prediction_service

# Initialize or refresh services
current_time = datetime.now()
if 'services' not in st.session_state or \
   'last_service_refresh' not in st.session_state or \
   (current_time - st.session_state.last_service_refresh).total_seconds() > 3600:  # Refresh every hour
    print("\nRefreshing services...")
    technical_analysis, prediction_service = get_services()
    st.session_state.services = {
        'technical_analysis': technical_analysis,
        'prediction': prediction_service
    }
    st.session_state.last_service_refresh = current_time
    st.session_state.last_update = current_time

technical_analysis = st.session_state.services['technical_analysis']
prediction_service = st.session_state.services['prediction']

# Validate services
if technical_analysis is None or prediction_service is None:
    st.error("Error initializing services. Please refresh the page.")
    st.stop()

# Sidebar
with st.sidebar:
    st.markdown("<h1 class='app-title'>ItGuess</h1>", unsafe_allow_html=True)
    st.markdown("<p class='app-subtitle'>Smart Stock Analysis & Prediction</p>", unsafe_allow_html=True)
    
    # Theme selector
    if 'theme' not in st.session_state:
        st.session_state.theme = 'Light'
    
    theme = st.selectbox(
        'Choose Theme',
        ['Light', 'Dark'],
        key='theme_select',
        index=['Light', 'Dark'].index(st.session_state.theme)
    )
    
    if theme != st.session_state.theme:
        st.session_state.theme = theme
        set_theme(theme.lower())
    
    # Search box with improved styling
    st.markdown("<div class='search-container'>", unsafe_allow_html=True)
    symbol = st.text_input("Enter stock symbol", placeholder="e.g. AAPL, GOOGL, MSFT")
    st.markdown("</div>", unsafe_allow_html=True)

    # Analysis settings
    with st.expander("Analysis Settings", expanded=True):
        st.subheader("Technical Analysis Settings")
        rsi_period = st.slider("RSI Period", 
                             min_value=1, max_value=21, value=14,
                             help="Relative Strength Index calculation period")
        
        ma_period = st.slider("Moving Average Period",
                            min_value=1, max_value=50, value=20,
                            help="Moving Average calculation period")
    
    # Auto-refresh settings
    with st.expander("Refresh Settings"):
        auto_refresh = st.checkbox("Auto-refresh data (5min)",
                                 help="Automatically refresh data every 5 minutes")
        if auto_refresh and (datetime.now() - st.session_state.last_update).seconds > 300:
            st.session_state.last_update = datetime.now()
            st.rerun()

# Main content
if symbol == "":  # Show welcome page when no symbol is entered
    # Hide default elements
    st.markdown("""
        <style>
            .stMarkdown {display: none;}
            .stMarkdown:first-of-type {display: block;}
            
            /* Sidebar styling */
            .css-1d391kg {
                padding-top: 1rem;
            }
            
            .css-1d391kg h1 {
                margin-bottom: 0.5rem;
            }
        </style>
    """, unsafe_allow_html=True)
    
    # Title and subtitle
    st.markdown('<h1 class="app-title">ItGuess</h1>', unsafe_allow_html=True)
    st.markdown('<p class="app-subtitle">Smart Stock Analysis & Prediction</p>', unsafe_allow_html=True)
    
    # Search prompt with typing animation
    st.markdown("""
        <div class="typing-container">
            <div class="typing-text">Enter a stock symbol to begin (e.g., AAPL for Apple Inc.)</div>
        </div>
    """, unsafe_allow_html=True)
    
    # Feature cards
    st.markdown("""
        <div class="features-container" style="margin-top: 50px;">
            <div class="feature-card">
                <span class="feature-icon">🎯</span>
                <h3 class="feature-title">Technical Analysis</h3>
                <p class="feature-description">
                    Advanced technical indicators including RSI, MACD, and Bollinger Bands for precise market analysis.
                </p>
            </div>
            
            <div class="feature-card">
                <span class="feature-icon">📈</span>
                <h3 class="feature-title">Price Prediction</h3>
                <p class="feature-description">
                    AI-powered price predictions based on technical analysis and market patterns.
                </p>
            </div>
            
            <div class="feature-card">
                <span class="feature-icon">📊</span>
                <h3 class="feature-title">Live Charts</h3>
                <p class="feature-description">
                    Real-time interactive charts with customizable indicators and time periods.
                </p>
            </div>
        </div>
    """, unsafe_allow_html=True)
elif symbol:  # Show stock analysis when symbol is entered
    try:
        # Validate symbol and get info
        stock = yf.Ticker(symbol)
        info = get_company_info(symbol)
        
        if info is None:
            st.error(f"Could not find stock with symbol: {symbol}")
            st.stop()
            
        # Load company info and logo
        with st.spinner("Loading company information..."):
            company_info = get_company_info(symbol)
            company_logo = load_company_logo(symbol)
            
            if not company_info:
                st.warning("Could not load company information, but proceeding with analysis")
                
        # Main header with company info
        col1, col2 = st.columns([1, 3])
        with col1:
            if not display_company_logo(symbol):
                # If logo not found, display a placeholder
                st.markdown("📈")
                
        with col2:
            if company_info:
                st.markdown(f"<h1 class='gradient-text'>{company_info['name']} ({symbol})</h1>", unsafe_allow_html=True)
                st.markdown(f"<p class='company-meta'>{company_info['sector']} | {company_info['industry']}</p>", 
                          unsafe_allow_html=True)
            else:
                st.markdown(f"<h1 class='gradient-text'>{symbol}</h1>", unsafe_allow_html=True)

        # Create tabs with animation
        tab_names = ["Overview", "Technical Analysis", "Price Prediction", "Live Chart"]
        tabs = st.tabs(tab_names)

        with tabs[0]:  # Overview Tab
            try:
                if info is not None:
                    col1, col2 = st.columns([1, 3])
                    
                    with col1:
                        # Display company logo
                        if not display_company_logo(symbol):
                            # If logo not found, display a placeholder
                            st.markdown("📈")
                    
                    with col2:
                        # Display company name as title
                        st.markdown(f"<h1 class='gradient-text'>{info.get('longName', symbol)}</h1>", unsafe_allow_html=True)
                        st.markdown(f"<p><strong>Sector:</strong> {info.get('sector', 'N/A')} | <strong>Industry:</strong> {info.get('industry', 'N/A')}</p>", unsafe_allow_html=True)
                
                # Quick Stats in first row
                st.subheader("Quick Stats")
                try:
                    if info is not None:
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            current_price = stock.history(period='1d')['Close'].iloc[-1]
                            st.metric(
                                "Current Price",
                                f"${current_price:.2f}",
                                f"{price_change_percentage:.2f}%" if 'price_change_percentage' in locals() else None
                            )
                            
                        with col2:
                            market_cap = info.get('marketCap', 0)
                            st.metric(
                                "Market Cap",
                                f"${market_cap/1e9:.2f}B" if market_cap else "N/A"
                            )
                            
                        with col3:
                            volume = info.get('volume', 0)
                            avg_volume = info.get('averageVolume', 0)
                            volume_change = ((volume - avg_volume) / avg_volume * 100) if avg_volume else 0
                            st.metric(
                                "Volume",
                                f"{volume:,}",
                                f"{volume_change:.1f}%" if volume_change else None
                            )
                except Exception as e:
                    st.error(f"Error displaying quick stats: {str(e)}")
                
                # Second row: Key Statistics
                st.subheader("Key Statistics")
                try:
                    if info is not None:
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("P/E Ratio", f"{info.get('trailingPE', 'N/A'):.2f}" if info.get('trailingPE') else "N/A")
                            st.metric("Beta", f"{info.get('beta', 'N/A'):.2f}" if info.get('beta') else "N/A")
                            
                        with col2:
                            st.metric("52 Week High", f"${info.get('fiftyTwoWeekHigh', 'N/A'):.2f}" if info.get('fiftyTwoWeekHigh') else "N/A")
                            st.metric("52 Week Low", f"${info.get('fiftyTwoWeekLow', 'N/A'):.2f}" if info.get('fiftyTwoWeekLow') else "N/A")
                            
                        with col3:
                            dividend_yield = info.get('dividendYield', 0)
                            st.metric("Dividend Yield", f"{dividend_yield*100:.2f}%" if dividend_yield else "N/A")
                            st.metric("Profit Margin", f"{info.get('profitMargins', 0)*100:.2f}%" if info.get('profitMargins') else "N/A")
                except Exception as e:
                    st.error(f"Error displaying key statistics: {str(e)}")

                # Third row: Peer Comparison
                st.subheader("Peer Comparison")
                try:
                    if info is not None and 'recommendationKey' in info:
                        peers = stock.get_info().get('recommendationKey', 'N/A')
                        if peers != 'N/A':
                            st.write(f"Analyst Recommendation: {peers.title()}")
                except Exception as e:
                    st.info("Peer comparison data not available")
                    
            except Exception as e:
                st.error(f"Error in Overview tab: {str(e)}")
        
        with tabs[1]:  # Technical Analysis Tab
            with st.spinner("Calculating technical indicators..."):
                # Get technical analysis data
                analysis_data = technical_analysis.analyze(symbol)
                
                if analysis_data is None:
                    st.error("Unable to perform technical analysis")
                else:
                    # Display technical indicators in columns
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.subheader("MACD")
                        macd_data = analysis_data['macd']
                        st.metric(
                            "MACD",
                            f"{macd_data['macd']:.2f}",
                            f"{macd_data['signal']:.2f}"
                        )
                        st.metric("Signal", macd_data['interpretation'])
                    
                    with col2:
                        st.subheader("Stochastic Oscillator")
                        stoch_data = analysis_data['stochastic']
                        st.metric("K", f"{stoch_data['k']:.2f}")
                        st.metric("D", f"{stoch_data['d']:.2f}")
                        st.metric("Signal", stoch_data['interpretation'])
                    
                    with col3:
                        st.subheader("Volume Analysis")
                        volume_data = analysis_data['volume']
                        st.metric("Volume", f"{volume_data['volume']:,.2f}")
                        st.metric("Signal", volume_data['interpretation'])
                    
                    # Technical Analysis Summary at the bottom
                    st.markdown("### Technical Analysis Summary")
                    
                    # Calculate overall signal
                    buy_signals = sum(1 for x in [macd_data['interpretation'], 
                                                stoch_data['interpretation'], 
                                                volume_data['interpretation']] 
                                    if x == 'Buy')
                    sell_signals = sum(1 for x in [macd_data['interpretation'], 
                                                 stoch_data['interpretation'], 
                                                 volume_data['interpretation']] 
                                     if x == 'Sell')
                    
                    signal_color = "green" if buy_signals > sell_signals else "red"
                    signal_text = "Buy" if buy_signals > sell_signals else "Sell"
                    
                    st.markdown(f"""
                    <div style='text-align: center; padding: 20px; background: white; border-radius: 10px; box-shadow: 0 2px 5px rgba(0,0,0,0.15);'>
                        <h2 style='color: {signal_color};'>{signal_text}</h2>
                        <p style='color: #666;'>↑ {buy_signals} Buy vs ↓ {sell_signals} Sell signals</p>
                    </div>
                    """, unsafe_allow_html=True)

        with tabs[2]:  # Price Prediction Tab
            st.subheader("Price Prediction Analysis")
            
            with st.spinner("Calculating price predictions..."):
                try:
                    # Train models first
                    prediction_service.train_models(symbol)
                    
                    # Get prediction data
                    prediction_data = prediction_service.predict(symbol)
                    
                    if prediction_data:
                        # Plot predicted vs actual prices
                        fig = go.Figure()
                        
                        # Add actual prices
                        fig.add_trace(go.Scatter(
                            x=prediction_data['dates'],
                            y=prediction_data['actual_prices'],
                            name='Actual Price',
                            line=dict(color='blue')
                        ))
                        
                        # Add predicted prices
                        fig.add_trace(go.Scatter(
                            x=prediction_data['dates'],
                            y=prediction_data['predicted_prices'],
                            name='Predicted Price',
                            line=dict(color='red', dash='dash')
                        ))
                        
                        # Add confidence intervals if available
                        if 'upper_bound' in prediction_data and 'lower_bound' in prediction_data:
                            fig.add_trace(go.Scatter(
                                x=prediction_data['dates'] + prediction_data['dates'][::-1],
                                y=prediction_data['upper_bound'] + prediction_data['lower_bound'][::-1],
                                fill='toself',
                                fillcolor='rgba(255,0,0,0.1)',
                                line=dict(color='rgba(255,0,0,0)'),
                                name='95% Confidence Interval'
                            ))
                        
                        fig.update_layout(
                            title=f"{symbol} Price Prediction",
                            xaxis_title="Date",
                            yaxis_title="Price",
                            template='plotly_white',
                            height=400
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Display metrics and predictions in columns
                        col1, col2 = st.columns([1, 1])
                        
                        with col1:
                            # Display prediction metrics
                            st.markdown("### Prediction Metrics")
                            
                            metrics = prediction_data.get('metrics', {})
                            accuracy = metrics.get('accuracy', 0)
                            mse = metrics.get('mse', 0)
                            r2 = metrics.get('r2', 0)
                            
                            st.metric("Model Accuracy", f"{accuracy:.1f}%")
                            st.metric("Mean Squared Error", f"{mse:.4f}")
                            st.metric("R² Score", f"{r2:.4f}")
                            
                            # Feature importance
                            st.markdown("### Feature Importance")
                            feature_importance = prediction_data.get('feature_importance', {})
                            for feature, importance in feature_importance.items():
                                st.metric(feature, f"{importance:.2f}%")
                        
                        with col2:
                            # Future predictions
                            st.markdown("### Future Price Predictions")
                            future_predictions = prediction_data.get('future_predictions', {})
                            
                            current_price = prediction_data['actual_prices'][-1]
                            
                            periods = {
                                '1_day': 'Next Day',
                                '3_days': 'Next 3 Days',
                                '1_week': 'Next Week',
                                '2_weeks': 'Next 2 Weeks',
                                '1_month': 'Next Month',
                                '3_months': 'Next 3 Months',
                                '1_year': 'Next Year'
                            }
                            
                            for key, label in periods.items():
                                if key in future_predictions:
                                    pred = future_predictions[key]
                                    price = pred['price']
                                    confidence = pred['confidence']
                                    change = ((price - current_price) / current_price) * 100
                                    
                                    price_color = "green" if change > 0 else "red"
                                    st.markdown(f"""
                                    <div style='padding: 10px; background: white; border-radius: 5px; margin: 5px 0; box-shadow: 0 1px 3px rgba(0,0,0,0.1);'>
                                        <div style='display: flex; justify-content: space-between; align-items: center;'>
                                            <span style='font-weight: bold;'>{label}</span>
                                            <span style='color: {price_color};'>${price:.2f} ({change:+.1f}%)</span>
                                        </div>
                                        <div style='color: #666; font-size: 0.8em;'>Confidence: {confidence*100:.1f}%</div>
                                    </div>
                                    """, unsafe_allow_html=True)
                    else:
                        st.error("Unable to generate predictions. Please try again later.")
                        
                except Exception as e:
                    st.error(f"Error in price prediction: {str(e)}")
                    import traceback
                    traceback.print_exc()

        with tabs[3]:  # Live Chart Tab
            st.subheader("Live Chart Analysis")
            
            # Time period selection
            col1, col2, col3 = st.columns([2, 2, 1])
            with col1:
                period_options = {
                    '1D': '1d',
                    '5D': '5d',
                    '1M': '1mo',
                    '3M': '3mo',
                    '6M': '6mo',
                    '1Y': '1y',
                    '2Y': '2y',
                    'YTD': 'ytd'
                }
                selected_period = st.select_slider(
                    "Select Time Period",
                    options=list(period_options.keys()),
                    value='3M'
                )
                period = period_options[selected_period]
            
            with col2:
                interval_options = {
                    '1 Minute': '1m',
                    '5 Minutes': '5m',
                    '15 Minutes': '15m',
                    '30 Minutes': '30m',
                    '1 Hour': '1h',
                    'Daily': '1d',
                    'Weekly': '1wk',
                    'Monthly': '1mo'
                }
                selected_interval = st.selectbox(
                    "Select Interval",
                    options=list(interval_options.keys()),
                    index=5  # Default to Daily
                )
                interval = interval_options[selected_interval]
            
            with col3:
                # Theme toggle for the chart
                chart_theme = st.selectbox(
                    "Chart Theme",
                    options=['Light', 'Dark'],
                    index=0
                )
            
            # Get stock data
            with st.spinner("Loading chart data..."):
                try:
                    stock = yf.Ticker(symbol)
                    data = stock.history(period=period, interval=interval)
                    
                    if not data.empty:
                        # Technical indicators selection
                        st.markdown("### Technical Indicators")
                        indicator_cols = st.columns(4)
                        
                        with indicator_cols[0]:
                            show_ma = st.checkbox("Moving Averages", value=True)
                            if show_ma:
                                ma_periods = st.multiselect(
                                    "MA Periods",
                                    options=[5, 10, 20, 50, 100, 200],
                                    default=[20, 50]
                                )
                        
                        with indicator_cols[1]:
                            show_bb = st.checkbox("Bollinger Bands")
                            if show_bb:
                                bb_period = st.number_input("BB Period", value=20, min_value=5, max_value=50)
                                bb_std = st.number_input("BB Std Dev", value=2, min_value=1, max_value=4)
                        
                        with indicator_cols[2]:
                            show_rsi = st.checkbox("RSI")
                            if show_rsi:
                                rsi_period = st.number_input("RSI Period", value=14, min_value=5, max_value=30)
                        
                        with indicator_cols[3]:
                            show_volume = st.checkbox("Volume", value=True)
                        
                        # Create the main chart
                        fig = make_subplots(
                            rows=2 if show_volume else 1,
                            cols=1,
                            shared_xaxes=True,
                            vertical_spacing=0.05,
                            row_heights=[0.7, 0.3] if show_volume else [1]
                        )
                        
                        # Add candlestick chart
                        fig.add_trace(
                            go.Candlestick(
                                x=data.index,
                                open=data['Open'],
                                high=data['High'],
                                low=data['Low'],
                                close=data['Close'],
                                name='Price'
                            ),
                            row=1, col=1
                        )
                        
                        # Add Moving Averages
                        if show_ma:
                            for period in ma_periods:
                                ma = data['Close'].rolling(window=period).mean()
                                fig.add_trace(
                                    go.Scatter(
                                        x=data.index,
                                        y=ma,
                                        name=f'MA{period}',
                                        line=dict(width=1)
                                    ),
                                    row=1, col=1
                                )
                        
                        # Add Bollinger Bands
                        if show_bb:
                            bb_ma = data['Close'].rolling(window=bb_period).mean()
                            bb_std = data['Close'].rolling(window=bb_period).std()
                            
                            fig.add_trace(
                                go.Scatter(
                                    x=data.index,
                                    y=bb_ma + (bb_std * 2),
                                    name='BB Upper',
                                    line=dict(dash='dash', width=1)
                                ),
                                row=1, col=1
                            )
                            
                            fig.add_trace(
                                go.Scatter(
                                    x=data.index,
                                    y=bb_ma - (bb_std * 2),
                                    name='BB Lower',
                                    line=dict(dash='dash', width=1),
                                    fill='tonexty'
                                ),
                                row=1, col=1
                            )
                        
                        # Add RSI
                        if show_rsi:
                            delta = data['Close'].diff()
                            gain = (delta.where(delta > 0, 0)).rolling(window=rsi_period).mean()
                            loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period).mean()
                            rs = gain / loss
                            rsi = 100 - (100 / (1 + rs))
                            
                            fig.add_trace(
                                go.Scatter(
                                    x=data.index,
                                    y=rsi,
                                    name='RSI',
                                    line=dict(color='purple')
                                ),
                                row=1, col=1
                            )
                            
                            # Add RSI reference lines
                            fig.add_hline(y=70, line_dash="dash", line_color="red", row=1)
                            fig.add_hline(y=30, line_dash="dash", line_color="green", row=1)
                        
                        # Add Volume
                        if show_volume:
                            colors = ['red' if close < open else 'green'
                                    for close, open in zip(data['Close'], data['Open'])]
                            
                            fig.add_trace(
                                go.Bar(
                                    x=data.index,
                                    y=data['Volume'],
                                    name='Volume',
                                    marker_color=colors
                                ),
                                row=2, col=1
                            )
                        
                        # Update layout
                        fig.update_layout(
                            title=f"{symbol} Live Chart",
                            xaxis_title="Date",
                            yaxis_title="Price",
                            height=800,
                            template='plotly_white' if chart_theme == 'Light' else 'plotly_dark',
                            showlegend=True,
                            legend=dict(
                                yanchor="top",
                                y=0.99,
                                xanchor="left",
                                x=0.01
                            ),
                            margin=dict(l=0, r=0, t=30, b=0)
                        )
                        
                        # Add range slider
                        fig.update_xaxes(rangeslider_visible=True)
                        
                        # Add drawing tools
                        config = {
                            'modeBarButtonsToAdd': [
                                'drawline',
                                'drawopenpath',
                                'drawclosedpath',
                                'drawcircle',
                                'drawrect',
                                'eraseshape'
                            ]
                        }
                        
                        # Display the chart
                        st.plotly_chart(fig, use_container_width=True, config=config)
                        
                        # Quick Stats
                        stats_cols = st.columns(4)
                        
                        with stats_cols[0]:
                            current_price = data['Close'].iloc[-1]
                            prev_close = data['Close'].iloc[-2]
                            price_change = current_price - prev_close
                            price_change_pct = (price_change / prev_close) * 100
                            
                            st.metric(
                                "Current Price",
                                f"${current_price:.2f}",
                                f"{price_change_pct:+.2f}%"
                            )
                        
                        with stats_cols[1]:
                            high = data['High'].iloc[-1]
                            low = data['Low'].iloc[-1]
                            st.metric("Day Range", f"${low:.2f} - ${high:.2f}")
                        
                        with stats_cols[2]:
                            volume = data['Volume'].iloc[-1]
                            avg_volume = data['Volume'].mean()
                            volume_change = ((volume - avg_volume) / avg_volume) * 100
                            st.metric(
                                "Volume",
                                f"{volume:,.0f}",
                                f"{volume_change:+.1f}% vs Avg"
                            )
                        
                        with stats_cols[3]:
                            volatility = data['Close'].pct_change().std() * np.sqrt(252) * 100
                            st.metric("Volatility", f"{volatility:.1f}%")
                        
                    else:
                        st.warning("No data available for the selected period")
                        
                except Exception as e:
                    st.error(f"Error loading chart: {str(e)}")
    
    except Exception as e:
        st.error(f"Error fetching data: {str(e)}")
else:
    st.info("Please enter a valid stock symbol. Example: AAPL for Apple Inc.")
