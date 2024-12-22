import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from services.technical_analysis import TechnicalAnalysisService
from services.prediction import PredictionService
from services.news_service import NewsService
from plotly.subplots import make_subplots
import requests
from PIL import Image
from io import BytesIO

# Page config
st.set_page_config(
    page_title="ItGuess - Stock Analysis",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern design
st.markdown("""
    <style>
        /* Global Styles */
        .stApp {
            background-color: #ffffff;
        }
        
        /* Header styling */
        h1 {
            font-family: 'SF Pro Display', -apple-system, BlinkMacSystemFont, sans-serif;
            font-weight: 800;
            font-size: 2.5rem;
            background: linear-gradient(120deg, #FF4B4B, #7E56DA);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin: 1rem 0;
        }
        
        /* Subheader styling */
        h2, h3 {
            color: #0F1642;
            font-family: 'SF Pro Display', -apple-system, BlinkMacSystemFont, sans-serif;
            font-weight: 600;
        }
        
        /* Card styling */
        div.stMetric {
            background: linear-gradient(135deg, rgba(255, 255, 255, 0.8), rgba(255, 255, 255, 0.4));
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.2);
            border-radius: 16px;
            padding: 1.5rem;
            box-shadow: 0 8px 32px rgba(31, 38, 135, 0.1);
        }
        
        /* Metrics styling */
        div[data-testid="stMetricValue"] {
            font-size: 2rem !important;
            font-weight: 700 !important;
            color: #0F1642 !important;
        }
        
        div[data-testid="stMetricLabel"] {
            font-size: 1rem !important;
            color: #666666 !important;
        }
        
        /* Sidebar styling */
        section[data-testid="stSidebar"] {
            background-color: #f8f9fa;
            padding: 2rem 1rem;
        }
        
        section[data-testid="stSidebar"] .stTextInput input {
            border: 2px solid #e9ecef;
            border-radius: 12px;
            padding: 1rem;
            font-size: 1rem;
            transition: all 0.3s ease;
        }
        
        section[data-testid="stSidebar"] .stTextInput input:focus {
            border-color: #7E56DA;
            box-shadow: 0 0 0 3px rgba(126, 86, 218, 0.1);
        }
        
        /* Button styling */
        .stButton button {
            background: linear-gradient(135deg, #FF4B4B, #7E56DA);
            color: white;
            border: none;
            padding: 0.75rem 1.5rem;
            border-radius: 12px;
            font-weight: 600;
            transition: all 0.3s ease;
        }
        
        .stButton button:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(126, 86, 218, 0.2);
        }
        
        /* Tabs styling */
        .stTabs [data-baseweb="tab-list"] {
            gap: 1rem;
            background-color: transparent;
        }
        
        .stTabs [data-baseweb="tab"] {
            background-color: transparent;
            border: none;
            color: #666666;
            font-weight: 600;
            padding: 1rem 1.5rem;
            border-radius: 12px;
            transition: all 0.3s ease;
        }
        
        .stTabs [data-baseweb="tab"][aria-selected="true"] {
            background: linear-gradient(135deg, #FF4B4B20, #7E56DA20);
            color: #7E56DA;
        }
        
        /* Plot styling */
        .js-plotly-plot {
            border-radius: 16px;
            padding: 1rem;
            background: linear-gradient(135deg, rgba(255, 255, 255, 0.8), rgba(255, 255, 255, 0.4));
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.2);
            box-shadow: 0 8px 32px rgba(31, 38, 135, 0.1);
        }
        
        /* Search box styling */
        .search-container {
            position: relative;
            margin: 2rem 0;
        }
        
        .search-container input {
            width: 100%;
            padding: 1rem 1.5rem;
            font-size: 1.1rem;
            border: 2px solid #e9ecef;
            border-radius: 12px;
            transition: all 0.3s ease;
        }
        
        .search-container input:focus {
            border-color: #7E56DA;
            box-shadow: 0 0 0 3px rgba(126, 86, 218, 0.1);
            outline: none;
        }
        
        /* Custom gradients */
        .gradient-text {
            background: linear-gradient(120deg, #FF4B4B, #7E56DA);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        
        .gradient-bg {
            background: linear-gradient(135deg, #FF4B4B20, #7E56DA20);
        }
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

# Initialize services
def get_services():
    print("\nInitializing services...")
    try:
        ta_service = TechnicalAnalysisService()
        print("Technical Analysis Service initialized")
        
        pred_service = PredictionService()
        print("Prediction Service initialized")
        
        news_service = NewsService()
        print("News Service initialized")
        
        return ta_service, pred_service, news_service
    except Exception as e:
        print(f"Error initializing services: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None, None

# Initialize or refresh services
current_time = datetime.now()
if 'services' not in st.session_state or \
   'last_service_refresh' not in st.session_state or \
   (current_time - st.session_state.last_service_refresh).total_seconds() > 3600:  # Refresh every hour
    print("\nRefreshing services...")
    st.session_state.services = get_services()
    st.session_state.last_service_refresh = current_time
    st.session_state.last_update = current_time

technical_analysis, prediction_service, news_service = st.session_state.services

# Validate services
if None in st.session_state.services:
    st.error("Error initializing services. Please refresh the page.")
    st.stop()

# Sidebar
with st.sidebar:
    st.markdown("<h1 class='gradient-text' style='text-align: center;'>ItGuess</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #666666; margin-bottom: 2rem;'>Smart Stock Analysis</p>", unsafe_allow_html=True)
    
    # Search box
    st.markdown("<div class='search-container'>", unsafe_allow_html=True)
    symbol = st.text_input("Enter stock symbol", placeholder="e.g. AAPL, GOOGL, MSFT")
    st.markdown("</div>", unsafe_allow_html=True)

    # Analysis settings
    with st.expander("Analysis Settings", expanded=True):
        st.subheader("Prediction Settings")
        period = st.selectbox("Prediction Period", 
                            ["1d", "3d", "1w", "2w", "1m"],
                            help="Select the time period for price prediction")
        
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
if symbol:
    try:
        # Validate symbol
        stock = yf.Ticker(symbol)
        
        # Try to fetch some basic data first
        try:
            hist = stock.history(period="1d")
            if hist.empty:
                st.error(f"No trading data available for symbol: {symbol}")
                st.stop()
        except Exception as e:
            st.error(f"Error fetching data for {symbol}: {str(e)}")
            st.stop()
            
        # Now try to get company info
        try:
            info = stock.info
            if not info:
                st.error(f"No company information available for symbol: {symbol}")
                st.stop()
        except Exception as e:
            st.error(f"Error fetching company info for {symbol}: {str(e)}")
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
            if company_logo:
                st.image(company_logo, width=100)
            else:
                st.markdown(f"<div class='company-symbol'>{symbol}</div>", unsafe_allow_html=True)
                
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
            with st.container():
                # Quick Stats in first row
                st.subheader("Quick Stats")
                try:
                    stock = yf.Ticker(symbol)
                    info = stock.info
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        market_cap = info.get('marketCap', 0)
                        market_cap_str = f"${market_cap/1e9:.2f}B" if market_cap >= 1e9 else f"${market_cap/1e6:.2f}M"
                        st.metric("Market Cap", market_cap_str)
                    
                    with col2:
                        pe_ratio = info.get('trailingPE', 'N/A')
                        pe_ratio = f"{pe_ratio:.2f}" if isinstance(pe_ratio, (int, float)) else pe_ratio
                        st.metric("P/E Ratio", pe_ratio)
                    
                    with col3:
                        high_52w = info.get('fiftyTwoWeekHigh', 0)
                        low_52w = info.get('fiftyTwoWeekLow', 0)
                        if high_52w and low_52w:
                            range_52w = f"${low_52w:.2f} - ${high_52w:.2f}"
                        else:
                            range_52w = "N/A"
                        st.metric("52W Range", range_52w)
                    
                    with col4:
                        volume = info.get('volume', 0)
                        volume_str = f"{volume/1e6:.1f}M" if volume >= 1e6 else f"{volume/1e3:.1f}K"
                        st.metric("Volume", volume_str)
                
                except Exception as e:
                    st.error(f"Error loading stock information: {str(e)}")

                # Second row: Price Movement and Peer Comparison
                col1, col2 = st.columns([3, 2])
                
                with col1:
                    st.subheader("Price Movement")
                    try:
                        hist = stock.history(period="6mo")
                        fig = go.Figure(data=[
                            go.Candlestick(
                                x=hist.index,
                                open=hist['Open'],
                                high=hist['High'],
                                low=hist['Low'],
                                close=hist['Close'],
                                name='Price'
                            )
                        ])
                        
                        fig.update_layout(
                            title=f"{symbol} 6-Month Price Movement",
                            yaxis_title="Price",
                            xaxis_title="Date",
                            height=400,
                            template='plotly_white'
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                    except Exception as e:
                        st.error(f"Error creating price movement chart: {str(e)}")
                
                with col2:
                    st.subheader("Peer Comparison")
                    peer_data = news_service.get_peer_comparison(symbol)
                    if peer_data is not None:
                        # Create radar chart for peer comparison
                        metrics = ['pe_ratio', 'price_to_sales', 'price_to_book', 'debt_to_equity']
                        fig = go.Figure()
                        
                        for idx, row in peer_data.iterrows():
                            fig.add_trace(go.Scatterpolar(
                                r=[row[m] for m in metrics],
                                theta=metrics,
                                fill='toself',
                                name=row['symbol']
                            ))
                            
                        fig.update_layout(
                            polar=dict(
                                radialaxis=dict(
                                    visible=True,
                                    range=[0, peer_data[metrics].max().max()]
                                )),
                            showlegend=True,
                            height=400
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("Peer comparison data not available")

                # Third row: News Sentiment
                st.subheader("News Sentiment Analysis")
                news_df = news_service.get_company_news(symbol, company_info['name'] if company_info else symbol)
                
                if news_df is not None and not news_df.empty:
                    # Sentiment distribution
                    sentiment_counts = news_df['sentiment_category'].value_counts()
                    
                    col1, col2 = st.columns([1, 2])
                    
                    with col1:
                        # Sentiment donut chart
                        colors = {'Positive': 'green', 'Neutral': 'gray', 'Negative': 'red'}
                        fig = go.Figure(data=[go.Pie(
                            labels=sentiment_counts.index,
                            values=sentiment_counts.values,
                            hole=.3,
                            marker_colors=[colors[cat] for cat in sentiment_counts.index]
                        )])
                        
                        fig.update_layout(
                            title="Sentiment Distribution",
                            height=300
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        # News list with sentiment
                        for _, news in news_df.head(5).iterrows():
                            sentiment_color = (
                                "🟢" if news['sentiment_category'] == 'Positive'
                                else "🔴" if news['sentiment_category'] == 'Negative'
                                else "⚪"
                            )
                            st.markdown(f"{sentiment_color} **{news['title']}**")
                            st.markdown(f"*Source: {news['source']} | {news['publishedAt']}*")
                            with st.expander("Read more"):
                                st.write(news['description'])
                                st.markdown(f"[Read full article]({news['url']})")
                else:
                    st.info("No recent news available")

        with tabs[1]:  # Technical Analysis Tab
            with st.spinner("Calculating technical indicators..."):
                # Get technical analysis data
                analysis_data = technical_analysis.analyze(symbol)
                
                if analysis_data is None:
                    st.error("Unable to calculate technical indicators. This could be due to insufficient data or invalid symbol.")
                    st.stop()
                
                # Create three columns for different indicator categories
                trend_col, momentum_col, volume_col = st.columns(3)
                
                with trend_col:
                    st.subheader("Trend Indicators")
                    
                    # MA Cross
                    ma_data = analysis_data['ma_cross']
                    if ma_data['MA20'] is not None and ma_data['MA50'] is not None:
                        st.write("Moving Average Crossover")
                        st.write(f"MA20: {ma_data['MA20']:.2f}")
                        st.write(f"MA50: {ma_data['MA50']:.2f}")
                        st.write(f"Signal: {ma_data['signal']}")
                    else:
                        st.warning("Moving Average data not available")
                    
                    # MACD
                    macd_data = analysis_data['macd']
                    if macd_data['MACD'] is not None and macd_data['Signal'] is not None:
                        st.write("Moving Average Convergence Divergence (MACD)")
                        st.write(f"MACD: {macd_data['MACD']:.2f}")
                        st.write(f"Signal: {macd_data['Signal']:.2f}")
                        st.write(f"Signal: {macd_data['signal']}")
                    else:
                        st.warning("MACD data not available")
                
                with momentum_col:
                    st.subheader("Momentum Indicators")
                    
                    # RSI
                    rsi_data = analysis_data['rsi']
                    if rsi_data['RSI'] is not None:
                        st.write("Relative Strength Index (RSI)")
                        st.write(f"RSI: {rsi_data['RSI']:.2f}")
                        st.write(f"Signal: {rsi_data['signal']}")
                    else:
                        st.warning("RSI data not available")
                        
                    # Stochastic
                    stoch_data = analysis_data['stochastic']
                    if stoch_data['K'] is not None and stoch_data['D'] is not None:
                        st.write("Stochastic Oscillator")
                        st.write(f"K: {stoch_data['K']:.2f}")
                        st.write(f"D: {stoch_data['D']:.2f}")
                        st.write(f"Signal: {stoch_data['signal']}")
                    else:
                        st.warning("Stochastic data not available")
                
                with volume_col:
                    st.subheader("Volume Indicators")
                    
                    # OBV
                    obv_data = analysis_data['obv']
                    if obv_data['OBV'] is not None:
                        st.write("On-Balance Volume (OBV)")
                        st.write(f"OBV: {obv_data['OBV']:.2f}")
                        st.write(f"Signal: {obv_data['signal']}")
                    else:
                        st.warning("OBV data not available")
                    
                    # Volume
                    volume_data = analysis_data['volume']
                    if volume_data['Volume'] is not None:
                        st.write("Volume Analysis")
                        st.write(f"Volume: {volume_data['Volume']:.2f}")
                        st.write(f"Signal: {volume_data['signal']}")
                    else:
                        st.warning("Volume data not available")
                    
                    # Technical Analysis Summary
                    st.subheader("Technical Analysis Summary")
                    signals = [
                        analysis_data['ma_cross']['signal'],
                        analysis_data['macd']['signal'],
                        analysis_data['rsi']['signal'],
                        analysis_data['stochastic']['signal'],
                        analysis_data['obv']['signal']
                    ]
                    
                    buy_signals = sum(1 for s in signals if s == "Buy")
                    sell_signals = sum(1 for s in signals if s == "Sell")
                    
                    summary = (
                        "Strong Buy" if buy_signals >= 4 else
                        "Buy" if buy_signals > sell_signals else
                        "Strong Sell" if sell_signals >= 4 else
                        "Sell" if sell_signals > buy_signals else
                        "Neutral"
                    )
                    
                    st.metric(
                        "Overall Signal",
                        summary,
                        delta=f"{buy_signals} Buy vs {sell_signals} Sell signals"
                    )
                    
                # Technical Analysis Summary
                st.subheader("Technical Analysis Summary")
                signals = [
                    analysis_data['ma_cross']['signal'],
                    analysis_data['macd']['signal'],
                    analysis_data['rsi']['signal'],
                    analysis_data['stochastic']['signal'],
                    analysis_data['obv']['signal']
                ]
                
                buy_signals = sum(1 for s in signals if s == "Buy")
                sell_signals = sum(1 for s in signals if s == "Sell")
                
                summary = (
                    "Strong Buy" if buy_signals >= 4 else
                    "Buy" if buy_signals > sell_signals else
                    "Strong Sell" if sell_signals >= 4 else
                    "Sell" if sell_signals > buy_signals else
                    "Neutral"
                )
                
                st.metric(
                    "Overall Signal",
                    summary,
                    delta=f"{buy_signals} Buy vs {sell_signals} Sell signals"
                )
        
        with tabs[2]:  # Price Prediction Tab
            with st.spinner("Analyzing technical indicators..."):
                try:
                    # Verify we have enough data
                    stock = yf.Ticker(symbol)
                    hist = stock.history(period="60d")
                    if len(hist) < 20:  # Need at least 20 days for technical analysis
                        st.error(f"Insufficient historical data for {symbol}. Need at least 20 days of trading data.")
                        st.stop()
                    
                    print(f"\nMaking prediction for {symbol}")
                    print(f"Historical data available: {len(hist)} days")
                    
                    predictions = prediction_service.predict(symbol)
                    if predictions is None:
                        st.error("Unable to generate predictions. Check the logs for details.")
                        st.stop()
                        
                    confidence_intervals = prediction_service.get_confidence_intervals(symbol)
                    if confidence_intervals is None:
                        st.error("Unable to calculate confidence intervals.")
                        st.stop()
                    
                    if predictions and confidence_intervals:
                        st.subheader("Technical Analysis Prediction")
                        
                        # Get current price
                        current_price = hist['Close'].iloc[-1]
                        
                        # Display prediction
                        pred_price = predictions['technical']
                        conf_interval = confidence_intervals[symbol]['technical']
                        
                        # Calculate percentage change
                        price_change = ((pred_price / current_price) - 1) * 100
                        
                        # Create columns for prediction display
                        pred_col1, pred_col2 = st.columns(2)
                        
                        with pred_col1:
                            st.metric(
                                "Current Price",
                                f"${current_price:.2f}"
                            )
                            
                            st.metric(
                                "Predicted Price",
                                f"${pred_price:.2f}",
                                f"{price_change:+.2f}%",
                                delta_color="normal"
                            )
                        
                        with pred_col2:
                            st.metric(
                                "Lower Bound",
                                f"${conf_interval[0]:.2f}",
                                f"{((conf_interval[0]/current_price - 1) * 100):+.2f}%",
                                delta_color="inverse"
                            )
                            
                            st.metric(
                                "Upper Bound",
                                f"${conf_interval[1]:.2f}",
                                f"{((conf_interval[1]/current_price - 1) * 100):+.2f}%",
                                delta_color="normal"
                            )
                        
                        # Add prediction explanation
                        st.markdown("### Analysis Details")
                        st.markdown("""
                        This prediction is based on multiple technical indicators including:
                        - Relative Strength Index (RSI)
                        - Moving Average Convergence Divergence (MACD)
                        - Moving Averages (5-day and 20-day)
                        - Price Momentum
                        - Volume Analysis
                        - Bollinger Bands
                        
                        The prediction represents a weighted combination of signals from these indicators.
                        The confidence interval shows the potential price range based on the strength and consistency of these signals.
                        """)
                        
                    else:
                        st.error("Unable to generate predictions. This could be due to insufficient data or invalid technical indicators.")
                except Exception as e:
                    st.error(f"Error during prediction: {str(e)}")
                    print(f"Error details: {str(e)}")
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
