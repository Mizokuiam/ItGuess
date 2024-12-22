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

# Page config
st.set_page_config(
    page_title="ItGuess - Stock Price Predictor",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'theme' not in st.session_state:
    st.session_state.theme = 'light'
if 'last_symbol' not in st.session_state:
    st.session_state.last_symbol = ''
if 'last_update' not in st.session_state:
    st.session_state.last_update = datetime.now()

# Custom CSS with dynamic theme
def get_theme_css():
    if st.session_state.theme == 'dark':
        return """
        :root {
            --primary-color: #4CAF50;
            --background-color: #1E1E1E;
            --text-color: #FFFFFF;
            --card-bg-color: #2D2D2D;
            --hover-color: #3D3D3D;
            --border-color: #404040;
        }
        """
    else:
        return """
        :root {
            --primary-color: #1E88E5;
            --background-color: #FFFFFF;
            --text-color: #333333;
            --card-bg-color: #F8F9FA;
            --hover-color: #E9ECEF;
            --border-color: #DEE2E6;
        }
        """

st.markdown(f"""
    <style>
        {get_theme_css()}
        
        /* Global Styles */
        .stApp {{
            background-color: var(--background-color);
            color: var(--text-color);
        }}
        
        /* Main title styling */
        .main-title {{
            color: var(--text-color);
            font-family: 'SF Pro Display', -apple-system, BlinkMacSystemFont, sans-serif;
            font-weight: 700;
            font-size: 2.5rem;
            margin-bottom: 0.5rem;
            letter-spacing: -0.5px;
            background: linear-gradient(135deg, var(--primary-color), #64B5F6);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            text-shadow: 2px 2px 4px rgba(30, 136, 229, 0.1);
        }}
        
        /* Card styling */
        .stMetric {{
            background-color: var(--card-bg-color);
            border-radius: 10px;
            padding: 1rem;
            border: 1px solid var(--border-color);
            transition: transform 0.2s;
        }}
        
        .stMetric:hover {{
            transform: translateY(-2px);
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }}
        
        /* Tab styling */
        .stTabs [data-baseweb="tab-list"] {{
            gap: 2px;
            background-color: var(--card-bg-color);
            border-radius: 10px;
            padding: 0.5rem;
        }}
        
        .stTabs [data-baseweb="tab"] {{
            border-radius: 6px;
            padding: 0.5rem 1rem;
            transition: all 0.2s;
        }}
        
        .stTabs [data-baseweb="tab"]:hover {{
            background-color: var(--hover-color);
        }}
        
        /* Button styling */
        .stButton>button {{
            border-radius: 6px;
            padding: 0.5rem 1rem;
            background-color: var(--primary-color);
            color: white;
            border: none;
            transition: all 0.2s;
        }}
        
        .stButton>button:hover {{
            transform: translateY(-1px);
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        }}
        
        /* Slider styling */
        .stSlider {{
            padding: 1rem 0;
        }}
        
        /* Tooltip styling */
        .tooltip {{
            position: relative;
            display: inline-block;
        }}
        
        .tooltip .tooltiptext {{
            visibility: hidden;
            background-color: var(--card-bg-color);
            color: var(--text-color);
            text-align: center;
            padding: 5px;
            border-radius: 6px;
            border: 1px solid var(--border-color);
            position: absolute;
            z-index: 1;
            bottom: 125%;
            left: 50%;
            transform: translateX(-50%);
            opacity: 0;
            transition: opacity 0.2s;
        }}
        
        .tooltip:hover .tooltiptext {{
            visibility: visible;
            opacity: 1;
        }}
        
        /* Animation classes */
        @keyframes fadeIn {{
            from {{ opacity: 0; }}
            to {{ opacity: 1; }}
        }}
        
        .fade-in {{
            animation: fadeIn 0.5s ease-in;
        }}
        
        /* Responsive design */
        @media (max-width: 768px) {{
            .main-title {{
                font-size: 2rem;
            }}
        }}
        
        /* Company symbol */
        .company-symbol {{
            font-size: 3rem;
            font-weight: bold;
            color: var(--primary-color);
        }}
        
        /* Company description */
        .company-description {{
            font-size: 1.2rem;
            color: var(--text-color);
        }}
    </style>
""", unsafe_allow_html=True)

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
@st.cache_resource
def get_services():
    return TechnicalAnalysisService(), PredictionService()

technical_analysis, prediction_service = get_services()

# Sidebar
with st.sidebar:
    st.image("https://raw.githubusercontent.com/Mizokuiam/ItGuess/master/static/images/logo.png", width=50)
    st.title("ItGuess")
    
    # Theme toggle
    theme = st.toggle("Dark Mode", value=st.session_state.theme == 'dark')
    st.session_state.theme = 'dark' if theme else 'light'
    
    # About section
    with st.expander("About ItGuess"):
        st.write("""
        ItGuess is an advanced stock price prediction tool that combines:
        - Technical Analysis
        - Machine Learning
        - Real-time Market Data
        
        Make informed investment decisions with our comprehensive analysis.
        """)
    
    # Settings sections
    st.subheader("Settings")
    
    # Stock input with popular stocks
    with st.expander("Stock Selection", expanded=True):
        popular_stocks = {
            "Tech": ["AAPL", "GOOGL", "MSFT"],
            "EV": ["TSLA", "NIO", "RIVN"],
            "Finance": ["JPM", "BAC", "GS"]
        }
        
        for sector, stocks in popular_stocks.items():
            cols = st.columns(len(stocks))
            for i, stock in enumerate(stocks):
                if cols[i].button(stock, key=f"quick_{stock}"):
                    st.session_state.symbol = stock
        
        symbol = st.text_input("Enter Stock Symbol (e.g., AAPL)", 
                             value=st.session_state.get('symbol', 'AAPL'),
                             help="Enter the stock symbol you want to analyze")
    
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
        # Load company info and logo
        with st.spinner("Loading company information..."):
            company_info = get_company_info(symbol)
            company_logo = load_company_logo(symbol)
            
        # Main header with company info
        col1, col2 = st.columns([1, 3])
        with col1:
            if company_logo:
                st.image(company_logo, width=100)
            else:
                st.markdown(f"<div class='company-symbol'>{symbol}</div>", unsafe_allow_html=True)
                
        with col2:
            if company_info:
                st.markdown(f"<h1 class='main-title'>{company_info['name']} ({symbol})</h1>", unsafe_allow_html=True)
                st.markdown(f"<p class='company-meta'>{company_info['sector']} | {company_info['industry']}</p>", 
                          unsafe_allow_html=True)
            else:
                st.markdown(f"<h1 class='main-title'>{symbol}</h1>", unsafe_allow_html=True)

        # Create tabs with animation
        tab_names = ["Overview", "Technical Analysis", "Price Prediction", "Live Chart"]
        tabs = st.tabs(tab_names)

        with tabs[0]:  # Overview Tab
            with st.container():
                # Company Description
                if company_info and company_info['description'] != 'N/A':
                    with st.expander("About the Company", expanded=True):
                        st.markdown(f"<div class='company-description'>{company_info['description']}</div>",
                                  unsafe_allow_html=True)
                
                # Quick Stats
                st.subheader("Quick Stats")
                try:
                    stock = yf.Ticker(symbol)
                    info = stock.info
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    # Market Cap
                    with col1:
                        market_cap = info.get('marketCap', 0)
                        market_cap_str = f"${market_cap/1e9:.2f}B" if market_cap >= 1e9 else f"${market_cap/1e6:.2f}M"
                        st.metric("Market Cap", market_cap_str)
                    
                    # P/E Ratio
                    with col2:
                        pe_ratio = info.get('trailingPE', 'N/A')
                        pe_ratio = f"{pe_ratio:.2f}" if isinstance(pe_ratio, (int, float)) else pe_ratio
                        st.metric("P/E Ratio", pe_ratio)
                    
                    # 52W Range
                    with col3:
                        high_52w = info.get('fiftyTwoWeekHigh', 0)
                        low_52w = info.get('fiftyTwoWeekLow', 0)
                        if high_52w and low_52w:
                            range_52w = f"${low_52w:.2f} - ${high_52w:.2f}"
                        else:
                            range_52w = "N/A"
                        st.metric("52W Range", range_52w)
                    
                    # Volume
                    with col4:
                        volume = info.get('volume', 0)
                        volume_str = f"{volume/1e6:.1f}M" if volume >= 1e6 else f"{volume/1e3:.1f}K"
                        st.metric("Volume", volume_str)
                
                except Exception as e:
                    st.error(f"Error loading stock information: {str(e)}")
                
                # Price Movement Chart
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
                        template='plotly_dark' if st.session_state.theme == 'dark' else 'plotly_white'
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                except Exception as e:
                    st.error(f"Error creating price movement chart: {str(e)}")

        with tabs[1]:  # Technical Analysis Tab
            with st.spinner("Calculating technical indicators..."):
                # Get technical analysis data
                technical_analysis.fetch_data(symbol)
                indicators = technical_analysis.calculate_indicators()
                
                if indicators:
                    # Technical Indicators Overview
                    st.subheader("Technical Indicators")
                    
                    # Create three columns for different types of indicators
                    momentum_col, trend_col, volume_col = st.columns(3)
                    
                    with momentum_col:
                        st.markdown("##### Momentum Indicators")
                        # RSI
                        rsi_val = indicators.get('RSI', 'N/A')
                        if rsi_val != 'N/A':
                            rsi_color = 'inverse' if float(rsi_val) > 70 else 'normal' if float(rsi_val) < 30 else 'off'
                            st.metric("RSI (14)", f"{rsi_val}", 
                                    delta="Overbought" if float(rsi_val) > 70 else "Oversold" if float(rsi_val) < 30 else " ",
                                    delta_color=rsi_color)
                            
                            # RSI Gauge Chart
                            fig = go.Figure(go.Indicator(
                                mode="gauge+number",
                                value=float(rsi_val),
                                domain={'x': [0, 1], 'y': [0, 1]},
                                gauge={
                                    'axis': {'range': [0, 100]},
                                    'bar': {'color': "#1E88E5"},
                                    'steps': [
                                        {'range': [0, 30], 'color': "lightgreen"},
                                        {'range': [30, 70], 'color': "lightgray"},
                                        {'range': [70, 100], 'color': "lightcoral"}
                                    ]
                                }
                            ))
                            fig.update_layout(height=200, margin=dict(l=10, r=10, t=30, b=10))
                            st.plotly_chart(fig, use_container_width=True)
                    
                    with trend_col:
                        st.markdown("##### Trend Indicators")
                        # MACD
                        macd_val = indicators.get('MACD', 'N/A')
                        signal_val = indicators.get('Signal', 'N/A')
                        if macd_val != 'N/A' and signal_val != 'N/A':
                            macd_signal_diff = float(macd_val) - float(signal_val)
                            st.metric("MACD", f"{macd_val}", 
                                    delta=f"{macd_signal_diff:.2f}",
                                    delta_color="normal" if macd_signal_diff > 0 else "inverse")
                        
                        # Moving Averages
                        ema20 = indicators.get('EMA20', 'N/A')
                        ema50 = indicators.get('EMA50', 'N/A')
                        if ema20 != 'N/A' and ema50 != 'N/A':
                            ma_diff = float(ema20) - float(ema50)
                            st.metric("EMA Cross", f"EMA20: {ema20}",
                                    delta=f"vs EMA50: {ma_diff:.2f}",
                                    delta_color="normal" if ma_diff > 0 else "inverse")
                    
                    with volume_col:
                        st.markdown("##### Volume Analysis")
                        # Volume
                        volume = indicators.get('Volume', 'N/A')
                        if volume != 'N/A':
                            volume_str = f"{int(volume):,}"
                            st.metric("Volume", volume_str)
                        
                        # Bollinger Bands
                        bb_upper = indicators.get('BB_Upper', 'N/A')
                        bb_lower = indicators.get('BB_Lower', 'N/A')
                        if bb_upper != 'N/A' and bb_lower != 'N/A':
                            bb_width = float(bb_upper) - float(bb_lower)
                            st.metric("BB Width", f"{bb_width:.2f}")
                    
                    # Technical Analysis Summary
                    st.subheader("Analysis Summary")
                    
                    # Generate analysis text based on indicators
                    summary_text = ""
                    if rsi_val != 'N/A':
                        if float(rsi_val) > 70:
                            summary_text += "- RSI indicates **overbought** conditions. Consider taking profits.\n"
                        elif float(rsi_val) < 30:
                            summary_text += "- RSI indicates **oversold** conditions. Watch for potential reversal.\n"
                    
                    if macd_val != 'N/A' and signal_val != 'N/A':
                        if float(macd_val) > float(signal_val):
                            summary_text += "- MACD shows **bullish** momentum.\n"
                        else:
                            summary_text += "- MACD shows **bearish** momentum.\n"
                    
                    if ema20 != 'N/A' and ema50 != 'N/A':
                        if float(ema20) > float(ema50):
                            summary_text += "- Moving averages indicate an **upward** trend.\n"
                        else:
                            summary_text += "- Moving averages indicate a **downward** trend.\n"
                    
                    st.markdown(summary_text)
                
                else:
                    st.error("Unable to calculate technical indicators. Please try again.")
        
        with tabs[2]:  # Price Prediction Tab
            # Price Prediction
            with st.spinner('Generating price predictions...'):
                st.header("Price Prediction")
                
                try:
                    # Train prediction models
                    if prediction_service.train_models(symbol):
                        # Make predictions
                        predictions = prediction_service.predict(period)
                        
                        if predictions and isinstance(predictions, dict):
                            col1, col2 = st.columns([2, 1])
                            with col1:
                                st.subheader("Price Predictions")
                                for model, pred in predictions.items():
                                    if isinstance(pred, (int, float)):
                                        st.metric(f"{model} Prediction", f"${pred:.2f}")
                                    else:
                                        st.metric(f"{model} Prediction", "N/A")
                            
                            with col2:
                                st.subheader("Model Metrics")
                                metrics = prediction_service.metrics
                                if metrics and isinstance(metrics, dict):
                                    for model, metric in metrics.items():
                                        if isinstance(metric, (int, float)):
                                            st.metric(f"{model} Accuracy", f"{metric:.1%}")
                                        else:
                                            st.metric(f"{model} Accuracy", "N/A")
                                else:
                                    st.warning("Model metrics not available")
                        else:
                            st.warning("Could not generate predictions. Please try again.")
                    else:
                        st.warning("Could not train prediction models. Not enough data points.")
                except Exception as e:
                    st.error(f"Error in price prediction: {str(e)}")
        
        with tabs[3]:  # Live Chart Tab
            # Live Chart with Technical Analysis
            st.header("Live Chart")
            
            try:
                # Get periods from session state
                rsi_period = st.session_state.get('rsi_period', 14)
                ma_period = st.session_state.get('ma_period', 20)
                
                # Create technical analysis instance with current data
                ta_service = TechnicalAnalysisService(symbol=symbol, data=hist.copy())
                
                # Calculate indicators
                rsi = ta_service.calculate_rsi(period=rsi_period)
                indicators = ta_service.calculate_indicators()
                
                # Create figure with secondary y-axis
                fig = make_subplots(rows=2, cols=1, 
                                  shared_xaxes=True,  
                                  vertical_spacing=0.03, 
                                  row_heights=[0.7, 0.3],
                                  subplot_titles=('Price', 'RSI'))

                # Add candlestick
                fig.add_trace(go.Candlestick(x=hist.index,
                                            open=hist['Open'],
                                            high=hist['High'],
                                            low=hist['Low'],
                                            close=hist['Close'],
                                            name='OHLC'),
                            row=1, col=1)

                # Add Volume as bar chart
                colors = ['red' if row['Open'] - row['Close'] >= 0 
                        else 'green' for index, row in hist.iterrows()]
                
                fig.add_trace(go.Bar(x=hist.index, 
                                   y=hist['Volume'],
                                   name='Volume',
                                   marker_color=colors,
                                   opacity=0.3),
                            row=1, col=1)

                # Add RSI
                if rsi is not None:
                    fig.add_trace(go.Scatter(x=hist.index, 
                                           y=rsi,
                                           name='RSI',
                                           line=dict(color='purple')),
                                row=2, col=1)
                    
                    # Add RSI reference lines
                    fig.add_hline(y=70, line_width=1, line_dash="dash", line_color="red", 
                                row=2, col=1)
                    fig.add_hline(y=30, line_width=1, line_dash="dash", line_color="green", 
                                row=2, col=1)

                # Update layout
                fig.update_layout(
                    title=f'{symbol} Live Chart',
                    yaxis=dict(title='Price'),
                    yaxis2=dict(title='Volume', overlaying='y', side='right'),
                    yaxis3=dict(title='RSI', domain=[0, 0.3]),
                    xaxis_rangeslider_visible=False,
                    height=800,
                    showlegend=True,
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1
                    )
                )

                # Update axes ranges
                fig.update_yaxes(title_text="Price", secondary_y=False, row=1, col=1)
                fig.update_yaxes(title_text="RSI", range=[0, 100], row=2, col=1)

                # Display the chart
                st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.error(f"Error creating live chart: {str(e)}")
    
    except Exception as e:
        st.error(f"Error fetching data: {str(e)}")
else:
    st.info("Please enter a valid stock symbol. Example: AAPL for Apple Inc.")
