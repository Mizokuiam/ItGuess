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
    technical_analysis = TechnicalAnalysisService()
    prediction_service = PredictionService()
    news_service = NewsService()
    return technical_analysis, prediction_service, news_service

technical_analysis, prediction_service, news_service = get_services()

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
                            template='plotly_dark' if st.session_state.theme == 'dark' else 'plotly_white'
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
                
                if analysis_data is not None:
                    # Create three columns for different indicator categories
                    trend_col, momentum_col, volume_col = st.columns(3)
                    
                    with trend_col:
                        st.subheader("Trend Indicators")
                        
                        # MA Cross
                        ma_data = analysis_data['ma_cross']
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            y=ma_data['MA20'][-30:],
                            name='MA20',
                            line=dict(color='blue')
                        ))
                        fig.add_trace(go.Scatter(
                            y=ma_data['MA50'][-30:],
                            name='MA50',
                            line=dict(color='orange')
                        ))
                        fig.update_layout(
                            height=100,
                            margin=dict(l=0, r=0, t=0, b=0),
                            showlegend=False
                        )
                        st.metric(
                            "MA Cross",
                            ma_data['signal'],
                            delta="Bullish" if ma_data['signal'] == "Buy" else "Bearish" if ma_data['signal'] == "Sell" else None,
                            delta_color="normal"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # MACD
                        macd_data = analysis_data['macd']
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            y=macd_data['MACD'][-30:],
                            name='MACD',
                            line=dict(color='blue')
                        ))
                        fig.add_trace(go.Scatter(
                            y=macd_data['Signal'][-30:],
                            name='Signal',
                            line=dict(color='orange')
                        ))
                        fig.update_layout(
                            height=100,
                            margin=dict(l=0, r=0, t=0, b=0),
                            showlegend=False
                        )
                        st.metric(
                            "MACD",
                            macd_data['signal'],
                            delta="Bullish" if macd_data['signal'] == "Buy" else "Bearish" if macd_data['signal'] == "Sell" else None
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with momentum_col:
                        st.subheader("Momentum Indicators")
                        
                        # RSI
                        rsi_data = analysis_data['rsi']
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            y=rsi_data['RSI'][-30:],
                            line=dict(color='purple')
                        ))
                        fig.add_hline(y=70, line_dash="dash", line_color="red")
                        fig.add_hline(y=30, line_dash="dash", line_color="green")
                        fig.update_layout(
                            height=100,
                            margin=dict(l=0, r=0, t=0, b=0),
                            showlegend=False
                        )
                        st.metric(
                            "RSI",
                            f"{rsi_data['value']:.1f}",
                            delta="Overbought" if rsi_data['value'] > 70 else "Oversold" if rsi_data['value'] < 30 else "Neutral"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Stochastic
                        stoch_data = analysis_data['stochastic']
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            y=stoch_data['K'][-30:],
                            name='%K',
                            line=dict(color='blue')
                        ))
                        fig.add_trace(go.Scatter(
                            y=stoch_data['D'][-30:],
                            name='%D',
                            line=dict(color='orange')
                        ))
                        fig.add_hline(y=80, line_dash="dash", line_color="red")
                        fig.add_hline(y=20, line_dash="dash", line_color="green")
                        fig.update_layout(
                            height=100,
                            margin=dict(l=0, r=0, t=0, b=0),
                            showlegend=False
                        )
                        st.metric(
                            "Stochastic",
                            stoch_data['signal'],
                            delta="Bullish" if stoch_data['signal'] == "Buy" else "Bearish" if stoch_data['signal'] == "Sell" else None
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with volume_col:
                        st.subheader("Volume Indicators")
                        
                        # OBV
                        obv_data = analysis_data['obv']
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            y=obv_data['OBV'][-30:],
                            line=dict(color='green')
                        ))
                        fig.update_layout(
                            height=100,
                            margin=dict(l=0, r=0, t=0, b=0),
                            showlegend=False
                        )
                        st.metric(
                            "On-Balance Volume",
                            obv_data['signal'],
                            delta="Increasing" if obv_data['trend'] == "up" else "Decreasing" if obv_data['trend'] == "down" else None
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Volume
                        volume_data = analysis_data['volume']
                        fig = go.Figure()
                        fig.add_trace(go.Bar(
                            y=volume_data['Volume'][-30:],
                            marker_color='lightblue'
                        ))
                        fig.add_trace(go.Scatter(
                            y=volume_data['MA20'][-30:],
                            line=dict(color='red')
                        ))
                        fig.update_layout(
                            height=100,
                            margin=dict(l=0, r=0, t=0, b=0),
                            showlegend=False
                        )
                        st.metric(
                            "Volume Analysis",
                            volume_data['signal'],
                            delta="Above Average" if volume_data['trend'] == "up" else "Below Average" if volume_data['trend'] == "down" else "Average"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
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
                    
                else:
                    st.error("Unable to calculate technical indicators")
        
        with tabs[2]:  # Price Prediction Tab
            with st.spinner("Training prediction models..."):
                # Train models if needed
                if prediction_service.train_models(symbol):
                    predictions = prediction_service.predict(symbol)
                    confidence_intervals = prediction_service.get_confidence_intervals(symbol)
                    feature_importance = prediction_service.get_feature_importance(symbol)
                    prediction_history = prediction_service.get_prediction_history(symbol)
                    
                    if predictions and confidence_intervals:
                        st.subheader("Price Predictions")
                        
                        # Create columns for each model's prediction
                        rf_col, nn_col = st.columns(2)
                        
                        with rf_col:
                            st.markdown("##### Random Forest Model")
                            rf_pred = predictions['rf']
                            rf_ci = confidence_intervals['rf']
                            
                            # Display prediction with confidence interval
                            st.metric(
                                "Predicted Price",
                                f"${rf_pred:.2f}",
                                delta=f"CI: ${rf_ci[0]:.2f} to ${rf_ci[1]:.2f}"
                            )
                            
                            # Feature importance plot
                            if feature_importance:
                                fig = go.Figure(go.Bar(
                                    x=list(feature_importance.values()),
                                    y=list(feature_importance.keys()),
                                    orientation='h'
                                ))
                                fig.update_layout(
                                    title="Feature Importance",
                                    height=300,
                                    margin=dict(l=0, r=0, t=30, b=0)
                                )
                                st.plotly_chart(fig, use_container_width=True)
                        
                        with nn_col:
                            st.markdown("##### Neural Network Model")
                            nn_pred = predictions['nn']
                            nn_ci = confidence_intervals['nn']
                            
                            # Display prediction with confidence interval
                            st.metric(
                                "Predicted Price",
                                f"${nn_pred:.2f}",
                                delta=f"CI: ${nn_ci[0]:.2f} to ${nn_ci[1]:.2f}"
                            )
                            
                            # Model performance metrics
                            metrics = prediction_service.metrics.get(symbol, {}).get('nn', {})
                            if metrics:
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.metric("RMSE", f"${metrics['rmse']:.2f}")
                                with col2:
                                    st.metric("R² Score", f"{metrics['r2']:.3f}")
                        
                        # Prediction History
                        if prediction_history:
                            st.subheader("Prediction History")
                            
                            # Create prediction history plot
                            fig = go.Figure()
                            
                            # Actual prices
                            fig.add_trace(go.Scatter(
                                y=prediction_history['actual'],
                                name='Actual Price',
                                line=dict(color='black')
                            ))
                            
                            # RF predictions
                            fig.add_trace(go.Scatter(
                                y=prediction_history['rf_pred'],
                                name='RF Predictions',
                                line=dict(color='blue', dash='dash')
                            ))
                            
                            # NN predictions
                            fig.add_trace(go.Scatter(
                                y=prediction_history['nn_pred'],
                                name='NN Predictions',
                                line=dict(color='red', dash='dash')
                            ))
                            
                            fig.update_layout(
                                title="Model Predictions vs Actual Prices",
                                xaxis_title="Time",
                                yaxis_title="Price",
                                height=400,
                                template='plotly_dark' if st.session_state.theme == 'dark' else 'plotly_white'
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Prediction Accuracy Metrics
                            st.subheader("Model Performance")
                            metrics = prediction_service.metrics.get(symbol, {})
                            
                            if metrics:
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    st.markdown("##### Random Forest Metrics")
                                    rf_metrics = metrics.get('rf', {})
                                    st.metric("RMSE", f"${rf_metrics.get('rmse', 0):.2f}")
                                    st.metric("R² Score", f"{rf_metrics.get('r2', 0):.3f}")
                                
                                with col2:
                                    st.markdown("##### Neural Network Metrics")
                                    nn_metrics = metrics.get('nn', {})
                                    st.metric("RMSE", f"${nn_metrics.get('rmse', 0):.2f}")
                                    st.metric("R² Score", f"{nn_metrics.get('r2', 0):.3f}")
                    else:
                        st.error("Unable to generate predictions")
                else:
                    st.error("Unable to train prediction models")
        
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
                    index=1 if st.session_state.theme == 'dark' else 0
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
                            template='plotly_dark' if chart_theme == 'Dark' else 'plotly_white',
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
