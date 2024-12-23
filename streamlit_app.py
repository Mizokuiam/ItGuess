import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from plotly.subplots import make_subplots
import requests
from PIL import Image
from io import BytesIO
from services.technical_analysis import TechnicalAnalysisService
from services.prediction import PredictionService

# Configure Streamlit page
st.set_page_config(
    page_title="ItGuess - Smart Stock Analysis",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for basic styling
st.markdown("""
<style>
    .app-title {
        font-family: sans-serif;
        font-size: 3em;
        font-weight: 700;
        text-align: center;
        margin: 20px 0;
    }
    
    .app-subtitle {
        font-family: sans-serif;
        font-size: 1.2em;
        text-align: center;
        margin-bottom: 30px;
        opacity: 0.9;
    }
    
    /* Feature cards styling */
    .features-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
        gap: 2rem;
        padding: 2rem;
        max-width: 1200px;
        margin: 2rem auto;
    }
    
    .feature-card {
        background-color: rgba(255, 255, 255, 0.05);
        border-radius: 15px;
        padding: 2rem;
        text-align: center;
        transition: all 0.3s ease;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        min-height: 250px;
    }
    
    .feature-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
        border-color: rgba(255, 75, 75, 0.5);
    }
    
    .feature-icon {
        font-size: 3em;
        margin-bottom: 1.5rem;
        display: block;
        background: linear-gradient(135deg, #FF4B4B, #FF8080);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.1);
    }
    
    .feature-title {
        font-size: 1.4em;
        font-weight: 600;
        margin: 0.5rem 0;
        color: inherit;
    }
    
    .feature-description {
        font-size: 1em;
        opacity: 0.85;
        line-height: 1.6;
        margin: 0;
        color: inherit;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display: none;}
    
    /* Input field styling */
    .stTextInput > div > div {
        background-color: rgba(255, 255, 255, 0.05);
        border-radius: 10px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 0.5rem 1rem;
    }
    
    .stTextInput > div > div:focus-within {
        border-color: #FF4B4B;
        box-shadow: 0 0 0 1px #FF4B4B;
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
        
        # Get additional key statistics
        try:
            market_stats = {
                'Market Cap': info.get('marketCap', 'N/A'),
                'Volume': info.get('volume', 'N/A'),
                'P/E Ratio': info.get('trailingPE', 'N/A'),
                '52 Week High': info.get('fiftyTwoWeekHigh', 'N/A'),
                '52 Week Low': info.get('fiftyTwoWeekLow', 'N/A'),
                'Beta': info.get('beta', 'N/A'),
                'Dividend Yield': info.get('dividendYield', 'N/A'),
                'Profit Margin': info.get('profitMargins', 'N/A'),
            }
            
            # Format the values
            for key, value in market_stats.items():
                if value != 'N/A':
                    if key == 'Market Cap':
                        value = f"${value:,.0f}" if value >= 1e9 else f"${value/1e6:.2f}M"
                    elif key == 'Volume':
                        value = f"{value:,.0f}"
                    elif key in ['P/E Ratio', 'Beta']:
                        value = f"{value:.2f}"
                    elif key in ['52 Week High', '52 Week Low']:
                        value = f"${value:.2f}"
                    elif key in ['Dividend Yield', 'Profit Margin']:
                        value = f"{value*100:.2f}%" if value is not None else 'N/A'
                market_stats[key] = value
                
            info.update(market_stats)
        except Exception as e:
            st.warning(f"Some market statistics may be unavailable: {str(e)}")
            
        return info
    except Exception as e:
        st.error(f"Error fetching company information: {str(e)}")
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
    
    # Search box with improved styling
    st.markdown("<div class='search-container'>", unsafe_allow_html=True)
    symbol = st.text_input("Enter stock symbol", placeholder="e.g. AAPL, GOOGL, MSFT", key="symbol_input")
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
if not symbol or len(symbol.strip()) == 0:  # Show welcome page when no symbol is entered
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
        <div class="features-grid">
            <div class="feature-card">
                <div class="feature-icon">📊</div>
                <h3 class="feature-title">Stock Information</h3>
                <p class="feature-description">
                    Comprehensive analysis of stock data, company details, and real-time market metrics for informed investment decisions.
                </p>
            </div>
            
            <div class="feature-card">
                <div class="feature-icon">📈</div>
                <h3 class="feature-title">Technical Analysis</h3>
                <p class="feature-description">
                    Advanced indicators including RSI, MACD, and Bollinger Bands for precise market trend analysis.
                </p>
            </div>
            
            <div class="feature-card">
                <div class="feature-icon">🔮</div>
                <h3 class="feature-title">Price Prediction</h3>
                <p class="feature-description">
                    AI-powered forecasting using machine learning to predict future stock price movements and trends.
                </p>
            </div>
            
            <div class="feature-card">
                <div class="feature-icon">📉</div>
                <h3 class="feature-title">Live Charts</h3>
                <p class="feature-description">
                    Interactive real-time charts with customizable timeframes and technical overlay indicators.
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
                st.markdown("")
                
        with col2:
            if company_info:
                st.markdown(f"<h1 class='gradient-text'>{company_info['longName']} ({symbol})</h1>", unsafe_allow_html=True)
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
                        display_company_logo(symbol)
                        
                    with col2:
                        if 'longName' in info:
                            st.title(info['longName'])
                        if 'symbol' in info:
                            st.subheader(f"Symbol: {info['symbol']}")
                        if 'sector' in info and 'industry' in info:
                            st.text(f"Sector: {info.get('sector', 'N/A')} | Industry: {info.get('industry', 'N/A')}")
                    
                    # Quick Stats
                    st.subheader("Quick Stats")
                    quick_stats_cols = st.columns(3)
                    
                    with quick_stats_cols[0]:
                        current_price = info.get('currentPrice', stock.history(period='1d')['Close'].iloc[-1])
                        st.metric("Current Price", f"${current_price:.2f}")
                    
                    with quick_stats_cols[1]:
                        st.metric("Market Cap", info.get('Market Cap', 'N/A'))
                    
                    with quick_stats_cols[2]:
                        st.metric("Volume", info.get('Volume', 'N/A'))
                    
                    # Key Statistics
                    st.subheader("Key Statistics")
                    key_stats_cols = st.columns(3)
                    
                    with key_stats_cols[0]:
                        st.metric("P/E Ratio", info.get('P/E Ratio', 'N/A'))
                        st.metric("Beta", info.get('Beta', 'N/A'))
                    
                    with key_stats_cols[1]:
                        st.metric("52 Week High", info.get('52 Week High', 'N/A'))
                        st.metric("52 Week Low", info.get('52 Week Low', 'N/A'))
                    
                    with key_stats_cols[2]:
                        st.metric("Dividend Yield", info.get('Dividend Yield', 'N/A'))
                        st.metric("Profit Margin", info.get('Profit Margin', 'N/A'))
                    
                    # Company Description
                    if 'longBusinessSummary' in info:
                        st.subheader("Company Description")
                        st.write(info['longBusinessSummary'])
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
                    
                    # Technical Analysis Summary
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
                                '1_month': 'Next Month'
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
