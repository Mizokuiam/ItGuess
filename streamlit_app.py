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
        <div class="features-container">
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
{{ ... }}
