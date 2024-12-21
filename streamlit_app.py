import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from services.technical_analysis import TechnicalAnalysisService
from services.prediction import PredictionService
from plotly.subplots import make_subplots

# Page config
st.set_page_config(
    page_title="ItGuess - Stock Price Predictor",
    page_icon="🎯",
    layout="wide"
)

# Custom CSS for modern minimalist design
st.markdown("""
    <style>
        /* Modern color scheme */
        :root {
            --primary-color: #1E88E5;
            --background-color: #FFFFFF;
            --text-color: #333333;
            --accent-color: #E3F2FD;
        }
        
        /* Main title styling */
        .main-title {
            color: var(--text-color);
            font-family: 'SF Pro Display', -apple-system, BlinkMacSystemFont, sans-serif;
            font-weight: 700;
            font-size: 2.5rem;
            margin-bottom: 0.5rem;
            letter-spacing: -0.5px;
            background: linear-gradient(135deg, #1E88E5, #64B5F6);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            text-shadow: 2px 2px 4px rgba(30, 136, 229, 0.1);
        }
        
        .main-title span.highlight {
            color: var(--primary-color);
            font-weight: 800;
        }
        
        /* Logo styling */
        .logo-container {
            display: flex;
            align-items: center;
            gap: 12px;
            margin-bottom: 1rem;
            padding: 0.5rem;
            border-radius: 16px;
        }
        
        .logo {
            background: linear-gradient(135deg, var(--primary-color), #64B5F6);
            color: white;
            width: 48px;
            height: 48px;
            display: flex;
            align-items: center;
            justify-content: center;
            border-radius: 12px;
            font-size: 24px;
            font-weight: 600;
            box-shadow: 0 2px 8px rgba(30, 136, 229, 0.2);
            font-family: 'SF Pro Display', -apple-system, BlinkMacSystemFont, sans-serif;
        }
        
        .subtitle {
            color: #666;
            font-size: 1.1rem;
            font-weight: 400;
            margin-bottom: 2rem;
            letter-spacing: 0.2px;
        }
        
        /* Sidebar styling */
        .sidebar-logo {
            background: linear-gradient(135deg, var(--primary-color), #64B5F6);
            color: white;
            width: 36px;
            height: 36px;
            display: flex;
            align-items: center;
            justify-content: center;
            border-radius: 8px;
            font-size: 18px;
            font-weight: 600;
            margin-bottom: 1rem;
            box-shadow: 0 2px 4px rgba(30, 136, 229, 0.15);
        }
        
        .sidebar-title {
            background: linear-gradient(135deg, #1E88E5, #64B5F6);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-weight: 700;
            margin-bottom: 1rem;
        }
    </style>
""", unsafe_allow_html=True)

# Title with modern logo
st.markdown("""
    <div class="logo-container">
        <div class="logo">IG</div>
        <h1 class="main-title">It<span class="highlight">Guess</span></h1>
    </div>
    <p class="subtitle">Predict stock prices using machine learning and technical analysis</p>
    """, unsafe_allow_html=True)

# Initialize services
@st.cache_resource
def get_services():
    return TechnicalAnalysisService(), PredictionService()

technical_analysis, prediction_service = get_services()

# Initialize session state
if 'last_symbol' not in st.session_state:
    st.session_state.last_symbol = ''
if 'last_update' not in st.session_state:
    st.session_state.last_update = datetime.now()

# Sidebar with matching design
with st.sidebar:
    st.markdown("""
        <div style="display: flex; align-items: center; gap: 8px;">
            <div class="sidebar-logo">IG</div>
            <h3 class="sidebar-title">ItGuess</h3>
        </div>
    """, unsafe_allow_html=True)
    st.header("Settings")
    symbol = st.text_input("Enter Stock Symbol (e.g., AAPL)", value="AAPL").upper()
    period = st.selectbox(
        "Prediction Period",
        ["1d", "1w", "1m", "3m", "6m", "1y"],
        index=0
    )
    
    # Add technical analysis settings
    st.subheader("Technical Analysis Settings")
    rsi_period = st.slider("RSI Period", min_value=1, max_value=21, value=14)
    ma_period = st.slider("Moving Average Period", min_value=1, max_value=50, value=20)
    
    # Add auto-refresh option
    auto_refresh = st.checkbox("Auto-refresh data (5min)", value=True)
    if auto_refresh and (datetime.now() - st.session_state.last_update).seconds > 300:
        st.session_state.last_update = datetime.now()
        st.rerun()

# Main content
if symbol:
    # Show loading message while fetching data
    with st.spinner(f'Fetching data for {symbol}...'):
        try:
            # Get stock data
            @st.cache_data(ttl=300)  # Cache for 5 minutes
            def get_stock_data(symbol):
                stock = yf.Ticker(symbol)
                info = dict(stock.info)  # Convert info to a regular dictionary
                hist = stock.history(period="1y")
                return info, hist

            # Get data
            info, hist = get_stock_data(symbol)
            
            if not hist.empty and len(hist) > 20:  # Ensure we have enough data
                # Create tabs for different sections
                tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Technical Analysis", "Price Prediction", "Live Chart"])
                
                with tab1:
                    # Display current price and company info
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        try:
                            price = hist['Close'].iloc[-1] if not hist.empty else None
                            change = ((price / hist['Close'].iloc[0] - 1) * 100) if not hist.empty and price else None
                            st.metric(
                                "Current Price",
                                f"${price:.2f}" if price else "N/A",
                                f"{change:.2f}%" if change else "N/A"
                            )
                        except (KeyError, TypeError, IndexError):
                            st.metric("Current Price", "N/A", "N/A")
                    
                    with col2:
                        try:
                            volume = hist['Volume'].iloc[-1] if not hist.empty else None
                            st.metric("Volume", f"{int(volume):,}" if volume else "N/A")
                        except (KeyError, TypeError, IndexError):
                            st.metric("Volume", "N/A")
                    
                    with col3:
                        try:
                            market_cap = info.get('marketCap', None)
                            st.metric("Market Cap", f"${int(market_cap):,}" if market_cap else "N/A")
                        except (KeyError, TypeError):
                            st.metric("Market Cap", "N/A")
                    
                    # Company information
                    st.subheader("Company Information")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("**Sector:**", info.get('sector', 'N/A'))
                        st.write("**Industry:**", info.get('industry', 'N/A'))
                        st.write("**Website:**", info.get('website', 'N/A'))
                    with col2:
                        st.write("**52 Week High:**", f"${info.get('fiftyTwoWeekHigh', 0):.2f}")
                        st.write("**52 Week Low:**", f"${info.get('fiftyTwoWeekLow', 0):.2f}")
                        st.write("**Average Volume:**", f"{info.get('averageVolume', 0):,}")
                
                with tab2:
                    st.header("Technical Analysis")
                    
                    try:
                        # Initialize technical analysis with current symbol and data
                        technical_analysis.symbol = symbol
                        technical_analysis.data = hist.copy()  # Create a copy of the data
                        
                        # Calculate indicators with the current periods
                        rsi = technical_analysis.calculate_rsi(hist['Close'].values, rsi_period)
                        if rsi is not None:
                            indicators = technical_analysis.calculate_indicators(hist)
                            
                            if indicators and any(v != 'N/A' for v in indicators.values()):
                                # Display indicators in columns
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    rsi_val = indicators.get('RSI', 'N/A')
                                    st.metric("RSI", f"{rsi_val}")
                                    
                                    macd = indicators.get('MACD', 'N/A')
                                    st.metric("MACD", f"{macd}")
                                
                                with col2:
                                    ema20 = indicators.get('EMA20', 'N/A')
                                    st.metric("EMA 20", f"{ema20}")
                                    
                                    signal = indicators.get('Signal', 'N/A')
                                    st.metric("Signal", f"{signal}")
                                
                                with col3:
                                    ema50 = indicators.get('EMA50', 'N/A')
                                    st.metric("EMA 50", f"{ema50}")
                                    
                                    volume = indicators.get('Volume', 'N/A')
                                    st.metric("Volume", f"{volume:,}" if isinstance(volume, (int, float)) else 'N/A')
                            else:
                                st.error("Unable to calculate indicators with current settings")
                        else:
                            st.error("Not enough data to calculate indicators")
                    except Exception as e:
                        st.error(f"Error in technical analysis: {str(e)}")
                
                with tab3:
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
                
                with tab4:
                    # Live Chart with Technical Analysis
                    st.header("Live Chart")
                    
                    try:
                        # Calculate technical indicators
                        technical_analysis.data = hist.copy()
                        rsi_data = technical_analysis.calculate_rsi(hist['Close'].values, rsi_period)  
                        ema_short = technical_analysis.calculate_ema(hist['Close'], period=ma_period)
                        ema_long = technical_analysis.calculate_ema(hist['Close'], period=50)
                        bb_upper, bb_middle, bb_lower = technical_analysis.calculate_bollinger_bands(hist['Close'], period=20)
                        
                        if rsi_data is not None:
                            # Create figure with secondary y-axis
                            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                                          vertical_spacing=0.03, 
                                          row_heights=[0.7, 0.3])

                            # Add candlestick
                            fig.add_trace(go.Candlestick(x=hist.index,
                                                   open=hist['Open'],
                                                   high=hist['High'],
                                                   low=hist['Low'],
                                                   close=hist['Close'],
                                                   name='OHLC'),
                                    row=1, col=1)

                            # Add EMAs
                            if ema_short is not None:
                                fig.add_trace(go.Scatter(x=hist.index, y=ema_short,
                                               line=dict(color='orange', width=1),
                                               name=f'EMA {ma_period}'),
                                    row=1, col=1)
                            
                            if ema_long is not None:
                                fig.add_trace(go.Scatter(x=hist.index, y=ema_long,
                                               line=dict(color='blue', width=1),
                                               name='EMA 50'),
                                    row=1, col=1)

                            # Add Bollinger Bands
                            if all(x is not None for x in [bb_upper, bb_lower]):
                                fig.add_trace(go.Scatter(x=hist.index, y=bb_upper,
                                               line=dict(color='gray', width=1, dash='dash'),
                                               name='BB Upper'),
                                    row=1, col=1)
                                
                                fig.add_trace(go.Scatter(x=hist.index, y=bb_lower,
                                               line=dict(color='gray', width=1, dash='dash'),
                                               name='BB Lower',
                                               fill='tonexty'),
                                    row=1, col=1)

                            # Add RSI
                            fig.add_trace(go.Scatter(x=hist.index, y=rsi_data,
                                               line=dict(color='purple', width=1),
                                               name='RSI'),
                                    row=2, col=1)

                            # Add RSI levels
                            fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
                            fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)

                            # Update layout
                            fig.update_layout(
                                xaxis_rangeslider_visible=False,
                                height=800,
                                title_text=f"{symbol} Technical Analysis",
                                showlegend=True,
                                legend=dict(
                                    yanchor="top",
                                    y=0.99,
                                    xanchor="left",
                                    x=0.01
                                )
                            )

                            # Update y-axes labels
                            fig.update_yaxes(title_text="Price", row=1, col=1)
                            fig.update_yaxes(title_text="RSI", row=2, col=1)

                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.error("Not enough data to calculate technical indicators")
                    except Exception as e:
                        st.error(f"Error creating live chart: {str(e)}")
            
            else:
                st.error("Not enough historical data available for analysis")
        
        except Exception as e:
            st.error(f"Error fetching data: {str(e)}")
else:
    st.info("Please enter a valid stock symbol. Example: AAPL for Apple Inc.")
