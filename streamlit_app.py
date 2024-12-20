import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from services.technical_analysis import TechnicalAnalysisService
from services.prediction import PredictionService

# Page config
st.set_page_config(
    page_title="ItGuess - Stock Price Predictor",
    page_icon="📈",
    layout="wide"
)

# Initialize services
@st.cache_resource
def get_services():
    return TechnicalAnalysisService(), PredictionService()

technical_analysis, prediction_service = get_services()

# Title
st.title("ItGuess - Stock Price Predictor")
st.markdown("Predict stock prices using machine learning and technical analysis")

# Initialize session state
if 'last_symbol' not in st.session_state:
    st.session_state.last_symbol = ''
if 'last_update' not in st.session_state:
    st.session_state.last_update = datetime.now()

# Sidebar
with st.sidebar:
    st.header("Settings")
    symbol = st.text_input("Enter Stock Symbol (e.g., AAPL)", value="AAPL").upper()
    period = st.selectbox(
        "Prediction Period",
        ["1d", "1w", "1m", "3m", "6m", "1y"],
        index=0
    )
    
    # Add technical analysis settings
    st.subheader("Technical Analysis Settings")
    rsi_period = st.slider("RSI Period", min_value=7, max_value=21, value=14)
    ma_period = st.slider("Moving Average Period", min_value=10, max_value=50, value=20)
    
    # Add auto-refresh option
    auto_refresh = st.checkbox("Auto-refresh data (5min)", value=True)
    if auto_refresh and (datetime.now() - st.session_state.last_update).seconds > 300:
        st.session_state.last_update = datetime.now()
        st.experimental_rerun()

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
                    # Technical Analysis
                    with st.spinner('Calculating technical indicators...'):
                        st.header("Technical Analysis")
                        
                        try:
                            # Initialize technical analysis with current symbol and data
                            technical_analysis.symbol = symbol
                            technical_analysis.data = hist.copy()  # Create a copy of the data
                            
                            # Calculate indicators
                            indicators = technical_analysis.calculate_indicators()
                            
                            if indicators:  # Check if we got valid indicators
                                # Display indicators in columns
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("RSI", str(indicators.get('RSI', 'N/A')))
                                    st.metric("MACD", str(indicators.get('MACD', 'N/A')))
                                with col2:
                                    st.metric("EMA (20)", str(indicators.get('EMA20', 'N/A')))
                                    st.metric("EMA (50)", str(indicators.get('EMA50', 'N/A')))
                                with col3:
                                    volume_sma = indicators.get('Volume_SMA', 'N/A')
                                    st.metric("Volume SMA", f"{volume_sma:,.0f}" if isinstance(volume_sma, (int, float)) else "N/A")
                                
                                # Add technical analysis chart
                                fig = go.Figure()
                                
                                # Add price line
                                fig.add_trace(go.Scatter(
                                    x=hist.index,
                                    y=hist['Close'],
                                    name="Price",
                                    line=dict(color='blue')
                                ))
                                
                                # Add EMAs if available
                                if 'EMA20' in indicators and 'EMA50' in indicators:
                                    fig.add_trace(go.Scatter(
                                        x=hist.index,
                                        y=hist['Close'].ewm(span=20, adjust=False).mean(),
                                        name="EMA 20",
                                        line=dict(color='orange')
                                    ))
                                    fig.add_trace(go.Scatter(
                                        x=hist.index,
                                        y=hist['Close'].ewm(span=50, adjust=False).mean(),
                                        name="EMA 50",
                                        line=dict(color='red')
                                    ))
                                
                                fig.update_layout(
                                    title="Technical Analysis Chart",
                                    xaxis_title="Date",
                                    yaxis_title="Price ($)",
                                    hovermode='x unified'
                                )
                                st.plotly_chart(fig, use_container_width=True)
                            else:
                                st.warning("Could not calculate technical indicators. Not enough data points.")
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
                    # Live Chart
                    with st.spinner('Loading live chart...'):
                        st.header("Live Chart")
                        
                        try:
                            # Create candlestick chart
                            fig = go.Figure(data=[go.Candlestick(
                                x=hist.index,
                                open=hist['Open'],
                                high=hist['High'],
                                low=hist['Low'],
                                close=hist['Close'],
                                name="OHLC"
                            )])
                            
                            fig.update_layout(
                                title=f'{symbol} Stock Price',
                                yaxis_title='Price ($)',
                                xaxis_title='Date',
                                hovermode='x unified'
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Display latest statistics
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                latest_close = hist['Close'].iloc[-1]
                                first_close = hist['Close'].iloc[0]
                                change = ((latest_close / first_close) - 1) * 100
                                st.metric("Latest Close", f"${latest_close:.2f}", f"{change:.1f}%")
                            with col2:
                                st.metric("Day High", f"${hist['High'].iloc[-1]:.2f}")
                            with col3:
                                st.metric("Day Low", f"${hist['Low'].iloc[-1]:.2f}")
                        except Exception as e:
                            st.error(f"Error creating live chart: {str(e)}")
            
            else:
                st.error("Not enough historical data available for analysis")
        
        except Exception as e:
            st.error(f"Error fetching data: {str(e)}")
else:
    st.info("Please enter a valid stock symbol. Example: AAPL for Apple Inc.")
