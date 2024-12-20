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
                            price = info.get('regularMarketPrice', 0)
                            change = info.get('regularMarketChangePercent', 0)
                            st.metric(
                                "Current Price",
                                f"${price:.2f}" if price else "N/A",
                                f"{change:.2f}%" if change else "N/A"
                            )
                        except (KeyError, TypeError):
                            st.metric("Current Price", "N/A", "N/A")
                    
                    with col2:
                        try:
                            volume = info.get('volume', 0)
                            st.metric("Volume", f"{volume:,}" if volume else "N/A")
                        except (KeyError, TypeError):
                            st.metric("Volume", "N/A")
                    
                    with col3:
                        try:
                            market_cap = info.get('marketCap', 0)
                            st.metric("Market Cap", f"${market_cap:,}" if market_cap else "N/A")
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
                        
                        # Calculate indicators with custom periods
                        indicators = technical_analysis.calculate_indicators(hist)
                        
                        # Display indicators in columns
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("RSI", f"{indicators.get('RSI', 0):.2f}")
                        with col2:
                            st.metric("MACD", f"{indicators.get('MACD', 0):.2f}")
                        with col3:
                            st.metric("EMA (20)", f"${indicators.get('EMA20', 0):.2f}")
                        with col4:
                            st.metric("EMA (50)", f"${indicators.get('EMA50', 0):.2f}")
                        
                        # Technical Analysis Chart
                        fig = go.Figure()
                        
                        # Add price
                        fig.add_trace(go.Scatter(
                            x=hist.index,
                            y=hist['Close'],
                            name="Price",
                            line=dict(color='blue')
                        ))
                        
                        # Add EMAs
                        fig.add_trace(go.Scatter(
                            x=hist.index,
                            y=hist['Close'].ewm(span=20).mean(),
                            name="EMA 20",
                            line=dict(color='orange')
                        ))
                        fig.add_trace(go.Scatter(
                            x=hist.index,
                            y=hist['Close'].ewm(span=50).mean(),
                            name="EMA 50",
                            line=dict(color='red')
                        ))
                        
                        # Add Bollinger Bands
                        bb_period = 20
                        bb_std = 2
                        bb_middle = hist['Close'].rolling(window=bb_period).mean()
                        bb_std_dev = hist['Close'].rolling(window=bb_period).std()
                        bb_upper = bb_middle + (bb_std_dev * bb_std)
                        bb_lower = bb_middle - (bb_std_dev * bb_std)
                        
                        fig.add_trace(go.Scatter(
                            x=hist.index,
                            y=bb_upper,
                            name="BB Upper",
                            line=dict(color='gray', dash='dash')
                        ))
                        fig.add_trace(go.Scatter(
                            x=hist.index,
                            y=bb_lower,
                            name="BB Lower",
                            line=dict(color='gray', dash='dash')
                        ))
                        
                        fig.update_layout(
                            title="Technical Analysis Chart",
                            xaxis_title="Date",
                            yaxis_title="Price ($)",
                            hovermode='x unified'
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Volume Chart
                        fig_volume = go.Figure()
                        fig_volume.add_trace(go.Bar(
                            x=hist.index,
                            y=hist['Volume'],
                            name="Volume"
                        ))
                        fig_volume.update_layout(
                            title="Trading Volume",
                            xaxis_title="Date",
                            yaxis_title="Volume",
                            hovermode='x unified'
                        )
                        st.plotly_chart(fig_volume, use_container_width=True)
                
                with tab3:
                    # Price Prediction
                    with st.spinner('Generating predictions...'):
                        st.header("Price Predictions")
                        
                        @st.cache_data(ttl=300)
                        def get_stock_predictions(symbol, period):
                            return prediction_service.get_prediction(symbol, period)
                        
                        predictions = get_stock_predictions(symbol, period)
                        
                        col1, col2 = st.columns([2, 1])
                        with col1:
                            # Prediction results table
                            results_df = pd.DataFrame({
                                "Model": ["Ensemble", "LSTM", "Random Forest", "Linear Regression"],
                                "Prediction": [
                                    f"${predictions.get('ensemble', 0):.2f}",
                                    f"${predictions.get('lstm', 0):.2f}",
                                    f"${predictions.get('rf', 0):.2f}",
                                    f"${predictions.get('lr', 0):.2f}"
                                ],
                                "R² Score": [
                                    "-",
                                    f"{predictions.get('metrics', {}).get('lstm', {}).get('r2', 0):.3f}",
                                    f"{predictions.get('metrics', {}).get('rf', {}).get('r2', 0):.3f}",
                                    f"{predictions.get('metrics', {}).get('lr', {}).get('r2', 0):.3f}"
                                ]
                            })
                            st.dataframe(results_df, use_container_width=True)
                        
                        with col2:
                            # Prediction details
                            confidence = predictions.get('confidence', 0)
                            st.metric(
                                "Confidence",
                                f"{confidence}%" if confidence else "N/A"
                            )
                            st.write(f"Target Date: {predictions.get('date', 'N/A')}")
                            st.info(
                                "Note: Predictions are based on historical data and technical analysis. "
                                "Past performance does not guarantee future results."
                            )
                        
                        # Prediction chart
                        fig = go.Figure()
                        current_price = hist['Close'].iloc[-1]
                        
                        # Add prediction lines
                        for model, color in [
                            ("ensemble", "rgb(0,100,80)"),
                            ("lstm", "rgb(100,0,0)"),
                            ("rf", "rgb(0,0,100)"),
                            ("lr", "rgb(100,100,0)")
                        ]:
                            if model in predictions:
                                fig.add_trace(go.Scatter(
                                    x=[hist.index[-1], datetime.strptime(predictions['date'], '%Y-%m-%d')],
                                    y=[current_price, predictions[model]],
                                    name=model.upper(),
                                    line=dict(
                                        color=color,
                                        width=3 if model == "ensemble" else 1,
                                        dash='solid' if model == "ensemble" else 'dot'
                                    )
                                ))
                        
                        fig.update_layout(
                            title="Price Predictions",
                            xaxis_title="Date",
                            yaxis_title="Price ($)",
                            hovermode='x unified'
                        )
                        st.plotly_chart(fig, use_container_width=True)
                
                with tab4:
                    # Live Chart
                    st.header("Live Price Chart")
                    
                    # Get intraday data
                    @st.cache_data(ttl=60)  # Cache for 1 minute
                    def get_intraday_data(symbol):
                        return yf.download(symbol, period="1d", interval="1m")
                    
                    intraday_data = get_intraday_data(symbol)
                    
                    if not intraday_data.empty:
                        fig = go.Figure()
                        
                        # Candlestick chart
                        fig.add_trace(go.Candlestick(
                            x=intraday_data.index,
                            open=intraday_data['Open'],
                            high=intraday_data['High'],
                            low=intraday_data['Low'],
                            close=intraday_data['Close'],
                            name="OHLC"
                        ))
                        
                        fig.update_layout(
                            title="Live Intraday Price",
                            xaxis_title="Time",
                            yaxis_title="Price ($)",
                            hovermode='x unified'
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Display latest statistics
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric(
                                "Latest Price",
                                f"${intraday_data['Close'][-1]:.2f}",
                                f"{((intraday_data['Close'][-1] / intraday_data['Close'][0]) - 1) * 100:.2f}%"
                            )
                        with col2:
                            st.metric("Day High", f"${intraday_data['High'].max():.2f}")
                        with col3:
                            st.metric("Day Low", f"${intraday_data['Low'].min():.2f}")
                        
                        # Auto-refresh message
                        if auto_refresh:
                            st.info("Chart auto-refreshes every 5 minutes")
            
            else:
                st.error(f"Insufficient data available for {symbol}. Please ensure the stock symbol is correct and has enough trading history.")
                
        except Exception as e:
            st.error(f"Error fetching data for {symbol}: {str(e)}")
            st.info("Please enter a valid stock symbol. Example: AAPL for Apple Inc.")
