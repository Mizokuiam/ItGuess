import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime, timedelta
import pandas as pd
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

# Sidebar
with st.sidebar:
    st.header("Settings")
    symbol = st.text_input("Enter Stock Symbol (e.g., AAPL)", value="AAPL").upper()
    period = st.selectbox(
        "Prediction Period",
        ["1d", "1w", "1m", "3m", "6m", "1y"],
        index=0
    )

# Main content
if symbol:
    # Show loading message while fetching data
    with st.spinner(f'Fetching data for {symbol}...'):
        try:
            # Get stock data
            @st.cache_data(ttl=300)  # Cache for 5 minutes
            def get_stock_data(symbol):
                stock = yf.Ticker(symbol)
                info = stock.info
                hist = stock.history(period="1y")
                return stock, info, hist

            stock, info, hist = get_stock_data(symbol)
            
            if not hist.empty and len(hist) > 20:  # Ensure we have enough data
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
                
                # Technical Analysis
                with st.spinner('Calculating technical indicators...'):
                    st.header("Technical Analysis")
                    @st.cache_data(ttl=300)
                    def get_indicators(hist_data):
                        return technical_analysis.calculate_indicators(hist_data)
                    
                    indicators = get_indicators(hist)
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("RSI", f"{indicators.get('RSI', 0):.2f}")
                    with col2:
                        st.metric("MACD", f"{indicators.get('MACD', 0):.2f}")
                    with col3:
                        st.metric("EMA (20)", f"${indicators.get('EMA20', 0):.2f}")
                    with col4:
                        st.metric("EMA (50)", f"${indicators.get('EMA50', 0):.2f}")
                
                # Charts
                st.header("Price Charts")
                tab1, tab2, tab3 = st.tabs(["Price", "Technical", "Prediction"])
                
                with tab1:
                    fig = go.Figure()
                    fig.add_trace(go.Candlestick(
                        x=hist.index,
                        open=hist['Open'],
                        high=hist['High'],
                        low=hist['Low'],
                        close=hist['Close'],
                        name="OHLC"
                    ))
                    fig.update_layout(
                        title="Stock Price History",
                        xaxis_title="Date",
                        yaxis_title="Price ($)",
                        hovermode='x unified'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with tab2:
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=hist.index,
                        y=hist['Close'],
                        name="Price"
                    ))
                    fig.add_trace(go.Scatter(
                        x=hist.index,
                        y=hist['Close'].ewm(span=20).mean(),
                        name="EMA 20"
                    ))
                    fig.add_trace(go.Scatter(
                        x=hist.index,
                        y=hist['Close'].ewm(span=50).mean(),
                        name="EMA 50"
                    ))
                    
                    # Add Bollinger Bands
                    bb_upper = hist['Close'].rolling(window=20).mean() + (hist['Close'].rolling(window=20).std() * 2)
                    bb_lower = hist['Close'].rolling(window=20).mean() - (hist['Close'].rolling(window=20).std() * 2)
                    fig.add_trace(go.Scatter(
                        x=hist.index,
                        y=bb_upper,
                        name="BB Upper",
                        line=dict(dash='dash')
                    ))
                    fig.add_trace(go.Scatter(
                        x=hist.index,
                        y=bb_lower,
                        name="BB Lower",
                        line=dict(dash='dash')
                    ))
                    
                    fig.update_layout(
                        title="Technical Analysis",
                        xaxis_title="Date",
                        yaxis_title="Price ($)",
                        hovermode='x unified'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
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
                    
                    with tab3:
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
            
            else:
                st.error(f"Insufficient data available for {symbol}. Please ensure the stock symbol is correct and has enough trading history.")
                
        except Exception as e:
            st.error(f"Error fetching data for {symbol}: {str(e)}")
            st.info("Please enter a valid stock symbol. Example: AAPL for Apple Inc.")
