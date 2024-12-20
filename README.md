# ItGuess - Stock Price Predictor

A machine learning-powered stock price prediction application built with Streamlit. This application uses multiple models (LSTM, Random Forest, and Linear Regression) along with technical analysis to predict stock prices.

## Features

- Real-time stock data fetching using Yahoo Finance API
- Technical Analysis:
  - RSI (Relative Strength Index)
  - MACD (Moving Average Convergence Divergence)
  - EMA (Exponential Moving Average)
  - Bollinger Bands
- Multiple Prediction Models:
  - LSTM (Deep Learning)
  - Random Forest
  - Linear Regression
  - Ensemble (Weighted Average)
- Interactive Charts:
  - Price History
  - Technical Indicators
  - Price Predictions
- Model Performance Metrics
- Confidence Scores

## Live Demo

Visit [ItGuess on Streamlit Cloud](https://itguess.streamlit.app)

## Technology Stack

- Python 3.11+
- Streamlit (Web Framework)
- YFinance (Stock Data API)
- TensorFlow (LSTM Model)
- Scikit-learn (Random Forest & Linear Regression)
- Plotly (Interactive Charts)
- Pandas & NumPy (Data Processing)

## Local Development

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/itguess.git
   cd itguess
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the application:
   ```bash
   streamlit run streamlit_app.py
   ```

4. Open your browser and navigate to `http://localhost:8501`

## Project Structure

```
itguess/
├── streamlit_app.py      # Main Streamlit application
├── requirements.txt      # Project dependencies
├── README.md            # Project documentation
└── services/           # Application services
    ├── __init__.py
    ├── technical_analysis.py  # Technical analysis calculations
    └── prediction.py          # ML model predictions
```

## Deployment

This application is deployed on Streamlit Cloud. To deploy your own instance:

1. Fork this repository
2. Create an account on [Streamlit Cloud](https://streamlit.io/cloud)
3. Connect your GitHub account
4. Deploy the application from your forked repository

## Disclaimer

This application is for educational purposes only. Stock predictions are based on historical data and technical analysis, which may not accurately predict future stock prices. Always do your own research and consult with financial advisors before making investment decisions.

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
