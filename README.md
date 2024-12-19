# ItGuess - Stock Price Predictor

An advanced stock price prediction web application that uses machine learning models to forecast stock prices based on historical data and various technical indicators.

## Features

- Real-time stock data fetching using Yahoo Finance API
- Multiple prediction models:
  - Linear Regression
  - Random Forest
  - LSTM (Long Short-Term Memory)
- Technical indicators:
  - Moving Averages (20-day, 50-day)
  - RSI (Relative Strength Index)
  - Bollinger Bands
  - EMA (Exponential Moving Average)
- Interactive web interface with real-time updates
- Historical price visualization
- Model performance metrics
- Automated data updates

## Installation

1. Clone the repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Start the Flask server:
```bash
python app.py
```
2. Open your browser and navigate to `http://localhost:5000`

## Disclaimer

This tool is for educational and research purposes only. Stock predictions are based on historical data and technical analysis, but past performance does not guarantee future results. Do not use this tool as your sole source for financial decisions.

## Technical Architecture

- Backend: Flask
- Data Source: Yahoo Finance API
- ML Framework: Scikit-learn, TensorFlow
- Data Processing: Pandas, NumPy
- Visualization: Plotly, Matplotlib
- Scheduling: Schedule library

## Project Structure

```
itguess/
├── static/          # Static files (CSS, JS)
├── templates/       # HTML templates
├── models/          # ML model files
├── data/           # Data storage
├── utils/          # Utility functions
├── app.py          # Main application
├── config.py       # Configuration
└── requirements.txt
```
