import numpy as np
from datetime import datetime, timedelta
import yfinance as yf

def calculate_returns(prices):
    """Calculate returns from price series"""
    return np.log(prices / prices.shift(1))

def calculate_volatility(returns, window=20):
    """Calculate rolling volatility"""
    return returns.rolling(window=window).std() * np.sqrt(252)

def calculate_sharpe_ratio(returns, risk_free_rate=0.01):
    """Calculate Sharpe Ratio"""
    excess_returns = returns - risk_free_rate/252
    return np.sqrt(252) * excess_returns.mean() / returns.std()

def calculate_drawdown(prices):
    """Calculate drawdown series"""
    rolling_max = prices.expanding().max()
    drawdown = prices / rolling_max - 1
    return drawdown

def calculate_max_drawdown(prices):
    """Calculate maximum drawdown"""
    drawdown = calculate_drawdown(prices)
    return drawdown.min()

def calculate_beta(stock_returns, market_returns):
    """Calculate beta relative to market"""
    covariance = np.cov(stock_returns, market_returns)[0][1]
    market_variance = np.var(market_returns)
    return covariance / market_variance

def get_market_data(start_date=None, end_date=None):
    """Get market (S&P 500) data"""
    if not start_date:
        start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
    if not end_date:
        end_date = datetime.now().strftime('%Y-%m-%d')
        
    spy = yf.download('^GSPC', start=start_date, end=end_date)
    return spy['Adj Close']

def format_large_number(number):
    """Format large numbers for display"""
    if number >= 1_000_000_000:
        return f"{number/1_000_000_000:.1f}B"
    elif number >= 1_000_000:
        return f"{number/1_000_000:.1f}M"
    elif number >= 1_000:
        return f"{number/1_000:.1f}K"
    return f"{number:.1f}"

def format_percentage(number):
    """Format percentage for display"""
    return f"{number:.2f}%"

def calculate_position_metrics(entry_price, current_price, shares):
    """Calculate position metrics"""
    position_value = shares * current_price
    cost_basis = shares * entry_price
    profit_loss = position_value - cost_basis
    profit_loss_percent = (profit_loss / cost_basis) * 100 if cost_basis > 0 else 0
    
    return {
        'position_value': position_value,
        'cost_basis': cost_basis,
        'profit_loss': profit_loss,
        'profit_loss_percent': profit_loss_percent
    }

def validate_stock_symbol(symbol):
    """Validate if a stock symbol exists"""
    try:
        stock = yf.Ticker(symbol)
        info = stock.info
        return 'regularMarketPrice' in info
    except:
        return False

def get_trading_signals(technical_indicators):
    """Generate trading signals based on technical indicators"""
    signals = []
    
    # RSI signals
    rsi = technical_indicators.get('RSI', 50)
    if rsi < 30:
        signals.append(('RSI', 'Oversold', 'Buy'))
    elif rsi > 70:
        signals.append(('RSI', 'Overbought', 'Sell'))
        
    # MACD signals
    macd = technical_indicators.get('MACD', 0)
    macd_signal = technical_indicators.get('MACD_Signal', 0)
    if macd > macd_signal:
        signals.append(('MACD', 'Bullish Crossover', 'Buy'))
    elif macd < macd_signal:
        signals.append(('MACD', 'Bearish Crossover', 'Sell'))
        
    # Moving Average signals
    price = technical_indicators.get('Close', 0)
    sma20 = technical_indicators.get('SMA20', 0)
    if price > sma20:
        signals.append(('SMA20', 'Price Above MA', 'Buy'))
    elif price < sma20:
        signals.append(('SMA20', 'Price Below MA', 'Sell'))
        
    return signals

def calculate_risk_metrics(portfolio_data):
    """Calculate portfolio risk metrics"""
    try:
        # Get returns for all positions
        returns = []
        weights = []
        total_value = sum(p['position_value'] for p in portfolio_data['positions'])
        
        for position in portfolio_data['positions']:
            stock = yf.Ticker(position['symbol'])
            hist = stock.history(period='1y')
            returns.append(calculate_returns(hist['Close']))
            weights.append(position['position_value'] / total_value)
            
        # Convert to numpy arrays
        returns = np.array(returns)
        weights = np.array(weights)
        
        # Calculate portfolio metrics
        portfolio_return = np.sum(returns.mean() * weights) * 252
        portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(np.cov(returns) * 252, weights)))
        sharpe_ratio = portfolio_return / portfolio_vol
        
        # Get correlation matrix
        correlation_matrix = np.corrcoef(returns)
        
        return {
            'portfolio_return': portfolio_return,
            'portfolio_volatility': portfolio_vol,
            'sharpe_ratio': sharpe_ratio,
            'correlation_matrix': correlation_matrix.tolist()
        }
    except Exception as e:
        return {
            'error': str(e)
        }
