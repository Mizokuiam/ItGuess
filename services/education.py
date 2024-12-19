class EducationService:
    @staticmethod
    def get_indicator_info(indicator):
        """Get information about technical indicators"""
        indicators = {
            'RSI': {
                'name': 'Relative Strength Index',
                'description': 'The Relative Strength Index (RSI) is a momentum indicator that measures the magnitude of recent price changes to evaluate overbought or oversold conditions.',
                'interpretation': [
                    'RSI > 70: Potentially overbought',
                    'RSI < 30: Potentially oversold',
                    'RSI = 50: Neutral momentum'
                ],
                'calculation': 'RSI = 100 - (100 / (1 + RS))\nwhere RS = Average Gain / Average Loss',
                'usage': 'Used to identify potential reversal points and market conditions.'
            },
            'MACD': {
                'name': 'Moving Average Convergence Divergence',
                'description': 'MACD is a trend-following momentum indicator that shows the relationship between two moving averages of a security's price.',
                'interpretation': [
                    'MACD crosses above signal line: Bullish signal',
                    'MACD crosses below signal line: Bearish signal',
                    'MACD above zero: Upward trend',
                    'MACD below zero: Downward trend'
                ],
                'calculation': 'MACD = 12-period EMA - 26-period EMA\nSignal Line = 9-period EMA of MACD',
                'usage': 'Used to identify trend direction, momentum, and potential reversal points.'
            },
            'BB': {
                'name': 'Bollinger Bands',
                'description': 'Bollinger Bands are volatility bands placed above and below a moving average, consisting of a middle band (SMA) and two outer bands.',
                'interpretation': [
                    'Price near upper band: Potentially overbought',
                    'Price near lower band: Potentially oversold',
                    'Bands squeeze: Potential breakout coming',
                    'Bands expand: Increased volatility'
                ],
                'calculation': 'Middle Band = 20-day SMA\nUpper Band = Middle Band + (2 × 20-day standard deviation)\nLower Band = Middle Band - (2 × 20-day standard deviation)',
                'usage': 'Used to measure volatility and identify potential price breakouts.'
            },
            'SMA': {
                'name': 'Simple Moving Average',
                'description': 'A Simple Moving Average is the average price over a specific period, smoothing out price fluctuations to help identify trends.',
                'interpretation': [
                    'Price above SMA: Upward trend',
                    'Price below SMA: Downward trend',
                    'SMA crossovers: Potential trend changes'
                ],
                'calculation': 'SMA = (P1 + P2 + ... + Pn) / n\nwhere P = Price and n = number of periods',
                'usage': 'Used to identify trend direction and support/resistance levels.'
            }
        }
        
        return indicators.get(indicator, {
            'name': 'Unknown Indicator',
            'description': 'Information not available.',
            'interpretation': [],
            'calculation': '',
            'usage': ''
        })
        
    @staticmethod
    def get_pattern_info(pattern):
        """Get information about chart patterns"""
        patterns = {
            'head_and_shoulders': {
                'name': 'Head and Shoulders',
                'description': 'A reversal pattern consisting of three peaks, with the middle peak (head) being the highest and the two outer peaks (shoulders) being lower and roughly equal.',
                'significance': 'Typically signals a bearish reversal in an uptrend.',
                'identification': [
                    'Left shoulder: Initial peak and decline',
                    'Head: Higher peak and decline',
                    'Right shoulder: Lower peak similar to left shoulder',
                    'Neckline: Support line connecting the troughs'
                ],
                'confirmation': 'Pattern confirms when price breaks below the neckline.'
            },
            'double_top': {
                'name': 'Double Top',
                'description': 'A reversal pattern showing two consecutive peaks at approximately the same price level.',
                'significance': 'Signals a potential bearish reversal after an uptrend.',
                'identification': [
                    'Two peaks at similar price levels',
                    'Noticeable valley between peaks',
                    'Support level at the valley'
                ],
                'confirmation': 'Pattern confirms when price breaks below the support level.'
            },
            'double_bottom': {
                'name': 'Double Bottom',
                'description': 'A reversal pattern showing two consecutive troughs at approximately the same price level.',
                'significance': 'Signals a potential bullish reversal after a downtrend.',
                'identification': [
                    'Two troughs at similar price levels',
                    'Noticeable peak between troughs',
                    'Resistance level at the peak'
                ],
                'confirmation': 'Pattern confirms when price breaks above the resistance level.'
            }
        }
        
        return patterns.get(pattern, {
            'name': 'Unknown Pattern',
            'description': 'Information not available.',
            'significance': '',
            'identification': [],
            'confirmation': ''
        })
        
    @staticmethod
    def get_strategy_info(strategy):
        """Get information about trading strategies"""
        strategies = {
            'trend_following': {
                'name': 'Trend Following',
                'description': 'A strategy that aims to capitalize on the momentum of a market moving in a sustained direction.',
                'key_concepts': [
                    'Trade in the direction of the trend',
                    'Use moving averages for trend identification',
                    'Wait for pullbacks to enter positions',
                    'Use trailing stops to protect profits'
                ],
                'indicators_used': [
                    'Moving Averages',
                    'MACD',
                    'ADX'
                ],
                'risk_management': [
                    'Set stop losses below recent swing lows for long positions',
                    'Use position sizing based on volatility',
                    'Trail stops as trend progresses'
                ]
            },
            'mean_reversion': {
                'name': 'Mean Reversion',
                'description': 'A strategy based on the theory that prices tend to return to their average over time.',
                'key_concepts': [
                    'Identify overbought/oversold conditions',
                    'Wait for price to deviate significantly from moving average',
                    'Enter trades when price starts returning to mean',
                    'Take profit at or near the mean'
                ],
                'indicators_used': [
                    'RSI',
                    'Bollinger Bands',
                    'Stochastic Oscillator'
                ],
                'risk_management': [
                    'Set tight stops above/below extreme levels',
                    'Use smaller position sizes due to counter-trend nature',
                    'Take partial profits as price approaches mean'
                ]
            }
        }
        
        return strategies.get(strategy, {
            'name': 'Unknown Strategy',
            'description': 'Information not available.',
            'key_concepts': [],
            'indicators_used': [],
            'risk_management': []
        })
