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
                'description': "MACD is a trend-following momentum indicator that shows the relationship between two moving averages of a security's price.",
                'interpretation': [
                    'MACD crosses above signal line: Bullish signal',
                    'MACD crosses below signal line: Bearish signal',
                    'MACD above zero: Upward trend',
                    'MACD below zero: Downward trend'
                ],
                'calculation': 'MACD = 12-period EMA - 26-period EMA\nSignal Line = 9-period EMA of MACD',
                'usage': 'Used to identify trend direction, momentum, and potential entry/exit points.'
            },
            'BB': {
                'name': 'Bollinger Bands',
                'description': 'Bollinger Bands are volatility bands placed above and below a moving average, consisting of a middle band (SMA) and two outer bands.',
                'interpretation': [
                    'Price near upper band: Potentially overbought',
                    'Price near lower band: Potentially oversold',
                    'Bands squeezing: Potential breakout incoming',
                    'Bands expanding: High volatility'
                ],
                'calculation': 'Middle Band = 20-day SMA\nUpper Band = Middle Band + (2 × 20-day std dev)\nLower Band = Middle Band - (2 × 20-day std dev)',
                'usage': 'Used to measure volatility and identify potential price breakouts.'
            },
            'SMA': {
                'name': 'Simple Moving Average',
                'description': 'A Simple Moving Average (SMA) calculates the average price over a specific period, smoothing price data to identify trends.',
                'interpretation': [
                    'Price above SMA: Upward trend',
                    'Price below SMA: Downward trend',
                    'SMA crossovers: Potential trend changes'
                ],
                'calculation': 'SMA = (P1 + P2 + ... + Pn) / n\nwhere P = Price and n = number of periods',
                'usage': 'Used to identify trend direction and support/resistance levels.'
            },
            'EMA': {
                'name': 'Exponential Moving Average',
                'description': 'An Exponential Moving Average (EMA) gives more weight to recent prices, making it more responsive to new information.',
                'interpretation': [
                    'Price above EMA: Upward trend',
                    'Price below EMA: Downward trend',
                    'EMA crossovers: Stronger trend change signals than SMA'
                ],
                'calculation': 'EMA = Price(t) × k + EMA(y) × (1 − k)\nwhere k = 2/(n + 1)',
                'usage': 'Used for trend following and generating trading signals.'
            },
            'STOCH': {
                'name': 'Stochastic Oscillator',
                'description': 'The Stochastic Oscillator is a momentum indicator comparing a closing price to its price range over time.',
                'interpretation': [
                    'Above 80: Overbought condition',
                    'Below 20: Oversold condition',
                    'Crossovers: Potential trading signals'
                ],
                'calculation': '%K = 100 × (C - L14)/(H14 - L14)\n%D = 3-period SMA of %K',
                'usage': 'Used to identify overbought and oversold conditions.'
            },
            'VOL': {
                'name': 'Volume',
                'description': 'Volume represents the total number of shares or contracts traded during a given period.',
                'interpretation': [
                    'High volume: Strong market movement',
                    'Low volume: Weak market movement',
                    'Volume precedes price: Volume often increases before significant price moves'
                ],
                'calculation': 'Volume = Total number of shares traded in a period',
                'usage': 'Used to confirm price movements and trend strength.'
            },
            'OBV': {
                'name': 'On-Balance Volume',
                'description': 'On-Balance Volume (OBV) measures buying and selling pressure by adding or subtracting volume based on price movement.',
                'interpretation': [
                    'Rising OBV: Buying pressure',
                    'Falling OBV: Selling pressure',
                    'Divergences: Potential trend reversals'
                ],
                'calculation': 'If Close > Close(prev): OBV = OBV(prev) + Volume\nIf Close < Close(prev): OBV = OBV(prev) - Volume',
                'usage': 'Used to confirm price trends and predict reversals.'
            },
            'SR': {
                'name': 'Support and Resistance',
                'description': 'Support and Resistance levels are price points where a stock has historically had difficulty falling below (support) or rising above (resistance).',
                'interpretation': [
                    'Price bouncing off support: Bullish signal',
                    'Price bouncing off resistance: Bearish signal',
                    'Level break: Potential trend continuation'
                ],
                'calculation': 'Based on historical price levels and market psychology',
                'usage': 'Used to identify potential entry and exit points, and price targets.'
            }
        }
        
        return indicators.get(indicator.upper(), {
            'error': 'Indicator information not found',
            'available_indicators': list(indicators.keys())
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
