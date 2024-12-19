from models import db, Portfolio, Position, Watchlist, WatchlistItem
import yfinance as yf
from datetime import datetime

class PortfolioService:
    @staticmethod
    def create_portfolio(user_id, name):
        """Create a new portfolio"""
        try:
            portfolio = Portfolio(user_id=user_id, name=name)
            db.session.add(portfolio)
            db.session.commit()
            return True, portfolio
        except Exception as e:
            db.session.rollback()
            return False, str(e)
            
    @staticmethod
    def add_position(portfolio_id, symbol, shares, entry_price):
        """Add a position to portfolio"""
        try:
            position = Position(
                portfolio_id=portfolio_id,
                symbol=symbol,
                shares=shares,
                entry_price=entry_price
            )
            db.session.add(position)
            db.session.commit()
            return True, position
        except Exception as e:
            db.session.rollback()
            return False, str(e)
            
    @staticmethod
    def update_position(position_id, shares=None, entry_price=None):
        """Update a position"""
        try:
            position = Position.query.get(position_id)
            if not position:
                return False, "Position not found"
                
            if shares is not None:
                position.shares = shares
            if entry_price is not None:
                position.entry_price = entry_price
                
            db.session.commit()
            return True, position
        except Exception as e:
            db.session.rollback()
            return False, str(e)
            
    @staticmethod
    def delete_position(position_id):
        """Delete a position"""
        try:
            position = Position.query.get(position_id)
            if not position:
                return False, "Position not found"
                
            db.session.delete(position)
            db.session.commit()
            return True, "Position deleted"
        except Exception as e:
            db.session.rollback()
            return False, str(e)
            
    @staticmethod
    def get_portfolio_value(portfolio_id):
        """Calculate current portfolio value"""
        try:
            portfolio = Portfolio.query.get(portfolio_id)
            if not portfolio:
                return False, "Portfolio not found"
                
            total_value = 0
            positions_data = []
            
            for position in portfolio.positions:
                # Get current price
                stock = yf.Ticker(position.symbol)
                current_price = stock.info.get('regularMarketPrice', 0)
                
                # Calculate position value
                position_value = position.shares * current_price
                total_value += position_value
                
                # Calculate profit/loss
                cost_basis = position.shares * position.entry_price
                profit_loss = position_value - cost_basis
                profit_loss_percent = (profit_loss / cost_basis) * 100 if cost_basis > 0 else 0
                
                positions_data.append({
                    'symbol': position.symbol,
                    'shares': position.shares,
                    'entry_price': position.entry_price,
                    'current_price': current_price,
                    'position_value': position_value,
                    'profit_loss': profit_loss,
                    'profit_loss_percent': profit_loss_percent
                })
                
            return True, {
                'total_value': total_value,
                'positions': positions_data
            }
        except Exception as e:
            return False, str(e)
            
class WatchlistService:
    @staticmethod
    def create_watchlist(user_id, name):
        """Create a new watchlist"""
        try:
            watchlist = Watchlist(user_id=user_id, name=name)
            db.session.add(watchlist)
            db.session.commit()
            return True, watchlist
        except Exception as e:
            db.session.rollback()
            return False, str(e)
            
    @staticmethod
    def add_symbol(watchlist_id, symbol):
        """Add a symbol to watchlist"""
        try:
            # Verify symbol exists
            stock = yf.Ticker(symbol)
            if 'regularMarketPrice' not in stock.info:
                return False, "Invalid symbol"
                
            item = WatchlistItem(watchlist_id=watchlist_id, symbol=symbol)
            db.session.add(item)
            db.session.commit()
            return True, item
        except Exception as e:
            db.session.rollback()
            return False, str(e)
            
    @staticmethod
    def remove_symbol(watchlist_id, symbol):
        """Remove a symbol from watchlist"""
        try:
            item = WatchlistItem.query.filter_by(
                watchlist_id=watchlist_id,
                symbol=symbol
            ).first()
            
            if not item:
                return False, "Symbol not found in watchlist"
                
            db.session.delete(item)
            db.session.commit()
            return True, "Symbol removed from watchlist"
        except Exception as e:
            db.session.rollback()
            return False, str(e)
            
    @staticmethod
    def get_watchlist_data(watchlist_id):
        """Get current data for all symbols in watchlist"""
        try:
            watchlist = Watchlist.query.get(watchlist_id)
            if not watchlist:
                return False, "Watchlist not found"
                
            symbols_data = []
            for item in watchlist.stocks:
                stock = yf.Ticker(item.symbol)
                info = stock.info
                
                symbols_data.append({
                    'symbol': item.symbol,
                    'price': info.get('regularMarketPrice', 0),
                    'change': info.get('regularMarketChangePercent', 0),
                    'volume': info.get('regularMarketVolume', 0),
                    'market_cap': info.get('marketCap', 0)
                })
                
            return True, symbols_data
        except Exception as e:
            return False, str(e)
