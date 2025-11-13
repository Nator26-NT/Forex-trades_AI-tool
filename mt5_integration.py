# mt5_integration.py
import logging
import pandas as pd
from datetime import datetime
import time

# MetaTrader5 is optional - app will work without it
try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False
    print("MetaTrader5 package not available. Install with: pip install MetaTrader5")

class MT5TradingBridge:
    def __init__(self):
        self.connected = False
        self.mt5_available = MT5_AVAILABLE
        self.logger = self._setup_logger()
    
    def _setup_logger(self):
        logger = logging.getLogger('MT5Bridge')
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger
    
    def connect_mt5(self, server="", login=0, password=""):
        """Connect to MetaTrader 5 terminal"""
        if not self.mt5_available:
            self.logger.error("MetaTrader5 package not installed")
            return False
        
        try:
            if not mt5.initialize():
                self.logger.error("MT5 initialization failed - Make sure MetaTrader 5 is running")
                return False
            
            # If credentials provided, attempt login
            if login != 0 and password:
                if not mt5.login(login, password=password, server=server):
                    self.logger.error(f"MT5 login failed: {mt5.last_error()}")
                    mt5.shutdown()
                    return False
                self.logger.info(f"Successfully connected to MT5 account: {login}")
            else:
                self.logger.info("Connected to MT5 terminal (no login)")
            
            self.connected = True
            return True
            
        except Exception as e:
            self.logger.error(f"MT5 connection error: {str(e)}")
            return False
    
    def disconnect_mt5(self):
        """Disconnect from MT5"""
        if self.connected and self.mt5_available:
            mt5.shutdown()
            self.connected = False
            self.logger.info("Disconnected from MT5")
    
    def get_account_info(self):
        """Get current account information"""
        if not self.connected or not self.mt5_available:
            return {"error": "Not connected to MT5"}
        
        try:
            account_info = mt5.account_info()
            if account_info is None:
                return {"error": "Failed to get account info"}
            
            return {
                'login': account_info.login,
                'balance': account_info.balance,
                'equity': account_info.equity,
                'margin': account_info.margin,
                'free_margin': account_info.margin_free,
                'leverage': account_info.leverage,
                'currency': account_info.currency,
                'server': account_info.server
            }
        except Exception as e:
            return {"error": f"Error getting account info: {str(e)}"}
    
    def get_symbol_info(self, symbol):
        """Get symbol information"""
        if not self.connected or not self.mt5_available:
            return {"error": "Not connected to MT5"}
        
        try:
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info is None:
                return {"error": f"Symbol {symbol} not found"}
            
            return {
                'symbol': symbol_info.name,
                'bid': symbol_info.bid,
                'ask': symbol_info.ask,
                'spread': symbol_info.spread,
                'digits': symbol_info.digits,
                'trade_mode': symbol_info.trade_mode,
                'trade_contract_size': symbol_info.trade_contract_size,
                'volume_min': symbol_info.volume_min,
                'volume_max': symbol_info.volume_max,
                'volume_step': symbol_info.volume_step,
                'currency_base': symbol_info.currency_base,
                'currency_profit': symbol_info.currency_profit,
                'currency_margin': symbol_info.currency_margin
            }
        except Exception as e:
            return {"error": f"Error getting symbol info: {str(e)}"}
    
    def calculate_position_size(self, symbol, risk_percent, stop_loss_pips, account_balance=None):
        """Calculate position size based on risk management"""
        if not self.connected or not self.mt5_available:
            return {"error": "Not connected to MT5"}
        
        try:
            # Get account balance
            if account_balance is None:
                account_info = self.get_account_info()
                if 'error' in account_info:
                    return account_info
                account_balance = account_info['balance']
            
            # Get symbol info
            symbol_info = self.get_symbol_info(symbol)
            if 'error' in symbol_info:
                return symbol_info
            
            # Calculate risk amount
            risk_amount = account_balance * (risk_percent / 100)
            
            # Simplified pip value calculation
            pip_value = 10  # Default pip value for standard lot
            
            # Calculate position size (in lots)
            position_size = risk_amount / (stop_loss_pips * pip_value)
            
            # Adjust for MT5 lot size constraints
            lot_step = symbol_info.get('volume_step', 0.01)
            position_size = round(position_size / lot_step) * lot_step
            
            # Ensure minimum and maximum lot sizes
            min_lot = symbol_info.get('volume_min', 0.01)
            max_lot = symbol_info.get('volume_max', 100)
            position_size = max(min_lot, min(position_size, max_lot))
            
            return {
                'position_size': round(position_size, 2),
                'risk_amount': round(risk_amount, 2),
                'pip_value': pip_value,
                'lot_step': lot_step
            }
            
        except Exception as e:
            return {"error": f"Error calculating position size: {str(e)}"}
    
    def place_trade(self, symbol, signal_type, volume, stop_loss, take_profit, deviation=20):
        """Place a trade in MT5"""
        if not self.connected or not self.mt5_available:
            return {"error": "Not connected to MT5"}
        
        try:
            # Prepare the trade request
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info is None:
                return {"error": f"Invalid symbol: {symbol}"}
            
            if not symbol_info.visible:
                if not mt5.symbol_select(symbol, True):
                    return {"error": f"Failed to select symbol: {symbol}"}
            
            # Get current prices
            tick = mt5.symbol_info_tick(symbol)
            if tick is None:
                return {"error": f"Failed to get tick data for {symbol}"}
            
            # Determine order type and price
            if signal_type.upper() == 'BUY':
                order_type = mt5.ORDER_TYPE_BUY
                price = tick.ask
            elif signal_type.upper() == 'SELL':
                order_type = mt5.ORDER_TYPE_SELL
                price = tick.bid
            else:
                return {"error": f"Invalid signal type: {signal_type}"}
            
            # Create trade request
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": volume,
                "type": order_type,
                "price": price,
                "sl": stop_loss,
                "tp": take_profit,
                "deviation": deviation,
                "magic": 234000,
                "comment": "AI Trading Signal",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            
            # Send trade request
            result = mt5.order_send(request)
            
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                return {
                    "error": f"Trade failed: {result.retcode} - {self.get_error_description(result.retcode)}",
                    "retcode": result.retcode
                }
            
            return {
                "success": True,
                "order": result.order,
                "deal": result.deal,
                "volume": result.volume,
                "price": result.price,
                "bid": result.bid,
                "ask": result.ask,
                "comment": result.comment
            }
            
        except Exception as e:
            return {"error": f"Error placing trade: {str(e)}"}
    
    def get_error_description(self, retcode):
        """Get human-readable error description"""
        error_descriptions = {
            10004: "Requote",
            10006: "Request rejected",
            10007: "Request canceled by trader",
            10008: "Order placed",
            10009: "Request completed",
            10010: "Only part of the request was completed",
            10011: "Request processing error",
            10012: "Request canceled by timeout",
            10013: "Invalid request",
            10014: "Invalid volume in the request",
            10015: "Invalid price in the request",
            10016: "Invalid stops in the request",
            10017: "Trade is disabled",
            10018: "Market is closed",
            10019: "Insufficient funds",
            10020: "Prices changed",
            10021: "There are no quotes to process the request",
            10022: "Invalid order expiration date in the request",
            10023: "Order state changed",
            10024: "Too frequent requests",
            10025: "No changes in request",
            10026: "Autotrading disabled by server",
            10027: "Autotrading disabled by client terminal",
            10028: "Request locked for processing",
            10029: "Order or position frozen",
        }
        return error_descriptions.get(retcode, f"Unknown error: {retcode}")
    
    def get_open_positions(self, symbol=None):
        """Get open positions"""
        if not self.connected or not self.mt5_available:
            return {"error": "Not connected to MT5"}
        
        try:
            positions = mt5.positions_get(symbol=symbol) if symbol else mt5.positions_get()
            if positions is None:
                return []
            
            positions_list = []
            for position in positions:
                positions_list.append({
                    'ticket': position.ticket,
                    'symbol': position.symbol,
                    'type': 'BUY' if position.type == 0 else 'SELL',
                    'volume': position.volume,
                    'open_price': position.price_open,
                    'current_price': position.price_current,
                    'sl': position.sl,
                    'tp': position.tp,
                    'profit': position.profit,
                    'swap': position.swap,
                    'commission': position.commission
                })
            
            return positions_list
            
        except Exception as e:
            return {"error": f"Error getting positions: {str(e)}"}
    
    def close_position(self, ticket, deviation=20):
        """Close a position by ticket"""
        if not self.connected or not self.mt5_available:
            return {"error": "Not connected to MT5"}
        
        try:
            positions = mt5.positions_get(ticket=ticket)
            if not positions:
                return {"error": f"Position {ticket} not found"}
            
            position = positions[0]
            symbol = position.symbol
            
            # Get current tick
            tick = mt5.symbol_info_tick(symbol)
            if tick is None:
                return {"error": f"Failed to get tick data for {symbol}"}
            
            # Determine close price
            if position.type == 0:  # BUY position
                close_price = tick.bid
                order_type = mt5.ORDER_TYPE_SELL
            else:  # SELL position
                close_price = tick.ask
                order_type = mt5.ORDER_TYPE_BUY
            
            # Create close request
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "position": ticket,
                "symbol": symbol,
                "volume": position.volume,
                "type": order_type,
                "price": close_price,
                "deviation": deviation,
                "magic": 234000,
                "comment": "AI Close Signal",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            
            # Send close request
            result = mt5.order_send(request)
            
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                return {
                    "error": f"Close failed: {result.retcode} - {self.get_error_description(result.retcode)}",
                    "retcode": result.retcode
                }
            
            return {
                "success": True,
                "order": result.order,
                "deal": result.deal,
                "profit": position.profit
            }
            
        except Exception as e:
            return {"error": f"Error closing position: {str(e)}"}

# Global MT5 instance
mt5_bridge = MT5TradingBridge()