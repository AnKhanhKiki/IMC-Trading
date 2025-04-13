import pandas as pd
from dataclasses import dataclass
import math
import statistics
from typing import Dict, List, Tuple
import string
import jsonpickle

@dataclass
class Order:
    symbol: str
    price: int
    quantity: int

class OrderDepth:
    def __init__(self):
        self.buy_orders: Dict[int, int] = {}  
        self.sell_orders: Dict[int, int] = {}  

class Listing:
    def __init__(self, symbol, product, denomination):
        self.symbol = symbol
        self.product = product
        self.denomination = denomination

class Observation:
    def __init__(self, plain_value_observations, transport_fees):
        self.plain_value_observations = plain_value_observations
        self.transport_fees = transport_fees

class Trade:
    def __init__(self, symbol: str, price: int, quantity: int, 
                 buyer: str = None, seller: str = None, timestamp: int = 0):
        self.symbol = symbol
        self.price = price
        self.quantity = quantity
        self.buyer = buyer
        self.seller = seller
        self.timestamp = timestamp

class TradingState:
    def __init__(self, traderData, timestamp, listings, order_depths, own_trades, market_trades, position, observations):
        self.traderData = traderData
        self.timestamp = timestamp
        self.listings = listings
        self.order_depths = order_depths
        self.own_trades = own_trades
        self.market_trades = market_trades
        self.position = position
        self.observations = observations

class Trader:
    def __init__(self):
        self.position_limit = 50  # Maximum position allowed
        self.lookback = 18        # SMA period length
        
        # Initialize data storage
        self.trader_data = {
            "price_history": []    # List of mid prices
        }

    def _get_mid_price(self, order_depth) -> float:
        """Calculate current mid price from best bid/ask"""
        best_bid = max(order_depth.buy_orders.keys(), default=0)
        best_ask = min(order_depth.sell_orders.keys(), default=0)
        if best_bid and best_ask:
            return (best_bid + best_ask) / 2
        return None

    def _calculate_sma(self) -> float:
        """Calculate 18-period Simple Moving Average"""
        if len(self.trader_data["price_history"]) < self.lookback:
            return None
        return sum(self.trader_data["price_history"][-self.lookback:]) / self.lookback

    def run(self, state: TradingState) -> Tuple[Dict[str, List[Order]], int, str]:
        result = {"SQUID_INK": []}
        current_pos = state.position.get("SQUID_INK", 0)
        
        # Load historical data
        if state.traderData:
            self.trader_data = jsonpickle.decode(state.traderData)
            # Handle legacy DataFrame format
            if isinstance(self.trader_data.get("price_history"), pd.DataFrame):
                self.trader_data["price_history"] = self.trader_data["price_history"]['mid_price'].tolist()

        # Update price history
        squid_od = state.order_depths.get("SQUID_INK", OrderDepth())
        current_mid = self._get_mid_price(squid_od)
        
        if current_mid is not None:
            self.trader_data["price_history"].append(current_mid)
            # Maintain rolling 18-period window
            if len(self.trader_data["price_history"]) > self.lookback:
                self.trader_data["price_history"] = self.trader_data["price_history"][-self.lookback:]

        # Get current market prices
        best_bid = max(squid_od.buy_orders.keys(), default=0)
        best_ask = min(squid_od.sell_orders.keys(), default=0)

        # Handle existing positions
        if current_pos != 0:
            if current_pos > 0:  # Close long position
                # Price aggressively at best bid +1 to ensure fill
                close_price = best_bid + 1 if best_bid else int(current_mid)
                result["SQUID_INK"].append(Order("SQUID_INK", close_price, -current_pos))
            else:  # Close short position
                # Price aggressively at best ask -1 to ensure fill
                close_price = best_ask - 1 if best_ask else int(current_mid)
                result["SQUID_INK"].append(Order("SQUID_INK", close_price, -current_pos))
        else:
            # SMA Strategy - only trade if we have sufficient history
            if len(self.trader_data["price_history"]) >= self.lookback:
                sma = self._calculate_sma()
                current_mid = self.trader_data["price_history"][-1]
                
                if sma and current_mid:
                    if sma < current_mid:  # SELL signal
                        # Price at best bid to ensure immediate execution
                        if best_bid > 0:
                            ask_price = best_bid
                        else:
                            ask_price = int(current_mid)
                        result["SQUID_INK"].append(Order("SQUID_INK", ask_price, -self.position_limit))
                    elif sma > current_mid:  # BUY signal
                        # Price at best ask to ensure immediate execution
                        if best_ask > 0:
                            bid_price = best_ask
                        else:
                            bid_price = int(current_mid)
                        result["SQUID_INK"].append(Order("SQUID_INK", bid_price, self.position_limit))

        # Persist data and return results
        state.traderData = jsonpickle.encode(self.trader_data)
        return result, 0, ""