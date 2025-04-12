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
        self.lookback = 18        # Number of periods for SMA
        
        # Initialize data storage
        self.trader_data = {
            "price_history": [],    # List of mid prices
            "pending_orders": {}    # Track unfilled orders
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
            # Convert DataFrame format from old versions to list
            if isinstance(self.trader_data.get("price_history"), pd.DataFrame):
                self.trader_data["price_history"] = self.trader_data["price_history"]['mid_price'].tolist()

        # Update price history
        squid_od = state.order_depths.get("SQUID_INK", OrderDepth())
        current_mid = self._get_mid_price(squid_od)
        
        if current_mid is not None:
            self.trader_data["price_history"].append(current_mid)
            # Maintain 18-period history
            if len(self.trader_data["price_history"]) > self.lookback:
                self.trader_data["price_history"] = self.trader_data["price_history"][-self.lookback:]

        # Handle existing position
        if current_pos != 0:
            best_bid = max(squid_od.buy_orders.keys(), default=0)
            best_ask = min(squid_od.sell_orders.keys(), default=0)
            pending = self.trader_data["pending_orders"].get("SQUID_INK", (None, 0))
            
            if current_pos > 0:  # Long position, need to sell
                target_price = best_bid + 1 if best_bid else current_mid - 1
                qty = -min(current_pos, self.position_limit)
                
                if pending[0] is not None:
                    result["SQUID_INK"].append(Order("SQUID_INK", pending[0], pending[1]))
                else:
                    result["SQUID_INK"].append(Order("SQUID_INK", target_price, qty))
                    self.trader_data["pending_orders"]["SQUID_INK"] = (target_price, qty)

            else:  # Short position, need to buy
                target_price = best_ask - 1 if best_ask else current_mid + 1
                qty = min(-current_pos, self.position_limit)
                
                if pending[0] is not None:
                    result["SQUID_INK"].append(Order("SQUID_INK", pending[0], pending[1]))
                else:
                    result["SQUID_INK"].append(Order("SQUID_INK", target_price, qty))
                    self.trader_data["pending_orders"]["SQUID_INK"] = (target_price, qty)

        # No position - SMA strategy
        else:
            if len(self.trader_data["price_history"]) >= self.lookback:
                sma = self._calculate_sma()
                current_mid = self.trader_data["price_history"][-1]
                
                if sma and current_mid:
                    if sma < current_mid:  # Sell signal
                        ask_price = current_mid + 1
                        result["SQUID_INK"].append(Order("SQUID_INK", ask_price, -self.position_limit))
                    elif sma > current_mid:  # Buy signal
                        bid_price = current_mid - 1
                        result["SQUID_INK"].append(Order("SQUID_INK", bid_price, self.position_limit))

        # Persist data for next iteration
        state.traderData = jsonpickle.encode(self.trader_data)
        return result, 0, ""