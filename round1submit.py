import pandas as pd
from dataclasses import dataclass
import math
import statistics
from typing import Dict, List, Tuple
import json
from json import JSONEncoder
import jsonpickle
import string

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
        self.strategy_config = {
            "RAINFOREST_RESIN": {
                "vwap_lookback": 10,
                "spread_ratio": 0.409,  # OPTIMAL, DO NOT CHANGE
                "max_size": 12,        
                "inventory_penalty": 0.0005  
            },
            "KELP": {
                "sma_period": 18,     
                "deviation_threshold": 0.000005,  # OPTIMAL DO DO CHANGE
                "price_improvement": 0.005 
            },
            "SQUID_INK": {
                "ma_period": 8,  # Moving average period
                "deviation_threshold": 0.02,  
                "max_size": 50,  # Max size per order
            }
            }
        self.history = {
        "RAINFOREST_RESIN": {"prices": [], "spreads": []},
        "KELP": {"prices": [], "ema": None},
        "SQUID_INK": {"prices": []}
    }


    def _resin_strategy(self, order_depth, position, mid):
        cfg = self.strategy_config["RAINFOREST_RESIN"]
        orders = []
        
        # Proper spread calculation
        best_bid = max(order_depth.buy_orders.keys(), default=mid)
        best_ask = min(order_depth.sell_orders.keys(), default=mid)
        spread = best_ask - best_bid
        
        # VWAP calculation
        vwap = self._calculate_vwap(order_depth)
        
        # Dynamic pricing
        bid_price = round(vwap - spread * cfg["spread_ratio"])
        ask_price = round(vwap + spread * cfg["spread_ratio"])
        
        # Inventory adjustment
        pos_adj = cfg["inventory_penalty"] * position
        bid_price = int(bid_price - pos_adj)
        ask_price = int(ask_price - pos_adj)
        
        # Order sizing
        bid_size = min(cfg["max_size"], 50 - position)
        ask_size = min(cfg["max_size"], 50 + position)
        
        # Place orders with price validation
        if bid_price >= best_ask:
            orders.append(Order("RAINFOREST_RESIN", best_ask, bid_size))
        elif bid_price > best_bid:
            orders.append(Order("RAINFOREST_RESIN", bid_price, bid_size))
            
        if ask_price <= best_bid:
            orders.append(Order("RAINFOREST_RESIN", best_bid, -ask_size))
        elif ask_price < best_ask:
            orders.append(Order("RAINFOREST_RESIN", ask_price, -ask_size))
            
        return orders

    def _kelp_strategy(self, position, mid, best_bid, best_ask):
        cfg = self.strategy_config["KELP"]
        orders = []

        if len(self.history["KELP"]["prices"]) >= cfg["sma_period"]:
            ema = self._calculate_ema("KELP", mid, cfg["sma_period"])
            deviation = (mid - ema) / ema

            spread = best_ask - best_bid
            buy_price = best_bid + spread * cfg["price_improvement"]
            sell_price = best_ask - spread * cfg["price_improvement"]

            buy_price = round(buy_price)
            sell_price = round(sell_price)

            if deviation < -cfg["deviation_threshold"]:
                size = min(25, 50 - position)
                orders.append(Order("KELP", buy_price, size))
            elif deviation > cfg["deviation_threshold"]:
                size = min(25, 50 + position)
                orders.append(Order("KELP", sell_price, -size))

        return orders
    
    def _squid_strategy(self, order_depth, position, mid, best_bid, best_ask):
        cfg = self.strategy_config["SQUID_INK"]
        orders = []
        prices = self.history["SQUID_INK"]["prices"]

        if len(prices) >= cfg["ma_period"]:
            # Calculate moving average (simple moving average for simplicity)
            moving_average = sum(prices[-cfg["ma_period"]:]) / cfg["ma_period"]

            # Calculate the deviation from the moving average
            deviation = (mid - moving_average) / moving_average

            # Determine if price has deviated enough to trigger a trade
            if deviation > cfg["deviation_threshold"]:
                # Price is significantly above moving average, go short
                size = min(cfg["max_size"], 50 + position)  # Avoid exceeding position limit
                if size > 0:
                    orders.append(Order("SQUID_INK", best_bid, -size))  # Sell order
                
            elif deviation < -cfg["deviation_threshold"]:
                # Price is significantly below moving average, go long
                size = min(cfg["max_size"], 50 - position)  # Avoid exceeding position limit
                if size > 0:
                    orders.append(Order("SQUID_INK", best_ask, size))  # Buy order

        return orders


    def _calculate_vwap(self, order_depth):
        total_volume = 0
        value_sum = 0
        
        for price, vol in order_depth.buy_orders.items():
            value_sum += price * abs(vol)
            total_volume += abs(vol)
            
        for price, vol in order_depth.sell_orders.items():
            value_sum += price * abs(vol)
            total_volume += abs(vol)
            
        return value_sum / total_volume if total_volume > 0 else 0
    
    def _calculate_ema(self, product: str, price: float, period: int):
        alpha = 2 / (period + 1)
        prev_ema = self.history[product]["ema"]
    
        if prev_ema is None:
            ema = price  # Initialize EMA with the first price
        else:
            ema = alpha * price + (1 - alpha) * prev_ema

        self.history[product]["ema"] = ema
        return ema


    def run(self, state: TradingState) -> Tuple[Dict[str, List[Order]], int, str]:
        result = {product: [] for product in self.strategy_config}
        
        for product in state.order_depths:
            if product not in self.strategy_config:
                continue
                
            od = state.order_depths[product]
            position = state.position.get(product, 0)
            
            # Update history
            best_bid = max(od.buy_orders.keys(), default=0)
            best_ask = min(od.sell_orders.keys(), default=0)
            mid = (best_bid + best_ask)/2 if best_bid and best_ask else 0
            self.history[product]["prices"].append(mid)
            
            if product == "RAINFOREST_RESIN":
                orders = self._resin_strategy(od, position, mid)
            elif product == "KELP":
                orders = self._kelp_strategy(position, mid, best_bid, best_ask)
            elif product == "SQUID_INK":
                orders = self._squid_strategy(od, position, mid, best_bid, best_ask)
            
            result[product] = orders
            
            # Maintain history limits
            self.history[product]["prices"] = self.history[product]["prices"][-100:]
            
        return result, 0, ""

trader = Trader()