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
                "spread_ratio": 0.409,
                "max_size": 12,
                "inventory_penalty": 0.0005,
                "position_limit": 50  # Explicitly added based on strategy logic
            },
            "KELP": {
                "sma_period": 18,
                "deviation_threshold": 0.000005,
                "price_improvement": 0.005,
                "position_limit": 50  # Explicitly added based on strategy logic
            },
            "SQUID_INK": {
                "ma_period": 8,
                "deviation_threshold": 0.02,
                "max_size": 50,
                "position_limit": 50  # Explicitly added based on strategy logic
            },
            "PICNIC_BASKET1": {
                "components": {"CROISSANTS": 6, "JAMS": 3, "DJEMBES": 1},
                "position_limit": 60
            },
            "PICNIC_BASKET2": {
                "components": {"CROISSANTS": 4, "JAMS": 2},
                "position_limit": 100
            },
            "CROISSANTS": {"position_limit": 250},
            "JAMS": {"position_limit": 350},
            "DJEMBES": {"position_limit": 60}
        }
        self.history = {
            "RAINFOREST_RESIN": {"prices": [], "spreads": []},
            "KELP": {"prices": [], "ema": None},
            "SQUID_INK": {"prices": []},
            "CROISSANTS": {"prices": [], "best_bid": [], "best_ask": []},
            "JAMS": {"prices": [], "best_bid": [], "best_ask": []},
            "DJEMBES": {"prices": [], "best_bid": [], "best_ask": []},
            "PICNIC_BASKET1": {"prices": [], "arb_opportunities": []},
            "PICNIC_BASKET2": {"prices": [], "arb_opportunities": []}
        }

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
            ema = price
        else:
            ema = alpha * price + (1 - alpha) * prev_ema
        self.history[product]["ema"] = ema
        return ema

    def _resin_strategy(self, order_depth, position, mid):
        cfg = self.strategy_config["RAINFOREST_RESIN"]
        orders = []
        best_bid = max(order_depth.buy_orders.keys(), default=mid)
        best_ask = min(order_depth.sell_orders.keys(), default=mid)
        spread = best_ask - best_bid
        vwap = self._calculate_vwap(order_depth)
        bid_price = round(vwap - spread * cfg["spread_ratio"])
        ask_price = round(vwap + spread * cfg["spread_ratio"])
        pos_adj = cfg["inventory_penalty"] * position
        bid_price = int(bid_price - pos_adj)
        ask_price = int(ask_price - pos_adj)
        bid_size = min(cfg["max_size"], cfg["position_limit"] - position)
        ask_size = min(cfg["max_size"], cfg["position_limit"] + position)
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
                size = min(25, cfg["position_limit"] - position)
                orders.append(Order("KELP", buy_price, size))
            elif deviation > cfg["deviation_threshold"]:
                size = min(25, cfg["position_limit"] + position)
                orders.append(Order("KELP", sell_price, -size))
        return orders

    def _squid_strategy(self, order_depth, position, mid, best_bid, best_ask):
        cfg = self.strategy_config["SQUID_INK"]
        orders = []
        prices = self.history["SQUID_INK"]["prices"]
        if len(prices) >= cfg["ma_period"]:
            moving_average = sum(prices[-cfg["ma_period"]:]) / cfg["ma_period"]
            deviation = (mid - moving_average) / moving_average
            if deviation > cfg["deviation_threshold"]:
                size = min(cfg["max_size"], cfg["position_limit"] + position)
                if size > 0:
                    orders.append(Order("SQUID_INK", best_bid, -size))
            elif deviation < -cfg["deviation_threshold"]:
                size = min(cfg["max_size"], cfg["position_limit"] - position)
                if size > 0:
                    orders.append(Order("SQUID_INK", best_ask, size))
        return orders

    def _basket_arbitrage_strategy(self, product: str, state: TradingState):
        cfg = self.strategy_config[product]
        orders = []
        component_orders = {}
        basket_position = state.position.get(product, 0)
        component_positions = {c: state.position.get(c, 0) for c in cfg["components"]}
        basket_od = state.order_depths.get(product)
        if not basket_od:
            return orders, component_orders
        basket_bid = max(basket_od.buy_orders.keys(), default=0)
        basket_ask = min(basket_od.sell_orders.keys(), default=0)
        composite_bid, composite_ask = 0, 0
        valid = True
        for component, qty in cfg["components"].items():
            od = state.order_depths.get(component)
            if not od:
                valid = False
                break
            best_bid = max(od.buy_orders.keys(), default=0)
            best_ask = min(od.sell_orders.keys(), default=0)
            if best_bid == 0 or best_ask == 0:
                valid = False
                break
            composite_bid += best_bid * qty
            composite_ask += best_ask * qty
        if not valid:
            return orders, component_orders

        def calculate_max_size(basket_direction: str):
            size_limits = []
            if basket_direction == "BUY":
                basket_limit = cfg["position_limit"] - basket_position
                size_limits.append(basket_limit)
                for c, qty in cfg["components"].items():
                    component_limit = (self.strategy_config[c]["position_limit"] + component_positions[c]) // qty
                    size_limits.append(component_limit)
                    available = sum(state.order_depths[c].buy_orders.values())
                    size_limits.append(available // qty)
            else:  # SELL direction
                basket_limit = cfg["position_limit"] + basket_position
                size_limits.append(basket_limit)
                for c, qty in cfg["components"].items():
                    component_limit = (self.strategy_config[c]["position_limit"] - component_positions[c]) // qty
                    size_limits.append(component_limit)
                    available = sum(state.order_depths[c].sell_orders.values())
                    size_limits.append(available // qty)
            return max(0, min(size_limits))

        if composite_bid > basket_ask:
            max_size = calculate_max_size("BUY")
            if max_size > 0:
                orders.append(Order(product, basket_ask, max_size))
                for c, qty in cfg["components"].items():
                    best_bid = max(state.order_depths[c].buy_orders.keys())
                    component_orders[c] = [Order(c, best_bid, -qty * max_size)]
        if basket_bid > composite_ask:
            max_size = calculate_max_size("SELL")
            if max_size > 0:
                orders.append(Order(product, basket_bid, -max_size))
                for c, qty in cfg["components"].items():
                    best_ask = min(state.order_depths[c].sell_orders.keys())
                    component_orders[c] = [Order(c, best_ask, qty * max_size)]
        return orders, component_orders

    def run(self, state: TradingState) -> Tuple[Dict[str, List[Order]], int, str]:
        result = {product: [] for product in self.strategy_config}
        for product in state.order_depths:
            if product not in self.strategy_config:
                continue
            od = state.order_depths[product]
            position = state.position.get(product, 0)
            best_bid = max(od.buy_orders.keys(), default=0)
            best_ask = min(od.sell_orders.keys(), default=0)
            mid = (best_bid + best_ask) / 2 if best_bid and best_ask else 0
            self.history[product]["prices"].append(mid)
            if product in ["CROISSANTS", "JAMS", "DJEMBES"]:
                self.history[product]["best_bid"].append(best_bid)
                self.history[product]["best_ask"].append(best_ask)
            elif product == "RAINFOREST_RESIN":
                orders = self._resin_strategy(od, position, mid)
                result[product] = orders
            elif product == "KELP":
                orders = self._kelp_strategy(position, mid, best_bid, best_ask)
                result[product] = orders
            elif product == "SQUID_INK":
                orders = self._squid_strategy(od, position, mid, best_bid, best_ask)
                result[product] = orders

        for product in ["PICNIC_BASKET1", "PICNIC_BASKET2"]:
            position = state.position.get(product, 0)
            limit = self.strategy_config[product]["position_limit"]
            if abs(position) > limit * 0.8:
                od = state.order_depths.get(product)
                if od:
                    if position > 0:
                        best_bid = max(od.buy_orders.keys())
                        result[product].append(Order(product, best_bid, -position))
                    else:
                        best_ask = min(od.sell_orders.keys())
                        result[product].append(Order(product, best_ask, -position))

        basket_orders = {}
        for product in ["PICNIC_BASKET1", "PICNIC_BASKET2"]:
            if product in state.order_depths:
                orders, component_orders = self._basket_arbitrage_strategy(product, state)
                result[product].extend(orders)
                for comp, comp_orders in component_orders.items():
                    if comp not in basket_orders:
                        basket_orders[comp] = []
                    basket_orders[comp].extend(comp_orders)

        for product, orders in basket_orders.items():
            if product not in result:
                result[product] = orders
            else:
                result[product].extend(orders)

        for product in self.history:
            if "prices" in self.history[product]:
                self.history[product]["prices"] = self.history[product]["prices"][-100:]
        
        return result, 0, ""