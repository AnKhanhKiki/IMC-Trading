import pandas as pd
from dataclasses import dataclass
import math
import statistics
from typing import Dict, List, Tuple

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
            "PICNIC_BASKET1": {
                "components": {"CROISSANTS": 6, "JAMS": 3, "DJEMBES": 1},
                "position_limit": 60,
                "profit_threshold": 50
            },
            "PICNIC_BASKET2": {
                "components": {"CROISSANTS": 4, "JAMS": 2},
                "position_limit": 100,
                "profit_threshold": 30
            },
            "CROISSANTS": {"position_limit": 250},
            "JAMS": {"position_limit": 350},
            "DJEMBES": {"position_limit": 60}
        }
        
        self.history = {
            "CROISSANTS": {"prices": [], "best_bid": [], "best_ask": []},
            "JAMS": {"prices": [], "best_bid": [], "best_ask": []},
            "DJEMBES": {"prices": [], "best_bid": [], "best_ask": []},
            "PICNIC_BASKET1": {"prices": [], "arb_opportunities": []},
            "PICNIC_BASKET2": {"prices": [], "arb_opportunities": []}
        }

    def _basket_arbitrage_strategy(self, product: str, state: TradingState):
        cfg = self.strategy_config[product]
        orders = []
        component_orders = {}

        basket_od = state.order_depths.get(product)
        if not basket_od:
            return orders, component_orders

        # Basket market prices
        basket_bid = max(basket_od.buy_orders.keys(), default=0)
        basket_ask = min(basket_od.sell_orders.keys(), default=0)
        
        if basket_bid == 0 or basket_ask == 0:
            return orders, component_orders

        composite_bid = 0
        composite_ask = 0
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

        arb_buy = composite_bid - basket_ask  # Buy basket, sell components
        arb_sell = basket_bid - composite_ask  # Sell basket, buy components

        price_std = statistics.stdev(self.history[product]["prices"][-20:]) if len(self.history[product]["prices"]) >= 20 else 0
        dynamic_threshold = max(cfg["profit_threshold"], price_std * 0.5)

        max_size = 1  # 1 Basket
        # Buy basket, sell components (instant round-trip)
        if arb_buy > dynamic_threshold:
            # Check we can instantly hit the bid on all components
            enough_liquidity = True
            for c, qty in cfg["components"].items():
                od = state.order_depths[c]
                if len(od.buy_orders) == 0:
                    enough_liquidity = False
                    break

            if enough_liquidity:
                orders.append(Order(product, basket_ask, max_size))
                for c, qty in cfg["components"].items():
                    best_bid = max(state.order_depths[c].buy_orders.keys())
                    component_orders[c] = [Order(c, best_bid, -qty * max_size)]

        # Sell basket, buy components (instant round-trip)
        if arb_sell > dynamic_threshold:
            # Check we can instantly lift the ask on all components
            enough_liquidity = True
            for c, qty in cfg["components"].items():
                od = state.order_depths[c]
                if len(od.sell_orders) == 0:
                    enough_liquidity = False
                    break

            if enough_liquidity:
                orders.append(Order(product, basket_bid, -max_size))
                for c, qty in cfg["components"].items():
                    best_ask = min(state.order_depths[c].sell_orders.keys())
                    component_orders[c] = [Order(c, best_ask, qty * max_size)]

        return orders, component_orders



    def run(self, state: TradingState) -> Tuple[Dict[str, List[Order]], int, str]:
        result = {product: [] for product in self.strategy_config}
        
        # Unwind positions if they exceed safety thresholds
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

        # Execute arbitrage strategy for each basket
        basket_orders = {}
        for product in ["PICNIC_BASKET1", "PICNIC_BASKET2"]:
            if product in state.order_depths:
                orders, component_orders = self._basket_arbitrage_strategy(product, state)
                result[product] = orders
                basket_orders.update(component_orders)

        # Combine component orders into result
        for product, orders in basket_orders.items():
            result[product] = orders if product not in result else result[product] + orders

        return result, 0, ""