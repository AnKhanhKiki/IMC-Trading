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
        # Basket configuration with components and position limits
        self.strategy_config = {
            "PICNIC_BASKET1": {
                "components": {"CROISSANTS": 6, "JAMS": 3, "DJEMBES": 1},
                "position_limit": 60,
            },
            "PICNIC_BASKET2": {
                "components": {"CROISSANTS": 4, "JAMS": 2},
                "position_limit": 100,
            },
            "CROISSANTS": {"position_limit": 250},
            "JAMS": {"position_limit": 350},
            "DJEMBES": {"position_limit": 60}
        }

        # Price history tracking for analysis
        self.history = {
            "CROISSANTS": {"prices": [], "best_bid": [], "best_ask": []},
            "JAMS": {"prices": [], "best_bid": [], "best_ask": []},
            "DJEMBES": {"prices": [], "best_bid": [], "best_ask": []},
            "PICNIC_BASKET1": {"prices": [], "arb_opportunities": []},
            "PICNIC_BASKET2": {"prices": [], "arb_opportunities": []}
        }

    def _basket_arbitrage_strategy(self, product: str, state: TradingState):
        """Core arbitrage logic between baskets and their components"""
        cfg = self.strategy_config[product]
        orders = []
        component_orders = {}

        # Get basket order book and best prices
        basket_od = state.order_depths.get(product)
        if not basket_od:
            return orders, component_orders

        # Calculate basket bid/ask prices
        basket_bid = max(basket_od.buy_orders.keys(), default=0)
        basket_ask = min(basket_od.sell_orders.keys(), default=0)
        
        # Calculate composite value from components
        composite_bid, composite_ask = 0, 0
        valid = True
        for component, qty in cfg["components"].items():
            od = state.order_depths.get(component)
            if not od:
                valid = False
                break
                
            # Sum best component prices
            best_bid = max(od.buy_orders.keys(), default=0)
            best_ask = min(od.sell_orders.keys(), default=0)
            composite_bid += best_bid * qty
            composite_ask += best_ask * qty

        if not valid:
            return orders, component_orders

        # Check arbitrage opportunities
        arb_buy_profit = composite_bid - basket_ask  # Buy basket, sell components
        arb_sell_profit = basket_bid - composite_ask  # Sell basket, buy components

        # Trade 1 basket at a time for simplicity
        max_size = 1

        # Execute buy basket/sell components arbitrage
        if arb_buy_profit > 0:
            # Check component liquidity and position limits
            liquidity_ok = True
            for c, qty in cfg["components"].items():
                available = sum(state.order_depths[c].buy_orders.values())
                if available < qty or abs(state.position.get(c,0) - qty) > self.strategy_config[c]["position_limit"]:
                    liquidity_ok = False
                    break

            if liquidity_ok:
                # Place basket buy order
                orders.append(Order(product, basket_ask, max_size))
                # Create component sell orders using best prices
                for c, qty in cfg["components"].items():
                    best_bid = max(state.order_depths[c].buy_orders.keys())
                    component_orders[c] = [Order(c, best_bid, -qty * max_size)]

        # Execute sell basket/buy components arbitrage
        if arb_sell_profit > 0:
            # Similar liquidity checks for reverse direction
            liquidity_ok = True
            for c, qty in cfg["components"].items():
                available = sum(state.order_depths[c].sell_orders.values())
                if available < qty or abs(state.position.get(c,0) + qty) > self.strategy_config[c]["position_limit"]:
                    liquidity_ok = False
                    break

            if liquidity_ok:
                # Place basket sell order
                orders.append(Order(product, basket_bid, -max_size))
                # Create component buy orders
                for c, qty in cfg["components"].items():
                    best_ask = min(state.order_depths[c].sell_orders.keys())
                    component_orders[c] = [Order(c, best_ask, qty * max_size)]

        return orders, component_orders

    def run(self, state: TradingState) -> Tuple[Dict[str, List[Order]], int, str]:
        """Main trading loop with position management and strategy execution"""
        result = {product: [] for product in self.strategy_config}

        # Update price history for analysis
        for product in ["PICNIC_BASKET1", "PICNIC_BASKET2"]:
            if product in state.order_depths:
                bids = state.order_depths[product].buy_orders
                asks = state.order_depths[product].sell_orders
                if bids and asks:
                    mid = (max(bids.keys()) + min(asks.keys())) / 2
                    self.history[product]["prices"].append(mid)

        # Force reduce positions near limit
        for product in ["PICNIC_BASKET1", "PICNIC_BASKET2"]:
            position = state.position.get(product, 0)
            limit = self.strategy_config[product]["position_limit"]
            if abs(position) > limit * 0.8:
                # Liquidate excess at best available price
                if position > 0:
                    best_bid = max(state.order_depths[product].buy_orders.keys())
                    result[product].append(Order(product, best_bid, -position))
                else:
                    best_ask = min(state.order_depths[product].sell_orders.keys())
                    result[product].append(Order(product, best_ask, -position))

        # Execute arbitrage strategies and combine orders
        for product in ["PICNIC_BASKET1", "PICNIC_BASKET2"]:
            if product in state.order_depths:
                basket_orders, comp_orders = self._basket_arbitrage_strategy(product, state)
                result[product].extend(basket_orders)
                for component, orders in comp_orders.items():
                    result[component].extend(orders)

        return result, 0, ""