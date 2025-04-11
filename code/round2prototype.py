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

        max_size = 1  # 1 Basket only

        # Buy basket, sell components 
        if arb_buy > dynamic_threshold:
            enough_liquidity = True
            for c, qty in cfg["components"].items():
                od = state.order_depths[c]
                total_bid_volume = sum(od.buy_orders.values())

                # Check if we can sell enough units
                if total_bid_volume < qty * max_size:
                    enough_liquidity = False
                    break

                current_pos = state.position.get(c, 0)
                limit = self.strategy_config.get(c, {}).get("position_limit", 250)
                projected_pos = current_pos - qty * max_size  
                if abs(projected_pos) > limit:
                    enough_liquidity = False
                    break

            if enough_liquidity:
                orders.append(Order(product, basket_ask, max_size))
                for c, qty in cfg["components"].items():
                    od = state.order_depths[c]
                    remaining_qty = qty * max_size
                    component_orders[c] = []

                    for price in sorted(od.buy_orders.keys(), reverse=True):  # best bid to worst
                        volume = od.buy_orders[price]
                        trade_qty = min(volume, remaining_qty)
                        component_orders[c].append(Order(c, price, -trade_qty))
                        remaining_qty -= trade_qty
                        if remaining_qty == 0:
                            break

        # Sell basket, buy components 
        if arb_sell > dynamic_threshold:
            enough_liquidity = True
            for c, qty in cfg["components"].items():
                od = state.order_depths[c]
                total_ask_volume = sum(od.sell_orders.values())

                # Check if we can buy enough units
                if total_ask_volume < qty * max_size:
                    enough_liquidity = False
                    break

                current_pos = state.position.get(c, 0)
                limit = self.strategy_config.get(c, {}).get("position_limit", 250)
                projected_pos = current_pos + qty * max_size  # buying component
                if abs(projected_pos) > limit:
                    enough_liquidity = False
                    break

            if enough_liquidity:
                orders.append(Order(product, basket_bid, -max_size))
                for c, qty in cfg["components"].items():
                    od = state.order_depths[c]
                    remaining_qty = qty * max_size
                    component_orders[c] = []

                    for price in sorted(od.sell_orders.keys()):  # best ask to worst
                        volume = od.sell_orders[price]
                        trade_qty = min(volume, remaining_qty)
                        component_orders[c].append(Order(c, price, trade_qty))
                        remaining_qty -= trade_qty
                        if remaining_qty == 0:
                            break

        return orders, component_orders


    def run(self, state: TradingState) -> Tuple[Dict[str, List[Order]], int, str]:
        result = {product: [] for product in self.strategy_config}

        # Update price history
        for product in ["PICNIC_BASKET1", "PICNIC_BASKET2"]:
            od = state.order_depths.get(product)
            if od and od.buy_orders and od.sell_orders:
                bid = max(od.buy_orders.keys())
                ask = min(od.sell_orders.keys())
                mid = (bid + ask) / 2
                self.history[product]["prices"].append(mid)

        # Unwind if over 80% position limit
        for product in ["PICNIC_BASKET1", "PICNIC_BASKET2"]:
            position = state.position.get(product, 0)
            limit = self.strategy_config[product]["position_limit"]
            od = state.order_depths.get(product)

            if abs(position) > limit * 0.8 and od:
                if position > 0 and od.buy_orders:
                    best_bid = max(od.buy_orders.keys())
                    volume = od.buy_orders[best_bid]
                    result[product].append(Order(product, best_bid, -min(position, volume)))
                elif position < 0 and od.sell_orders:
                    best_ask = min(od.sell_orders.keys())
                    volume = od.sell_orders[best_ask]
                    result[product].append(Order(product, best_ask, -max(position, -volume)))

        # Run arbitrage strategy
        basket_orders = {}
        for product in ["PICNIC_BASKET1", "PICNIC_BASKET2"]:
            if product in state.order_depths:
                orders, comp_orders = self._basket_arbitrage_strategy(product, state)
                result[product].extend(orders)
                for c, olist in comp_orders.items():
                    if c not in result:
                        result[c] = []
                    result[c].extend(olist)

        return result, 0, ""