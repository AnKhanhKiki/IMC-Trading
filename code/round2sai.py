import json
import math
import statistics
from dataclasses import dataclass
from typing import Dict, List, Tuple

import pandas as pd
from pandas.io.sql import com


@dataclass
class Order:
    symbol: str
    price: int
    quantity: int


class MyEncoder(json.JSONEncoder):
    """
    JSONEncoder subclass that leverages an object's `__json__()` method,
    if available, to obtain its default JSON representation.

    """

    def default(self, obj):
        if hasattr(obj, "__dict__"):
            return obj.__dict__
        return json.JSONEncoder.default(self, obj)


class OrderDepth:
    def __init__(self):
        self.buy_orders: Dict[int, int] = {}
        self.sell_orders: Dict[int, int] = {}

    def __repr__(self):
        return json.dumps(self, cls=MyEncoder)


class Listing:
    def __init__(self, symbol, product, denomination):
        self.symbol = symbol
        self.product = product
        self.denomination = denomination

    def __repr__(self):
        return json.dumps(self, cls=MyEncoder)


class Observation:
    def __init__(self, plain_value_observations, transport_fees):
        self.plain_value_observations = plain_value_observations
        self.transport_fees = transport_fees

    def __repr__(self):
        return json.dumps(self, cls=MyEncoder)


class Trade:
    def __init__(
        self,
        symbol: str,
        price: int,
        quantity: int,
        buyer: str = None,
        seller: str = None,
        timestamp: int = 0,
    ):
        self.symbol = symbol
        self.price = price
        self.quantity = quantity
        self.buyer = buyer
        self.seller = seller
        self.timestamp = timestamp

    def __repr__(self):
        return json.dumps(self, cls=MyEncoder)


class TradingState:
    def __init__(
        self,
        traderData,
        timestamp,
        listings,
        order_depths,
        own_trades,
        market_trades,
        position,
        observations,
    ):
        self.traderData = traderData
        self.timestamp = timestamp
        self.listings = listings
        self.order_depths = order_depths
        self.own_trades = own_trades
        self.market_trades = market_trades
        self.position = position
        self.observations = observations

    def __repr__(self):
        return json.dumps(self, cls=MyEncoder)


class Trader:
    def __init__(self):
        # Defines basket compositions and position limits
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
            "DJEMBES": {"position_limit": 60},
        }

        # Price history tracking
        self.history = {
            "CROISSANTS": {"prices": [], "best_bid": [], "best_ask": []},
            "JAMS": {"prices": [], "best_bid": [], "best_ask": []},
            "DJEMBES": {"prices": [], "best_bid": [], "best_ask": []},
            "PICNIC_BASKET1": {"prices": [], "arb_opportunities": []},
            "PICNIC_BASKET2": {"prices": [], "arb_opportunities": []},
        }

    def _basket_arbitrage_strategy(self, product: str, state: TradingState):
        """
        Identifies and executes arbitrage opportunities between basket products
        and their components. Implements:
        1. Composite price calculation
        2. Position limit management
        3. Liquidity-aware order sizing
        """
        cfg = self.strategy_config[product]
        orders = []
        component_orders = {}

        # Get current positions for basket and components
        basket_position = state.position.get(product, 0)
        component_positions = {c: state.position.get(c, 0) for c in cfg["components"]}

        # Get order book data for basket product
        basket_od = state.order_depths.get(product)
        if not basket_od:
            # Give up
            return orders, component_orders

        # Extract best bid/ask prices for the basket
        basket_bid = max(basket_od.buy_orders.keys(), default=0)
        basket_ask = min(basket_od.sell_orders.keys(), default=0)

        # Calculate theoretical bid/ask prices based on component markets
        composite_bid, composite_ask = 0, 0
        valid = True

        component_prices_and_qty = {c: {} for c in cfg["components"].keys()}
        for component, qty in cfg["components"].items():
            od = state.order_depths.get(component)
            if not od:
                # No order depth for that component
                valid = False
                break

            # Get best component prices
            best_bid = max(od.buy_orders.keys(), default=0)
            best_ask = min(od.sell_orders.keys(), default=0)
            component_prices_and_qty[component] = {
                "buy": (best_bid, od.buy_orders[best_bid]),
                "sell": (best_ask, abs(od.sell_orders[best_ask])),
            }

            if best_bid == 0 or best_ask == 0:
                # Invalid BBO
                valid = False
                break

            # Sum component prices weighted by basket composition
            composite_bid += best_bid * qty
            composite_ask += best_ask * qty

        if not valid:
            return orders, component_orders

        purchased_orders = {k: 0 for k in cfg["components"].keys()}
        purchased_orders[product] = 0
        # Execute when composite bid > basket ask (buy basket/sell components)
        if composite_bid > basket_ask:
            # We want to BUY basket, SELL components
            # But we want to figure out the maximum amount we can do this.
            max_quantity = (
                state.order_depths[product].buy_orders[basket_bid] - basket_position
            )

            print(product, purchased_orders)

            while purchased_orders[product] <= max_quantity and not (
                purchased_orders[product] + basket_position
                >= self.strategy_config[product]["position_limit"]
            ):
                # Check quantity of all other components in book
                # For each component, if we have not purchased above and not
                # breached position limit. We are selling here.
                components = []
                for component, qty in cfg["components"].items():
                    # If how much you purchased is less than available take it on.
                    # Also check, if we take on the total purchased and new qty on
                    # on top of current position, we exceed the limit.
                    # If either are true, give up.
                    #
                    # THIS IS FOR SELL SO POSITIONS ARE NEGATIVE.
                    if (
                        purchased_orders[component]
                        >= abs(component_prices_and_qty[component]["sell"][1])
                        or (
                            component_positions[component]
                            - purchased_orders[component]
                            - qty
                        )
                        > self.strategy_config[component]["position_limit"]
                    ):
                        # Give up if we can no longer purchase this anymore.
                        break
                    components.append(component)

                if len(components) == 3:
                    # We have all components available to update
                    purchased_orders[product] += 1
                    for component, qty in cfg["components"].items():
                        purchased_orders[component] += qty
                else:
                    # Else give up
                    break

            if purchased_orders[product] > 0:
                # print("SELL BASKET, BUY COMPONENTS")
                # print(purchased_orders)
                # print(component_prices_and_qty)
                # We have to add orders now.
                for name, volume in purchased_orders.items():
                    if name == product:
                        orders.append(Order(product, basket_ask, volume))
                    else:
                        component_orders[name] = [
                            Order(
                                name, component_prices_and_qty[name]["sell"][0], -volume
                            )
                        ]

        # Execute when basket bid > composite ask (sell basket/buy components)
        if basket_bid > composite_ask:
            # We want to SELL basket, BUY components
            max_quantity = (
                state.order_depths[product].sell_orders[basket_ask] + basket_position
            )

            while purchased_orders[product] <= abs(max_quantity) and not (
                abs(-purchased_orders[product] + basket_position)
                >= self.strategy_config[product]["position_limit"]
            ):
                # Check quantity of all other components in book
                # For each component, if we have not purchased above and not
                # breached position limit. We are selling here.
                components = []
                for component, qty in cfg["components"].items():
                    # If how much you purchased is less than available take it on.
                    # Also check, if we take on the total purchased and new qty on
                    # on top of current position, we exceed the limit.
                    # If either are true, give up.
                    #
                    # THIS IS FOR BUY SO POSITIONS ARE POSITIVE.
                    if (
                        purchased_orders[component]
                        >= abs(component_prices_and_qty[component]["buy"][1])
                        or (
                            component_positions[component]
                            + purchased_orders[component]
                            + qty
                        )
                        > self.strategy_config[component]["position_limit"]
                    ):
                        # Give up if we can no longer purchase this anymore.
                        break
                    components.append(component)

                if len(components) == len(cfg["components"]):
                    # We have all components available to update
                    purchased_orders[product] += 1
                    for component, qty in cfg["components"].items():
                        purchased_orders[component] += qty
                else:
                    # Else give up
                    break

            if purchased_orders[product] > 0:
                # print("BUY BASKET, SELL COMPONENTS")
                # print(purchased_orders)
                # print(component_prices_and_qty)
                # We have to add orders now.
                for name, volume in purchased_orders.items():
                    if name == product:
                        orders.append(Order(product, basket_bid, -volume))
                    else:
                        component_orders[name] = [
                            Order(
                                name, component_prices_and_qty[name]["buy"][0], volume
                            )
                        ]

        return orders, component_orders

    def run(self, state: TradingState) -> Tuple[Dict[str, List[Order]], int, str]:
        result = {product: [] for product in self.strategy_config}

        # Force close positions exceeding 80% of limit to maintain safety margin
        """WE CAN MESS ABOUT THIS LIMIT"""
        for product in [
            "PICNIC_BASKET1",
            "PICNIC_BASKET2",
            "DJEMBES",
            "CROISSANTS",
            "JAMS",
        ]:
            position = state.position.get(product, 0)
            limit = self.strategy_config[product]["position_limit"]
            print(product, abs(position), limit)

            # if abs(position) > limit * 0.8:
            if abs(position) > 0:
                od = state.order_depths.get(product)
                if od:
                    # Liquidate entire position at best available price
                    if position > 0:
                        best_bid = max(od.buy_orders.keys())
                        result[product].append(Order(product, best_bid, -position))
                    else:
                        best_ask = min(od.sell_orders.keys())
                        result[product].append(Order(product, best_ask, -position))

        # Execute basket arbitrage for both products
        basket_orders = {}
        for product in ["PICNIC_BASKET1", "PICNIC_BASKET2"]:
            if product in state.order_depths:
                # Get arbitrage orders for basket and components
                orders, component_orders = self._basket_arbitrage_strategy(
                    product, state
                )
                result[product] = orders
                basket_orders.update(component_orders)

        # Combine component orders from different baskets
        # Handles cases where components are shared between baskets
        for product, orders in basket_orders.items():
            if product not in result:
                result[product] = orders
            else:
                result[product].extend(orders)

        return result, 0, ""
