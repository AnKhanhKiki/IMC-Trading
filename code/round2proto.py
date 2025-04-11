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
        # Defines basket compositions and position limits
        self.strategy_config = {
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
        
        # Price history tracking
        self.history = {
            "CROISSANTS": {"prices": [], "best_bid": [], "best_ask": []},
            "JAMS": {"prices": [], "best_bid": [], "best_ask": []},
            "DJEMBES": {"prices": [], "best_bid": [], "best_ask": []},
            "PICNIC_BASKET1": {"prices": [], "arb_opportunities": []},
            "PICNIC_BASKET2": {"prices": [], "arb_opportunities": []}
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
            return orders, component_orders
        
        # Extract best bid/ask prices for the basket
        basket_bid = max(basket_od.buy_orders.keys(), default=0)
        basket_ask = min(basket_od.sell_orders.keys(), default=0)
        
        # Calculate theoretical bid/ask prices based on component markets
        composite_bid, composite_ask = 0, 0
        valid = True
        
        for component, qty in cfg["components"].items():
            od = state.order_depths.get(component)
            if not od:
                valid = False
                break
                
            # Get best component prices
            best_bid = max(od.buy_orders.keys(), default=0)
            best_ask = min(od.sell_orders.keys(), default=0)
            
            if best_bid == 0 or best_ask == 0:
                valid = False
                break
                
            # Sum component prices weighted by basket composition
            composite_bid += best_bid * qty
            composite_ask += best_ask * qty

        if not valid:
            return orders, component_orders

        # Determine maximum executable size considering:
        # - Position limits
        # - Available liquidity
        # - Current positions
        def calculate_max_size(basket_direction: str):
            size_limits = []
            
            if basket_direction == "BUY":
                # Buy basket capacity: remaining long positions
                basket_limit = cfg["position_limit"] - basket_position
                size_limits.append(basket_limit)
                
                # Component sell capacity: remaining short positions
                for c, qty in cfg["components"].items():
                    component_limit = (self.strategy_config[c]["position_limit"] + component_positions[c]) // qty
                    size_limits.append(component_limit)
                    
                # Check available liquidity in component markets
                for c, qty in cfg["components"].items():
                    available = sum(state.order_depths[c].buy_orders.values())
                    size_limits.append(available // qty)
                    
            else:  # SELL direction
                # Sell basket capacity: remaining short positions
                basket_limit = cfg["position_limit"] + basket_position
                size_limits.append(basket_limit)
                
                # Component buy capacity: remaining long positions
                for c, qty in cfg["components"].items():
                    component_limit = (self.strategy_config[c]["position_limit"] - component_positions[c]) // qty
                    size_limits.append(component_limit)
                    
                # Check available liquidity in component markets
                for c, qty in cfg["components"].items():
                    available = sum(state.order_depths[c].sell_orders.values())
                    size_limits.append(available // qty)
                    
            return max(0, min(size_limits))

        # Execute when composite bid > basket ask (buy basket/sell components)
        if composite_bid > basket_ask:
            max_size = calculate_max_size("BUY")
            if max_size > 0:
                # Create basket buy order
                orders.append(Order(product, basket_ask, max_size))
                # Create component sell orders
                for c, qty in cfg["components"].items():
                    best_bid = max(state.order_depths[c].buy_orders.keys())
                    component_orders[c] = [Order(c, best_bid, -qty * max_size)]

        # Execute when basket bid > composite ask (sell basket/buy components)
        if basket_bid > composite_ask:
            max_size = calculate_max_size("SELL")
            if max_size > 0:
                # Create basket sell order
                orders.append(Order(product, basket_bid, -max_size))
                # Create component buy orders
                for c, qty in cfg["components"].items():
                    best_ask = min(state.order_depths[c].sell_orders.keys())
                    component_orders[c] = [Order(c, best_ask, qty * max_size)]

        return orders, component_orders


    def run(self, state: TradingState) -> Tuple[Dict[str, List[Order]], int, str]:
        result = {product: [] for product in self.strategy_config}
        

        # Force close positions exceeding 80% of limit to maintain safety margin
        '''WE CAN MESS ABOUT THIS LIMIT'''
        for product in ["PICNIC_BASKET1", "PICNIC_BASKET2"]:
            position = state.position.get(product, 0)
            limit = self.strategy_config[product]["position_limit"]
            
            if abs(position) > limit * 0.8:
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
                orders, component_orders = self._basket_arbitrage_strategy(product, state)
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