import pandas as pd
from typing import Dict, List
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

import math
import statistics

class Trader:
    def __init__(self, lookbacks=None, band_factor=0.5, max_trade_size=20):
        if lookbacks is None:
            lookbacks = [10, 20, 40, 80]
        self.lookbacks = lookbacks
        self.band_factor = band_factor
        self.max_trade_size = max_trade_size
        # First strategy state (moving average)
        self.mid_price_history = {"KELP": [], "RAINFOREST_RESIN": []}
        # Second strategy state (multi-lookback)
        self.price_history = {lb: {"SQUID_INK": []} for lb in lookbacks}
        self.performance = {lb: {"SQUID_INK": 0.0} for lb in lookbacks}
        self.last_mid_price = {"SQUID_INK": 0.0}

    def run(self, state: TradingState) -> Tuple[Dict[str, List[Order]], int, str]:
        result = {}
        for product in state.order_depths:
            order_depth = state.order_depths[product]
            position = state.position.get(product, 0)
            position_limit = 50
            orders = []

            if not order_depth.buy_orders or not order_depth.sell_orders:
                result[product] = orders
                continue

            best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else 0
            best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else float('inf')
            mid_price = (best_bid + best_ask) / 2 if best_bid and best_ask != float('inf') else 0

            if product in ["KELP", "RAINFOREST_RESIN"]:
                # First strategy: Moving average-based fair value
                self.mid_price_history[product].append(mid_price)
                if len(self.mid_price_history[product]) > 10:
                    self.mid_price_history[product] = self.mid_price_history[product][-10:]

                fair_value = (sum(self.mid_price_history[product]) / len(self.mid_price_history[product]) 
                            if self.mid_price_history[product] else mid_price)

                if best_ask < fair_value and position < position_limit:
                    qty = min(10, position_limit - position)
                    orders.append(Order(product, int(best_ask), qty))
                if best_bid > fair_value and position > -position_limit:
                    qty = min(10, position + position_limit)
                    orders.append(Order(product, int(best_bid), -qty))

            elif product == "SQUID_INK":
                # Second strategy: Multi-lookback with volatility bands
                for lb in self.lookbacks:
                    self.price_history[lb][product].append(mid_price)
                    if len(self.price_history[lb][product]) > lb:
                        self.price_history[lb][product] = self.price_history[lb][product][-lb:]

                # Performance update
                if self.last_mid_price[product] > 0:
                    price_change = mid_price - self.last_mid_price[product]
                    for lb in self.lookbacks:
                        signal = self._signal_for_lookback(lb, product)
                        self.performance[lb][product] += signal * price_change
                self.last_mid_price[product] = mid_price

                # Weighted signals
                total_perf = sum(abs(self.performance[lb][product]) for lb in self.lookbacks) or 1.0
                signals = {
                    lb: (self._signal_for_lookback(lb, product) * (abs(self.performance[lb][product]) / total_perf))
                    for lb in self.lookbacks
                }
                meta_signal = sum(signals.values())

                # Volatility-based bands
                longest_lb = max(self.lookbacks)
                if len(self.price_history[longest_lb][product]) > 1:
                    price_stdev = statistics.pstdev(self.price_history[longest_lb][product])
                else:
                    price_stdev = 0

                fair_value = (sum(self.price_history[longest_lb][product]) / 
                            len(self.price_history[longest_lb][product]))
                lower_band = fair_value - self.band_factor * price_stdev
                upper_band = fair_value + self.band_factor * price_stdev

                if meta_signal > 0 and best_ask < lower_band and position < position_limit:
                    deviation = lower_band - best_ask
                    raw_qty = max(1, math.ceil(deviation))
                    qty = min(raw_qty, position_limit - position, self.max_trade_size)
                    if qty > 0:
                        orders.append(Order(product, int(best_ask), qty))

                if meta_signal < 0 and best_bid > upper_band and position > -position_limit:
                    deviation = best_bid - upper_band
                    raw_qty = max(1, math.ceil(deviation))
                    qty = min(raw_qty, position + position_limit, self.max_trade_size)
                    if qty > 0:
                        orders.append(Order(product, int(best_bid), -qty))

            result[product] = orders

        return result, 0, ""

    def _signal_for_lookback(self, lb: int, product: str) -> float:
        arr = self.price_history[lb][product]
        if len(arr) < 2:
            return 0.0
        slope = (arr[-1] - arr[0]) / lb
        return 1.0 if slope > 0 else -1.0


trader = Trader()
