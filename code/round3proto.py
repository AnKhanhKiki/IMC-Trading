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
        self.price: int = price
        self.quantity: int = quantity
        self.buyer: str = buyer
        self.seller: str = seller
        self.timestamp: int = timestamp

class TradingState:
    def __init__(self, traderData, timestamp, listings, order_depths, own_trades, market_trades, position, observations):
        self.traderData = traderData
        self.timestamp: int = timestamp
        self.listings = listings
        self.order_depths = order_depths
        self.own_trades = own_trades
        self.market_trades = market_trades
        self.position = position
        self.observations = observations

class Trader:
    def __init__(self):
        # Existing strategy config
        self.strategy_config = {
            "RAINFOREST_RESIN": {
                "vwap_lookback": 10,
                "spread_ratio": 0.409,
                "max_size": 12,
                "inventory_penalty": 0.0005,
                "position_limit": 50  
            },
            "KELP": {
                "sma_period": 18,
                "deviation_threshold": 0.000005,
                "price_improvement": 0.005,
                "position_limit": 50  
            },
            "SQUID_INK": {
                "ma_period": 8,
                "deviation_threshold": 0.02,
                "max_size": 50,
                "position_limit": 50  
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
            "DJEMBES": {"position_limit": 60},
            # Add new products
            "VOLCANIC_ROCK": {"position_limit": 400}
        }
        # Add Volcanic Rock Vouchers to strategy config
        voucher_strikes = [9500, 9750, 10000, 10250, 10500]
        for strike in voucher_strikes:
            symbol = f"VOLCANIC_ROCK_VOUCHER_{strike}"
            self.strategy_config[symbol] = {"position_limit": 200}
        
        # Existing history setup
        self.history = {
            "RAINFOREST_RESIN": {"prices": [], "spreads": []},
            "KELP": {"prices": [], "ema": None},
            "SQUID_INK": {"prices": []},
            "CROISSANTS": {"prices": [], "best_bid": [], "best_ask": []},
            "JAMS": {"prices": [], "best_bid": [], "best_ask": []},
            "DJEMBES": {"prices": [], "best_bid": [], "best_ask": []},
            "PICNIC_BASKET1": {"prices": [], "arb_opportunities": []},
            "PICNIC_BASKET2": {"prices": [], "arb_opportunities": []},
            # Add new products to history
            "VOLCANIC_ROCK": {"prices": []}
        }
        for strike in voucher_strikes:
            symbol = f"VOLCANIC_ROCK_VOUCHER_{strike}"
            self.history[symbol] = {"prices": []}
        
        self.current_round = 0  # To track days left

        def norm_cdf(self, x):
        # Approximation of the cumulative distribution function for the standard normal distribution
            a1 = 0.254829592
            a2 = -0.284496736
            a3 = 1.421413741
            a4 = -1.453152027
            a5 = 1.061405429
            p = 0.3275911

            sign = 1
            if x < 0:
                sign = -1
            x = abs(x) / math.sqrt(2)

            t = 1.0 / (1.0 + p * x)
            y = 1.0 - (((((a5 * t + a4) * t + a3) * t + a2) * t + a1) * t * math.exp(-x * x))

            return 0.5 * (1.0 + sign * y)
        
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
    
    def _calculate_volcanic_volatility(self, vr_symbol: str) -> float:
        """Enhanced volatility calculation with EMA smoothing"""
        vr_prices = self.history[vr_symbol]["prices"]
        if len(vr_prices) >= 5:  # Require more data points
            returns = []
            for i in range(1, len(vr_prices)):
                if vr_prices[i-1] == 0 or vr_prices[i] == 0:
                    continue
                try:
                    log_return = math.log(vr_prices[i]/vr_prices[i-1])
                    returns.append(abs(log_return))  # Use absolute returns
                except:
                    continue
            
            if len(returns) >= 5:
                # Apply EMA to recent volatilities
                recent_vol = statistics.stdev(returns[-5:])
                if 'prev_vol' not in self.history[vr_symbol]:
                    self.history[vr_symbol]['prev_vol'] = recent_vol
                else:
                    alpha = 0.3  # EMA smoothing factor
                    self.history[vr_symbol]['prev_vol'] = alpha*recent_vol + (1-alpha)*self.history[vr_symbol]['prev_vol']
                return min(0.5, max(0.15, self.history[vr_symbol]['prev_vol']))  # Bound between 15%-50%
        return 0.25  # Default mid-range volatility

    def _volcanic_voucher_strategy(self, state: TradingState, vr_mid: float, sigma_daily: float) -> Dict[str, List[Order]]:
        """Enhanced strategy with dynamic buffers and trend adaptation"""
        voucher_symbols = [f"VOLCANIC_ROCK_VOUCHER_{strike}" 
                          for strike in [9500, 9750, 10000, 10250, 10500]]
        days_left = 7 - (self.current_round - 1)
        orders = {}
        
        # Trend detection (simple MA crossover)
        vr_prices = self.history["VOLCANIC_ROCK"]["prices"]
        short_term = 3
        long_term = 10
        trend = "neutral"
        
        if len(vr_prices) >= long_term:
            short_ma = sum(vr_prices[-short_term:])/short_term
            long_ma = sum(vr_prices[-long_term:])/long_term
            trend = "up" if short_ma > long_ma*1.002 else "down" if short_ma < long_ma*0.998 else "neutral"

        for symbol in voucher_symbols:
            if symbol not in state.order_depths:
                continue
                
            od = state.order_depths[symbol]
            position = state.position.get(symbol, 0)
            best_bid = max(od.buy_orders.keys(), default=0)
            best_ask = min(od.sell_orders.keys(), default=0)
            
            if not best_bid or not best_ask or not vr_mid:
                continue

            strike = int(symbol.split('_')[-1])
            moneyness = vr_mid / strike
            
            # Dynamic buffer based on moneyness and volatility
            base_buffer = 0.02
            if moneyness > 1.02:  # Deep ITM
                buffer = base_buffer * 0.7
            elif moneyness < 0.98:  # Deep OTM
                buffer = base_buffer * 1.5
            else:
                buffer = base_buffer
                
            # Adjust buffer for trend
            if trend == "up":
                buffer *= 0.8 if "10500" in symbol else 1.2
            elif trend == "down":
                buffer *= 0.8 if "9500" in symbol else 1.2

            try:
                T = max(0.1, days_left/365)  # Minimum 0.1 years to avoid div/0
                sigma = sigma_daily * math.sqrt(365)
                d1 = (math.log(vr_mid/strike) + (sigma**2/2)*T) / (sigma*math.sqrt(T))
                d2 = d1 - sigma*math.sqrt(T)
                C = vr_mid * self.norm_cdf(d1) - strike * self.norm_cdf(d2)
            except:
                C = max(vr_mid - strike, 0)

            market_price = (best_bid + best_ask) / 2
            symbol_orders = []
            
            # Aggressive trading in high volatility
            if sigma_daily > 0.3:
                buffer *= 0.7
                
            if C - market_price > buffer:
                max_buy = min(
                    self.strategy_config[symbol]["position_limit"] - position,
                    50 if days_left <= 2 else 25  # Larger sizes near expiry
                )
                ask_vol = -od.sell_orders.get(best_ask, 0)
                if max_buy > 0 and ask_vol > 0:
                    price = best_ask - 1 if trend == "up" else best_ask  # Price improvement in uptrend
                    symbol_orders.append(Order(symbol, price, min(max_buy, ask_vol)))
                    
            elif market_price - C > buffer:
                max_sell = min(
                    self.strategy_config[symbol]["position_limit"] + position,
                    50 if days_left <= 2 else 25
                )
                bid_vol = od.buy_orders.get(best_bid, 0)
                if max_sell > 0 and bid_vol > 0:
                    price = best_bid + 1 if trend == "down" else best_bid
                    symbol_orders.append(Order(symbol, price, -min(max_sell, bid_vol)))
            
            orders[symbol] = symbol_orders
        
        return orders

    def run(self, state: TradingState) -> Tuple[Dict[str, List[Order]], int, str]:
        self.current_round += 1  # Increment round counter
        result = {product: [] for product in self.strategy_config}
        
        # Existing strategy processing
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
        
        basket_orders = {}
        for product in ["PICNIC_BASKET1", "PICNIC_BASKET2"]:
            if product in state.order_depths:
                # Position closing logic
                position = state.position.get(product, 0)
                limit = self.strategy_config[product]["position_limit"]
                if abs(position) > limit * 0.8:
                    od = state.order_depths[product]
                    if position > 0 and od.buy_orders:
                        best_bid = max(od.buy_orders.keys())
                        result[product].append(Order(product, best_bid, -position))
                    elif position < 0 and od.sell_orders:
                        best_ask = min(od.sell_orders.keys())
                        result[product].append(Order(product, best_ask, -position))
                
                # Arbitrage logic
                orders, component_orders = self._basket_arbitrage_strategy(product, state)
                result[product].extend(orders)
                for comp, comp_orders in component_orders.items():
                    basket_orders.setdefault(comp, []).extend(comp_orders)
        for product, orders in basket_orders.items():
            result[product].extend(orders)

        vr_symbol = "VOLCANIC_ROCK"
        vr_mid = None
        if vr_symbol in state.order_depths:
            best_bid = max(state.order_depths[vr_symbol].buy_orders.keys(), default=0)
            best_ask = min(state.order_depths[vr_symbol].sell_orders.keys(), default=0)
            if best_bid and best_ask:
                vr_mid = (best_bid + best_ask) / 2
                self.history[vr_symbol]["prices"].append(vr_mid)
        
        if vr_mid:
            sigma_daily = self._calculate_volcanic_volatility(vr_symbol)
            voucher_orders = self._volcanic_voucher_strategy(state, vr_mid, sigma_daily)
            for symbol, symbol_orders in voucher_orders.items():
                result[symbol].extend(symbol_orders)
        
        # History maintenance
        for product in self.history:
            if "prices" in self.history[product]:
                self.history[product]["prices"] = self.history[product]["prices"][-1000:]
        
        return result, 0, ""