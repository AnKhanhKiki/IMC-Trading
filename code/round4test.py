import pandas as pd
import numpy as np
import math
import statistics
from typing import Dict, List, Tuple
import json
from json import JSONEncoder
import jsonpickle
import string


Time = int
Symbol = str
Product = str
Position = int
UserId = str
ObservationValue = int

class Order:
    def __init__(self, symbol: Symbol, price: int, quantity: int) -> None:
        self.symbol = symbol
        self.price = price
        self.quantity = quantity

    def __str__(self) -> str:
        return "(" + self.symbol + ", " + str(self.price) + ", " + str(self.quantity) + ")"

    def __repr__(self) -> str:
        return "(" + self.symbol + ", " + str(self.price) + ", " + str(self.quantity) + ")"


class OrderDepth:
    def __init__(self):
        self.buy_orders: Dict[int, int] = {}
        self.sell_orders: Dict[int, int] = {}

class Listing:
    def __init__(self, symbol, product, denomination):
        self.symbol = symbol
        self.product = product
        self.denomination = denomination

class ConversionObservation:
    def __init__(self, bidPrice: float, askPrice: float, transportFees: float, exportTariff: float, importTariff: float, sugarPrice: float, sunlightIndex: float):
        self.bidPrice = bidPrice
        self.askPrice = askPrice
        self.transportFees = transportFees
        self.exportTariff = exportTariff
        self.importTariff = importTariff
        self.sugarPrice = sugarPrice
        self.sunlightIndex = sunlightIndex

class Observation:
    def __init__(self, plainValueObservations: Dict[Product, ObservationValue], conversionObservations: Dict[Product, ConversionObservation]) -> None:
        self.plainValueObservations = plainValueObservations
        self.conversionObservations = conversionObservations

    def __str__(self) -> str:
        return "(plainValueObservations: " + jsonpickle.encode(self.plainValueObservations) + ", conversionObservations: " + jsonpickle.encode(self.conversionObservations) + ")"

class Trade:
    def __init__(self, symbol: str, price: int, quantity: int,
                 buyer: str = None, seller: str = None, timestamp: int = 0):
        self.symbol = symbol
        self.price: int = price
        self.quantity: int = quantity
        self.buyer: str = buyer
        self.seller: str = seller
        self.timestamp: int = timestamp

class TradingState(object):
    def __init__(self,
                 traderData: str,
                 timestamp: Time,
                 listings: Dict[Symbol, Listing],
                 order_depths: Dict[Symbol, OrderDepth],
                 own_trades: Dict[Symbol, List[Trade]],
                 market_trades: Dict[Symbol, List[Trade]],
                 position: Dict[Product, Position],
                 observations: Observation):
        self.traderData = traderData
        self.timestamp = timestamp
        self.listings = listings
        self.order_depths = order_depths
        self.own_trades = own_trades
        self.market_trades = market_trades
        self.position = position
        self.observations = observations

    def toJSON(self):
        return json.dumps(self, default=lambda o: o.__dict__, sort_keys=True)

class ProsperityEncoder(JSONEncoder):
        def default(self, o):
            return o.__dict__

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
            "VOLCANIC_ROCK": {"position_limit": 400},
            "MAGNIFICENT_MACARONS": {"position_limit": 75, "CSI": 45} # Add MACARONS config
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
            "VOLCANIC_ROCK": {"prices": []},
            "MAGNIFICENT_MACARONS": {"prices": [], "sunlight_sugar_regression": {
                "sunlight_coef": -2.5135,
                "sugar_coef": 3.5815,
                "intercept": 76.7563,
                "r_squared": 0.2108
            }, "observations": []} # Add MACARONS history
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

    def _calculate_vwap(self, order_depth: OrderDepth):
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

    def _resin_strategy(self, order_depth: OrderDepth, position: int, mid: int):
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
    
    def _update_sunlight_sugar_regression(self, product):
        """
        Update regression coefficients focusing specifically on sunlight and sugar price.
        This analysis aims to find the relationship between:
        midPrice = a*sunlightIndex + b*sugarPrice + c
        
        Where a, b, and c are the coefficients we're solving for.
        """
        try:
            # Get observations
            observations = np.array(self.history[product]["observations"])
            
            if len(observations) < 30:
                print("Not enough data points for regression analysis yet")
                return
                
            # Extract only the relevant features
            sunlight_index = observations[:, 1]  # sunlightIndex is at index 1
            sugar_price = observations[:, 0]     # sugarPrice is at index 0
            mid_price = observations[:, -1]      # midPrice is the last element
            
            # Create feature matrix with sunlight and sugar
            X = np.column_stack([sunlight_index, sugar_price])
            
            # Add column of ones for intercept
            X_with_intercept = np.column_stack([np.ones(X.shape[0]), X])
            
            # Calculate coefficients using normal equation: β = (X^T X)^-1 X^T y
            XT_X = np.dot(X_with_intercept.T, X_with_intercept)
            XT_y = np.dot(X_with_intercept.T, mid_price)
            
            try:
                # Try to use direct matrix inverse
                inv_XT_X = np.linalg.inv(XT_X)
                coefficients = np.dot(inv_XT_X, XT_y)
            except np.linalg.LinAlgError:
                # If matrix is singular, use pseudo-inverse for more stability
                inv_XT_X = np.linalg.pinv(XT_X)
                coefficients = np.dot(inv_XT_X, XT_y)
            
            # Store coefficients as [sunlight_coef, sugar_coef, intercept]
            sunlight_coef = coefficients[1]
            sugar_coef = coefficients[2]
            intercept = coefficients[0]
            
            # Calculate R² to evaluate model fit
            y_pred = np.dot(X_with_intercept, coefficients)
            ss_total = np.sum((mid_price - np.mean(mid_price))**2)
            ss_residual = np.sum((mid_price - y_pred)**2)
            r_squared = 1 - (ss_residual / ss_total)
            
            # Store regression results
            self.history[product]["sunlight_sugar_regression"] = {
                "sunlight_coef": sunlight_coef,
                "sugar_coef": sugar_coef,
                "intercept": intercept,
                "r_squared": r_squared
            }
            
            # Print results
            print(f"Updated Sunlight-Sugar Regression:")
            print(f"midPrice = {sugar_coef:.2f}*sugarPrice + {sunlight_coef:.2f}*sunlightIndex + {intercept:.2f}")
            print(f"R² = {r_squared:.4f}")
            
        except Exception as e:
            print(f"Error updating sunlight-sugar regression: {e}")

    def _estimate_fair_price(self, product, sunlight_index, sugar_price):
        """
        Estimate the fair price of macarons using the sunlight-sugar regression model
        
        Returns:
            float: Estimated fair price
            float: CSI value from config
        """
        # Check if we have regression results
        if "sunlight_sugar_regression" not in self.history[product]:
            return None, self.strategy_config["MAGNIFICENT_MACARONS"]["CSI"]
        
        # Get regression coefficients
        reg = self.history[product]["sunlight_sugar_regression"]
        
        # Calculate basic fair price using regression
        fair_price = (
            reg["sugar_coef"] * sugar_price +
            reg["sunlight_coef"] * sunlight_index +
            reg["intercept"]
        )
        
        # Get CSI from strategy config
        csi = self.strategy_config["MAGNIFICENT_MACARONS"]["CSI"]
        
        # Apply CSI adjustment if sunlight is below CSI
        if sunlight_index < csi:
            # Calculate adjustment based on how far below CSI
            csi_diff = csi - sunlight_index
            # Amplify price estimate when below CSI
            csi_premium = csi_diff * abs(reg["sunlight_coef"]) * 2.5
            fair_price += csi_premium
            print(f"Applied CSI premium: +{csi_premium:.2f} (sunlight {sunlight_index:.2f} vs CSI {csi:.2f})")
        
        return fair_price, csi

    def _macarons_strategy(self, state: TradingState, conversion_data: ConversionObservation) -> Tuple[List[Order], int]:
        orders = []
        conversion_amount = 0
        product = "MAGNIFICENT_MACARONS"
        position = state.position.get(product, 0)

        current_mid_price = (conversion_data.bidPrice + conversion_data.askPrice) / 2

        
        if "positions" not in self.history[product]:
            self.history[product]["positions"] = {
                "avg_entry_price": 0,
                "max_price_seen": current_mid_price,
                "stop_loss_pct": 0.10,
                "hard_stop_pct": 0.15,
                "holding_ticks": 0  # Add tick counter for long holding risk
            }

        # If we’re in a position, increment holding tick counter
        if position != 0:
            self.history[product]["positions"]["holding_ticks"] += 1
        else:
            self.history[product]["positions"]["holding_ticks"] = 0  # Reset on flat
        
        # Add price to history
        if conversion_data.sunlightIndex > self.strategy_config[product]["CSI"]:
            self.history[product]["prices"].append(current_mid_price)
        if len(self.history[product]["prices"]) > 1000:
            self.history[product]["prices"] = self.history[product]["prices"][-1000:]
        
        # Store observation as [sugarPrice, sunlightIndex, transportFees, exportTariff, importTariff, midPrice]
        current_observation = [
            conversion_data.sugarPrice,
            conversion_data.sunlightIndex, 
            conversion_data.transportFees,
            conversion_data.exportTariff,
            conversion_data.importTariff,
            current_mid_price
        ]
        
        self.history[product]["observations"].append(current_observation)
        if len(self.history[product]["observations"]) > 1000:
            self.history[product]["observations"] = self.history[product]["observations"][-1000:]
        
        # Update regression coefficients periodically
        if len(self.history[product]["observations"]) >= 50 and len(self.history[product]["observations"]) % 20 == 0:
            self._update_sunlight_sugar_regression(product)
        
        # Calculate effective prices for conversion
        effective_sell_price = conversion_data.bidPrice - conversion_data.transportFees - conversion_data.exportTariff
        effective_buy_price = conversion_data.askPrice + conversion_data.transportFees + conversion_data.importTariff
        
        # Get market order book
        order_depth = state.order_depths.get(product, OrderDepth())
        best_bid = max(order_depth.buy_orders.keys(), default=0)
        best_ask = min(order_depth.sell_orders.keys(), default=float('inf'))
        
        # Calculate actual midPrice from market
        actual_mid_price = (best_bid + best_ask) / 2 if best_bid > 0 and best_ask < float('inf') else current_mid_price
        
        # Estimate fair price using regression model
        fair_price, csi = self._estimate_fair_price(
            product, 
            conversion_data.sunlightIndex,
            conversion_data.sugarPrice
        )
        
        # If we don't have a regression model yet, use a simple approach
        if fair_price is None:
            fair_price = actual_mid_price
        
        # Calculate the price difference to determine if market is over/undervalued
        price_difference = actual_mid_price - fair_price
        
        # Position management
        position_limit = 75
        
        # Dynamic threshold based on observed volatility
        if "price_volatility" not in self.history[product]:
            self.history[product]["price_volatility"] = 5.0  # Initial value
        
        # Update volatility estimate (simple exponential moving average)
        if len(self.history[product]["prices"]) > 1:
            latest_return = abs(self.history[product]["prices"][-1] - self.history[product]["prices"][-2])
            alpha = 0.1  # Smoothing factor
            self.history[product]["price_volatility"] = (
                (1 - alpha) * self.history[product]["price_volatility"] + 
                alpha * latest_return
            )
        
        # Adaptive threshold based on volatility
        threshold = max(0.01 * fair_price, 0.4 * self.history[product]["price_volatility"])
        
        # Log values for debugging
        print(f"Sugar: {conversion_data.sugarPrice}, Sunlight: {conversion_data.sunlightIndex}")
        if "sunlight_sugar_regression" in self.history[product]:
            reg = self.history[product]["sunlight_sugar_regression"]
            print(f"Regression: {reg['sugar_coef']:.2f}*sugar + {reg['sunlight_coef']:.2f}*sunlight + {reg['intercept']:.2f}, R²: {reg['r_squared']:.4f}")
        print(f"Fair Price: {fair_price:.2f}, Actual Mid: {actual_mid_price:.2f}")
        print(f"Price difference: {price_difference:.2f}, Threshold: {threshold:.2f}")
        print(f"Current Volatility: {self.history[product]['price_volatility']:.2f}")
        
        # Check if sunlight is below CSI (critical trading condition)
        sunlight_below_csi = False
        if conversion_data.sunlightIndex < csi:
            sunlight_below_csi = True
            print(f"ALERT: Sunlight ({conversion_data.sunlightIndex:.2f}) below CSI ({csi:.2f})")
        
        # Calculate storage cost consideration (0.1 per unit per timestamp)
        storage_impact = 0.1  # Cost per unit per timestamp
        
        # Tracking entry prices and stop losses
        if "positions" not in self.history[product]:
            self.history[product]["positions"] = {
                "avg_entry_price": 0,
                "max_price_seen": current_mid_price,
                "stop_loss_pct": 0.10,  # 10% trailing stop
                "hard_stop_pct": 0.15   # 15% hard stop from entry
            }
        
        # Update max price seen (for trailing stop)
        if current_mid_price > self.history[product]["positions"]["max_price_seen"]:
            self.history[product]["positions"]["max_price_seen"] = current_mid_price
        
        # TRADING LOGIC
        
        # CASE 1: Sunlight below CSI - aggressive long strategy
        if sunlight_below_csi:
            # Calculate how aggressively to buy based on how far below CSI
            csi_diff = csi - conversion_data.sunlightIndex
            aggression_factor = min(1.0, csi_diff / 10)  # Scale based on difference
            
            # Target position as percentage of limit based on CSI difference
            target_position_pct = min(1.0, 0.55 + aggression_factor * 0.8)
            target_position = int(position_limit * target_position_pct)
            
            print(f"CSI Strategy: Target position {target_position} units ({target_position_pct:.1%} of limit)")
            
            # Only buy if we don't have enough already
            if position < target_position:
                buy_quantity = target_position - position
                
                # Check if there are sell orders to match against
                if best_ask < float('inf'):
                    available_quantity = abs(order_depth.sell_orders.get(best_ask, 0))
                    buy_quantity = min(buy_quantity, available_quantity)

                    if buy_quantity > 0:
                        orders.append(Order(product, best_ask, buy_quantity))
                        print(f"CSI Strategy: Buying {buy_quantity} at {best_ask}")
                    else:
                        # No market liquidity – fallback to conversion
                        effective_buy_price = conversion_data.askPrice + conversion_data.transportFees + conversion_data.importTariff
                        if effective_buy_price < fair_price + threshold:
                            conversion_amount = min(10, target_position - position)
                            print(f"No market depth. Converting to buy {conversion_amount} units at {effective_buy_price:.2f}")


        # === CASE 2 : NORMAL‑MARKET — MEAN‑REVERSION, SHORT‑ONLY ===
        else:
            # -------- SHORT when price is well above fair value --------
            if best_bid > 0 and price_difference > 1.5 * threshold:
                # Size grows with mispricing but capped by limits
                sell_aggr = min(1.0, price_difference / (3 * threshold))
                target_short = int(position_limit * sell_aggr)          # desired absolute short
                if position > 0:                                        # if still long, flatten first
                    target_short = max(0, target_short - position)
                short_qty = min(target_short, order_depth.buy_orders.get(best_bid, 0))

                if short_qty > 0:
                    orders.append(Order(product, best_bid, -short_qty))
                    print(f"NM SHORT: Selling {short_qty} @ {best_bid} (Δ={price_difference:.2f})")

            # -------- COVER shorts when price normalises --------
            if position < 0 and price_difference < -0.3 * threshold:
                cover_qty = min(abs(position), abs(order_depth.sell_orders.get(best_ask, 0)))
                if cover_qty > 0 and best_ask < float('inf'):
                    orders.append(Order(product, best_ask, cover_qty))
                    print(f"NM COVER: Buying {cover_qty} @ {best_ask} (back to fair)")

                
        
        
        # Check stop loss conditions if we have a long position
        if position > 0:
            pos_info = self.history[product]["positions"]
            
            # Calculate stop prices
            trailing_stop = pos_info["max_price_seen"] * (1 - pos_info["stop_loss_pct"])
            hard_stop = pos_info["avg_entry_price"] * (1 - pos_info["hard_stop_pct"]) if pos_info["avg_entry_price"] > 0 else 0
            
            # Check if stop is triggered and we're not in crisis mode (below CSI)
            # Track sunlight recovery time
            if conversion_data.sunlightIndex > csi:
                if "recovery_since" not in self.history[product]:
                    self.history[product]["recovery_since"] = state.timestamp
            else:
                self.history[product].pop("recovery_since", None)

            # Stop loss allowed only if sunlight has been above CSI for 10+ ticks
            if position > 0 and "recovery_since" in self.history[product]:
                if state.timestamp - self.history[product]["recovery_since"] > 10:
                    stop_loss_triggered = (actual_mid_price < trailing_stop or actual_mid_price < hard_stop)
                    hold_too_long = self.history[product]["positions"]["holding_ticks"] >= 50

                    if stop_loss_triggered or hold_too_long:
                        reason = "TIMEOUT" if hold_too_long else "STOP LOSS"
                        if best_bid > 0:
                            print(f"{reason} TRIGGERED: Selling {position} at {best_bid}, holding {self.history[product]['positions']['holding_ticks']} ticks")
                            orders.append(Order(product, best_bid, -position))
                            self.history[product]["positions"]["holding_ticks"] = 0

        
        # CONVERSION LOGIC
        
        # Sell via conversion if we have a long position and it's profitable
        if position > 0 and effective_sell_price > best_bid:
            # Calculate profit including storage cost consideration 
            conversion_profit = effective_sell_price - (fair_price - threshold)
            
            if conversion_profit > 0:
                max_sell_conversion = min(10, position)
                if max_sell_conversion > 0:
                    conversion_amount = -max_sell_conversion  # Negative for selling
                    print(f"Converting to sell {max_sell_conversion} units at {effective_sell_price:.2f}")
        
        # Buy via conversion if we have a short position and it's profitable
        if position < 0 and effective_buy_price < best_ask:
            # Calculate profit including storage cost consideration
            conversion_profit = (best_ask + storage_impact) - effective_buy_price
            
            if conversion_profit > 0:
                max_buy_conversion = min(10, abs(position))
                if max_buy_conversion > 0:
                    conversion_amount = max_buy_conversion  # Positive for buying
                    print(f"Converting to buy {max_buy_conversion} units at {effective_buy_price:.2f}")
        
        # Make sure we're not requesting both buy and sell conversions
        # Conversion request should be:
        # - Positive (1 to 10) to BUY when we have a SHORT position
        # - Negative (-1 to -10) to SELL when we have a LONG position
        # - Zero when no conversion is needed or when we have no position
        
        # Important: Only allow conversion in the valid direction for current position
        if (position > 0 and conversion_amount > 0) or (position < 0 and conversion_amount < 0):
            print(f"WARNING: Invalid conversion request {conversion_amount} for position {position}. Setting to 0.")
            conversion_amount = 0
        
        # Ensure conversion is within limits
        conversion_amount = max(-10, min(10, conversion_amount))
        
        # Update average entry price based on new orders
        if len(orders) > 0:
            total_cost = 0
            total_quantity = 0
            for order in orders:
                if order.quantity > 0:  # Buy orders
                    total_cost += order.price * order.quantity
                    total_quantity += order.quantity
            
            if total_quantity > 0:
                current_position = position
                current_avg_price = self.history[product]["positions"]["avg_entry_price"]
                
                # Update average entry price
                if current_position <= 0:
                    # New position, set avg price directly
                    self.history[product]["positions"]["avg_entry_price"] = total_cost / total_quantity
                else:
                    # Update existing position with weighted average
                    self.history[product]["positions"]["avg_entry_price"] = (
                        (current_avg_price * current_position + total_cost) / 
                        (current_position + total_quantity)
                    )
        
        return orders, conversion_amount

    def run(self, state: TradingState) -> Tuple[Dict[str, List[Order]], int, str]:
        self.current_round += 1  # Increment round counter
        result = {product: [] for product in self.strategy_config}
        conversion_requests = 0 # Initialize conversion requests

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

        # Handle MAGNIFICENT_MACARONS
        if "MAGNIFICENT_MACARONS" in state.observations.conversionObservations:
            conversion_data = state.observations.conversionObservations["MAGNIFICENT_MACARONS"]
            macaron_orders, macaron_conversion = self._macarons_strategy(state, conversion_data)
            result["MAGNIFICENT_MACARONS"].extend(macaron_orders)
            conversion_requests = macaron_conversion

        # History maintenance
        for product in self.history:
            if "prices" in self.history[product]:
                self.history[product]["prices"] = self.history[product]["prices"][-1000:]

        return result, conversion_requests, state.traderData
