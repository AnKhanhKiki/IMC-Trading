# Algorithmic Trading Bot for Prosperity Simulation

This repository contains a sophisticated, multi-product algorithmic trading bot written in Python for the **Prosperity trading simulation**. The bot employs a variety of strategies, from classic statistical arbitrage and mean reversion to more complex options pricing and regression-based models, to trade a diverse portfolio of assets.

---

## 🚀 Core Features
- **Multi-Product Strategy**: Implements unique, tailored trading logic for 10+ different products.
- **Diverse Models**: Utilizes a range of financial models, including:
  - VWAP-based market making
  - Mean Reversion (SMA & EMA)
  - Statistical Arbitrage for ETF-like baskets
  - Black-Scholes options pricing for vouchers
  - Multivariate Linear Regression for commodity price prediction
- **Dynamic Configuration**: All strategy parameters are centralized in a `strategy_config` dictionary for easy tuning and maintenance.
- **State Management**: Maintains a comprehensive history of prices and indicators for each product to inform trading decisions.
- **Risk Management**: Incorporates position limits, dynamic order sizing, and stop-loss mechanisms to manage risk.

---

## 📈 Product Strategies

### 🥥 Rainforest Resin
- **Strategy**: VWAP-Based Market Making  
- **Logic**: Calculates the Volume-Weighted Average Price (VWAP) from the order book. It then places bid and ask orders around this VWAP, adjusted by a dynamic spread. The strategy also adjusts pricing based on current inventory to mitigate risk, lowering bid/ask prices when long and raising them when short.

### 🌱 Kelp
- **Strategy**: Mean Reversion with EMA  
- **Logic**: Calculates the Exponential Moving Average (EMA) of the mid-price to establish a short-term fair value. When the current mid-price deviates significantly from the EMA, the bot places orders assuming the price will revert to the mean.

### 🦑 Squid Ink
- **Strategy**: Mean Reversion with SMA  
- **Logic**: Uses a Simple Moving Average (SMA) to identify trading opportunities. If the market price moves too far from its historical average, the bot enters a counter-trend position.

### 🧺 Picnic Baskets (PICNIC_BASKET1 & PICNIC_BASKET2)
- **Strategy**: Statistical Arbitrage  
- **Logic**:  
  - These baskets consist of other goods (CROISSANTS, JAMS, DJEMBES).  
  - The strategy calculates the fair value of the basket by summing the real-time prices of its components.  
  - If **Basket Price < Component Sum**, the bot buys the basket and sells the components.  
  - If **Basket Price > Component Sum**, the bot sells the basket and buys the components.  
  - Order sizing is dynamic and respects position limits for both the basket and its components.  

### 🌋 Volcanic Rock Vouchers
- **Strategy**: Options Pricing (Black-Scholes Model)  
- **Logic**:  
  - Treats vouchers as European call options on Volcanic Rock.  
  - **Volatility**: Calculated from historical log returns, smoothed with EMA.  
  - **Fair Value**: Black-Scholes formula applied with strike price, underlying price, time to expiry, and volatility.  
  - **Execution**: Trades when the market price deviates from the theoretical value, using buffers adjusted for trend, moneyness, and volatility.  

### 🍰 Magnificent Macarons
- **Strategy**: Regression & Event-Driven  
- **Logic**:  
  - Runs a **multivariate linear regression** in real-time:  

    ```
    midPrice = β1 ⋅ sunlightIndex + β2 ⋅ sugarPrice + c
    ```

  - **Critical Sunlight Index (CSI)**:  
    - When `sunlightIndex < CSI (45)`, triggers an aggressive **long-only strategy** anticipating a supply shock.  
    - When `sunlightIndex ≥ CSI`, reverts to regression-based fair value trading and implements a **post-panic shorting** strategy.  

- **Risk Management**:
  - Trailing stop-loss based on max price seen.
  - Hard stop-loss based on average entry price.
  - Time-based stop for stale positions.

---

## 📂 Code Structure
- **`Trader` class**: Encapsulates all trading logic.
  - `__init__(self)`: Initializes strategy configs and historical data structures.
  - `run(self, state: TradingState)`: Main entry point. Iterates through products, applies strategies, aggregates orders.
- **Strategy Functions**: Private methods (`_macarons_strategy`, `_volcanic_voucher_strategy`, etc.) handle product-specific logic.
- **Data Classes**: `Order`, `TradingState`, `ConversionObservation`, etc. define structures for simulation environment.

---

## 🛠️ Dependencies
The bot only uses standard Python libraries:

```bash
pip install numpy pandas jsonpickle


