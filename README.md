# Advanced Algorithmic Trading Strategies for a Simulated Market

## 📜 Abstract
This repository contains the source code for a high-frequency algorithmic trading bot designed for the Prosperity simulated market environment. The bot implements a portfolio of sophisticated, product-specific quantitative strategies to achieve its objective. The methodologies employed range from econometric analysis for price prediction and statistical arbitrage on co-integrated assets to the application of stochastic calculus for derivatives pricing. This document provides a detailed mathematical exposition of the models underpinning each trading strategy, their implementation, and the overarching risk management framework.

---

## 🧠 Core Mathematical Framework
The bot operates on a discrete-time basis, receiving a `TradingState` snapshot at each time step *t*.  
The primary goal is to identify and exploit temporary market inefficiencies or predictive signals derived from market data and exogenous variables.  
The core logic is encapsulated within the `Trader` class, which manages state, configuration, and the execution of product-specific models.

---

## 📈 Product-Specific Quantitative Strategies

### 🥥 Rainforest Resin  
**Model:** Inventory-Adjusted VWAP Market Making

**Mathematical Formulation:** This strategy is a simplified implementation of classic market-making models. The core is to establish a fair value baseline and quote symmetrically around it, adjusting for inventory risk.

**Fair Value Estimation:**  

![arb_buy](./math_svgs/arb_buy.svg)

