Advanced Algorithmic Trading Strategies for a Simulated Market
📜 Abstract
This repository contains the source code for a high-frequency algorithmic trading bot designed for the Prosperity simulated market environment. The bot implements a portfolio of sophisticated, product-specific quantitative strategies to achieve its objective. The methodologies employed range from econometric analysis for price prediction and statistical arbitrage on co-integrated assets to the application of stochastic calculus for derivatives pricing. This document provides a detailed mathematical exposition of the models underpinning each trading strategy, their implementation, and the overarching risk management framework.

🧠 Core Mathematical Framework
The bot operates on a discrete-time basis, receiving a TradingState snapshot at each time step t. The primary goal is to identify and exploit temporary market inefficiencies or predictive signals derived from market data and exogenous variables. The core logic is encapsulated within the Trader class, which manages state, configuration, and the execution of product-specific models.

📈 Product-Specific Quantitative Strategies
Each financial instrument is traded using a bespoke model tailored to its unique statistical properties and market microstructure.

🥥 Rainforest Resin
Model: Inventory-Adjusted VWAP Market Making

Mathematical Formulation: This strategy is a simplified implementation of classic market-making models. The core is to establish a fair value baseline and quote symmetrically around it, adjusting for inventory risk.

Fair Value Estimation: The fair value, P_fair, is estimated using the Volume-Weighted Average Price (VWAP) of the current order book:
P_{\text{fair}}(t) = \text{VWAP} = \frac{\sum_{i \in \text{book}} P_i \cdot V_i}{\sum_{i \in \text{book}} V_i}
where P_i and V_i are the price and volume of each order in the book.

Optimal Quoting: The initial bid (P_bid) and ask (P_ask) prices are set as a fraction of the market spread (δ=P_best_ask−P_best_bid) around the fair value:
P_{\text{bid}}(t) = P_{\text{fair}}(t) - \delta(t) \cdot r_{\text{spread}}
P_{\text{ask}}(t) = P_{\text{fair}}(t) + \delta(t) \cdot r_{\text{spread}}
where r_spread is a configurable ratio parameter.

Inventory Risk Adjustment: To manage inventory risk, a linear penalty term is applied to the quotes. The adjustment skews the quotes to incentivize trades that reduce the current position (I_t). The final, adjusted quotes (P 
′
 ∗bid,P 
′
 ∗ask) are:
P'_{\text{bid}}(t) = P_{\text{bid}}(t) - \lambda \cdot I_t
P'_{\text{ask}}(t) = P_{\text{ask}}(t) - \lambda \cdot I_t
where I_t is the inventory (position) at time t and λ is the inventory penalty coefficient.

Of course. The errors you encountered are likely because GitHub's standard Markdown renderer requires a specific format for mathematical equations, rather than the standard LaTeX $ delimiters.

Here is the corrected version of the detailed README. You can copy and paste this entire block directly into your GitHub README file, and the equations should render correctly.

Advanced Algorithmic Trading Strategies for a Simulated Market
📜 Abstract
This repository contains the source code for a high-frequency algorithmic trading bot designed for the Prosperity simulated market environment. The bot implements a portfolio of sophisticated, product-specific quantitative strategies to achieve its objective. The methodologies employed range from econometric analysis for price prediction and statistical arbitrage on co-integrated assets to the application of stochastic calculus for derivatives pricing. This document provides a detailed mathematical exposition of the models underpinning each trading strategy, their implementation, and the overarching risk management framework.

🧠 Core Mathematical Framework
The bot operates on a discrete-time basis, receiving a TradingState snapshot at each time step t. The primary goal is to identify and exploit temporary market inefficiencies or predictive signals derived from market data and exogenous variables. The core logic is encapsulated within the Trader class, which manages state, configuration, and the execution of product-specific models.

📈 Product-Specific Quantitative Strategies
Each financial instrument is traded using a bespoke model tailored to its unique statistical properties and market microstructure.

🥥 Rainforest Resin
Model: Inventory-Adjusted VWAP Market Making

Mathematical Formulation: This strategy is a simplified implementation of classic market-making models. The core is to establish a fair value baseline and quote symmetrically around it, adjusting for inventory risk.

Fair Value Estimation: The fair value, P_fair, is estimated using the Volume-Weighted Average Price (VWAP) of the current order book:

Code snippet

P_{\text{fair}}(t) = \text{VWAP} = \frac{\sum_{i \in \text{book}} P_i \cdot V_i}{\sum_{i \in \text{book}} V_i}
where P_i and V_i are the price and volume of each order in the book.

Optimal Quoting: The initial bid (P_bid) and ask (P_ask) prices are set as a fraction of the market spread (δ=P_best_ask−P_best_bid) around the fair value:

Code snippet

P_{\text{bid}}(t) = P_{\text{fair}}(t) - \delta(t) \cdot r_{\text{spread}}
Code snippet

P_{\text{ask}}(t) = P_{\text{fair}}(t) + \delta(t) \cdot r_{\text{spread}}
where r_spread is a configurable ratio parameter.

Inventory Risk Adjustment: To manage inventory risk, a linear penalty term is applied to the quotes. The adjustment skews the quotes to incentivize trades that reduce the current position (I_t). The final, adjusted quotes (P 
′
 ∗bid,P 
′
 ∗ask) are:

Code snippet

P'_{\text{bid}}(t) = P_{\text{bid}}(t) - \lambda \cdot I_t
Code snippet

P'_{\text{ask}}(t) = P_{\text{ask}}(t) - \lambda \cdot I_t
where I_t is the inventory (position) at time t and λ is the inventory penalty coefficient.

🌱 Kelp & 🦑 Squid Ink
Model: Stochastic Mean Reversion

Mathematical Formulation: These strategies model the mid-price process, P_t, as a discrete-time process that tends to revert to a moving average, μ_t.

Moving Average Estimation (μ_t): The long-term mean μ_t is estimated using either a Simple Moving Average (SMA) or an Exponential Moving Average (EMA).

SMA:
\mu_t = \text{SMA}_N(t) = \frac{1}{N} \sum_{i=0}^{N-1} P_{t-i}
EMA:
\mu_t = \text{EMA}_N(t) = \alpha \cdot P_t + (1 - \alpha) \cdot \text{EMA}_N(t-1)
where \alpha = 2 / (N+1).
Trading Signal: A trading signal is generated when the normalized deviation of the current price from the estimated mean exceeds a predefined threshold, θ.
\text{Signal} = \frac{P_t - \mu_t}{\mu_t}
If Signal > θ, the asset is considered overvalued (sell). If Signal < −θ, it's considered undervalued (buy).

Of course. The errors you encountered are likely because GitHub's standard Markdown renderer requires a specific format for mathematical equations, rather than the standard LaTeX $ delimiters.

Here is the corrected version of the detailed README. You can copy and paste this entire block directly into your GitHub README file, and the equations should render correctly.

Advanced Algorithmic Trading Strategies for a Simulated Market
📜 Abstract
This repository contains the source code for a high-frequency algorithmic trading bot designed for the Prosperity simulated market environment. The bot implements a portfolio of sophisticated, product-specific quantitative strategies to achieve its objective. The methodologies employed range from econometric analysis for price prediction and statistical arbitrage on co-integrated assets to the application of stochastic calculus for derivatives pricing. This document provides a detailed mathematical exposition of the models underpinning each trading strategy, their implementation, and the overarching risk management framework.

🧠 Core Mathematical Framework
The bot operates on a discrete-time basis, receiving a TradingState snapshot at each time step t. The primary goal is to identify and exploit temporary market inefficiencies or predictive signals derived from market data and exogenous variables. The core logic is encapsulated within the Trader class, which manages state, configuration, and the execution of product-specific models.

📈 Product-Specific Quantitative Strategies
Each financial instrument is traded using a bespoke model tailored to its unique statistical properties and market microstructure.

🥥 Rainforest Resin
Model: Inventory-Adjusted VWAP Market Making

Mathematical Formulation: This strategy is a simplified implementation of classic market-making models. The core is to establish a fair value baseline and quote symmetrically around it, adjusting for inventory risk.

Fair Value Estimation: The fair value, P_fair, is estimated using the Volume-Weighted Average Price (VWAP) of the current order book:

Code snippet

P_{\text{fair}}(t) = \text{VWAP} = \frac{\sum_{i \in \text{book}} P_i \cdot V_i}{\sum_{i \in \text{book}} V_i}
where P_i and V_i are the price and volume of each order in the book.

Optimal Quoting: The initial bid (P_bid) and ask (P_ask) prices are set as a fraction of the market spread (δ=P_best_ask−P_best_bid) around the fair value:

Code snippet

P_{\text{bid}}(t) = P_{\text{fair}}(t) - \delta(t) \cdot r_{\text{spread}}
Code snippet

P_{\text{ask}}(t) = P_{\text{fair}}(t) + \delta(t) \cdot r_{\text{spread}}
where r_spread is a configurable ratio parameter.

Inventory Risk Adjustment: To manage inventory risk, a linear penalty term is applied to the quotes. The adjustment skews the quotes to incentivize trades that reduce the current position (I_t). The final, adjusted quotes (P 
′
 ∗bid,P 
′
 ∗ask) are:

Code snippet

P'_{\text{bid}}(t) = P_{\text{bid}}(t) - \lambda \cdot I_t
Code snippet

P'_{\text{ask}}(t) = P_{\text{ask}}(t) - \lambda \cdot I_t
where I_t is the inventory (position) at time t and λ is the inventory penalty coefficient.

🌱 Kelp & 🦑 Squid Ink
Model: Stochastic Mean Reversion

Mathematical Formulation: These strategies model the mid-price process, P_t, as a discrete-time process that tends to revert to a moving average, μ_t.

Moving Average Estimation (μ_t): The long-term mean μ_t is estimated using either a Simple Moving Average (SMA) or an Exponential Moving Average (EMA).

SMA:

Code snippet

\mu_t = \text{SMA}_N(t) = \frac{1}{N} \sum_{i=0}^{N-1} P_{t-i}
EMA:

Code snippet

\mu_t = \text{EMA}_N(t) = \alpha \cdot P_t + (1 - \alpha) \cdot \text{EMA}_N(t-1)
where \alpha = 2 / (N+1).

Trading Signal: A trading signal is generated when the normalized deviation of the current price from the estimated mean exceeds a predefined threshold, θ.

Code snippet

\text{Signal} = \frac{P_t - \mu_t}{\mu_t}
If Signal > θ, the asset is considered overvalued (sell). If Signal < −θ, it's considered undervalued (buy).

🧺 Picnic Baskets
Model: Statistical Arbitrage on a Synthetic ETF

Mathematical Formulation: The strategy assumes the basket price is a linear combination of its component prices. The spread between the basket's market price and its theoretical value is modeled as a mean-reverting process.

Theoretical Basket Value: The theoretical bid and ask values of the basket are calculated from the component markets. Let the basket be composed of k components with quantities q_1,...,q_k.

Cost to Create (Composite Ask):

P_{\text{comp\_ask}}(t) = \sum_{i=1}^{k} q_i \cdot P_{C_i, \text{ask}}(t)

Value to Dismantle (Composite Bid):

P_{\text{comp\_bid}}(t) = \sum_{i=1}^{k} q_i \cdot P_{C_i, \text{bid}}(t)

Arbitrage Conditions:

Buy Arbitrage (Buy Basket, Sell Components): Execute if $P\_{B, \text{ask}}(t) \< P\_{\text{comp\_bid}}(t)$.

Sell Arbitrage (Sell Basket, Buy Components): Execute if P_B,bid(t)P_comp_ask(t).

🌋 Volcanic Rock Vouchers
Model: Black-Scholes-Merton Option Pricing

Mathematical Formulation: The voucher is treated as a European call option. Its theoretical price, C(S_t,t), is calculated using the Black-Scholes-Merton (BSM) formula. The risk-free rate r is assumed to be zero.

Volatility Estimation (σ): The volatility of the underlying asset (VOLCANIC_ROCK) is estimated from the standard deviation of historical logarithmic returns.

r_t = \ln\left(\frac{S_t}{S_{t-1}}\right)
\sigma_{\text{annual}} = \text{StDev}(r_{t}, \ldots, r_{t-N}) \cdot \sqrt{365}

BSM Formula: The fair value of the call option is:
C(S_t, t) = S_t \Phi(d_1) - K e^{-r(T-t)} \Phi(d_2)
where:
d_1 = \frac{\ln(S_t/K) + (r + \frac{\sigma^2}{2})(T-t)}{\sigma\sqrt{T-t}}
d_2 = d_1 - \sigma\sqrt{T-t}
S_t: Current price of the underlying asset.

K: Strike price of the voucher.

T−t: Time to expiration in years.

σ: Annualized volatility.

Φ(⋅): The CDF of the standard normal distribution.

🍰 Magnificent Macarons
Model: Multivariate Regression with a Non-Linear Event Trigger

Mathematical Formulation: This strategy uses an econometric model to predict the fair price, but switches to a behavioral, event-driven model under specific conditions.

Fair Value Econometric Model: The fair price  
P
^
 _t is estimated using Ordinary Least Squares (OLS) regression.
P_t = \beta_0 + \beta_1 X_{1,t} + \beta_2 X_{2,t} + \epsilon_t

where X_1,t is sugarPrice and X_2,t is sunlightIndex. The vector of estimated coefficients  
β
^
​
  is found via the normal equation:

\hat{\beta} = (X^T X)^{-1} X^T Y
The model's fit is evaluated using the coefficient of determination, R 
2
 :
R^2 = 1 - \frac{\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}{\sum_{i=1}^{n}(y_i - \bar{y})^2}

Event-Driven Adjustment (CSI): The model incorporates a non-linear adjustment based on the Critical Sunlight Index (CSI). The adjusted fair price,  
P
^
  
′
 _t, is:
 \hat{P}'_t = 
\begin{cases} 
\hat{P}_t + (\text{CSI} - X_{2,t}) \cdot |\hat{\beta}_2| \cdot k & \text{if } X_{2,t} < \text{CSI} \\ 
\hat{P}_t & \text{otherwise} 
\end{cases}

where k is an amplifier multiplier (k=2.5).

⚙️ Risk Management Framework
Position Limits: A hard constraint is applied to the inventory I_t for each product j:

|I_{t,j}| \le I_{\text{max},j}

Stop-Loss Orders: For the high-risk Macarons strategy, a path-dependent exit rule is implemented.

Trailing Stop:

P_t < \left(\max_{i \in [t_0, t]} P_i\right) \cdot (1 - \text{SL}_{\%})

Hard Stop:
P_t < P_{\text{entry}} \cdot (1 - \text{SL}_{\text{hard}\%})

where t_0 is the time of entry.

💻 Code Architecture
Trader class: The central controller that orchestrates all operations.

__init__(self): Initializes strategy_config (hyperparameters) and history (state variables).

run(self, state: TradingState): The main event loop that ingests market data, invokes the relevant quantitative models, and dispatches orders.

Strategy Functions: Private methods within Trader that contain the implementation of the mathematical models described above.

🛠️ Dependencies
The project requires the following standard Python libraries for numerical computing:
pip install numpy pandas jsonpickle
