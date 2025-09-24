IMC TRADING PROSPERITY CHALLENGE
📜 Abstract
This repository contains the source code for a high-frequency algorithmic trading bot designed for the Prosperity simulated market environment. The bot implements a portfolio of sophisticated, product-specific quantitative strategies to achieve its objective. The methodologies employed range from econometric analysis for price prediction and statistical arbitrage on co-integrated assets to the application of stochastic calculus for derivatives pricing. This document provides a detailed mathematical exposition of the models underpinning each trading strategy, their implementation, and the overarching risk management framework.

🧠 Core Mathematical Framework
The bot operates on a discrete-time basis, receiving a TradingState snapshot at each time step t. The primary goal is to identify and exploit temporary market inefficiencies or predictive signals derived from market data and exogenous variables. The core logic is encapsulated within the Trader class, which manages state, configuration, and the execution of product-specific models.

📈 Product-Specific Quantitative Strategies
Each financial instrument is traded using a bespoke model tailored to its unique statistical properties and market microstructure.

🥥 Rainforest Resin
Model: Inventory-Adjusted VWAP Market Making

Mathematical Formulation: This strategy is a simplified implementation of classic market-making models (e.g., Avellaneda-Stoikov). The core is to establish a fair value baseline and quote symmetrically around it, adjusting for inventory risk.

Fair Value Estimation: The fair value, P_fair, is estimated using the Volume-Weighted Average Price (VWAP) of the current order book:

P 
fair
​
 (t)=VWAP= 
∑ 
i∈book
​
 V 
i
​
 
∑ 
i∈book
​
 P 
i
​
 ⋅V 
i
​
 
​
 

where P_i and V_i are the price and volume of each order in the book.

Optimal Quoting: The initial bid (P_bid) and ask (P_ask) prices are set as a fraction of the market spread (δ=P_best_ask−P_best_bid) around the fair value:

P 
bid
​
 (t)=P 
fair
​
 (t)−δ(t)⋅r 
spread
​
 
P 
ask
​
 (t)=P 
fair
​
 (t)+δ(t)⋅r 
spread
​
 

where r_spread is a configurable ratio parameter.

Inventory Risk Adjustment: To manage inventory risk, a linear penalty term is applied to the quotes. The adjustment skews the quotes to incentivize trades that reduce the current position (I_t). The final, adjusted quotes (P 
′
 ∗bid,P 
′
 ∗ask) are:

P 
bid
′
​
 (t)=P 
bid
​
 (t)−λ⋅I 
t
​
 
P 
ask
′
​
 (t)=P 
ask
​
 (t)−λ⋅I 
t
​
 

where I_t is the inventory (position) at time t and λ is the inventory penalty coefficient. A positive I_t (long position) lowers both bid and ask prices, making it more likely to sell and less likely to buy.

🌱 Kelp & 🦑 Squid Ink
Model: Stochastic Mean Reversion

Mathematical Formulation: These strategies model the mid-price process, P_t, as a discrete-time Ornstein-Uhlenbeck process, which tends to revert to a moving average, μ_t.

Moving Average Estimation (μ_t): The long-term mean μ_t is estimated using either a Simple Moving Average (SMA) for Squid Ink or an Exponential Moving Average (EMA) for Kelp.

SMA: μ_t=SMA∗N(t)= 
N
1
​
 ∑∗i=0 
N−1
 P_t−i

EMA: $ \mu_t = \text{EMA}_N(t) = \alpha \cdot P_t + (1 - \alpha) \cdot \text{EMA}_N(t-1) $, where α= 
N+1
2
​
 .

Trading Signal: A trading signal is generated when the normalized deviation of the current price from the estimated mean exceeds a predefined threshold, θ.

Signal= 
μ 
t
​
 
P 
t
​
 −μ 
t
​
 
​
 
If Signal > θ, the asset is considered overvalued, and a short position is initiated.

If Signal < −θ, the asset is considered undervalued, and a long position is initiated.

🧺 Picnic Baskets (PICNIC_BASKET1 & PICNIC_BASKET2)
Model: Statistical Arbitrage on a Synthetic ETF

Mathematical Formulation: The strategy assumes that the basket price, P_B,t, is a linear combination of its component prices, forming a co-integrated relationship. The spread between the basket's market price and its theoretical value is modeled as a mean-reverting process.

Theoretical Basket Value: The theoretical bid and ask values of the basket are calculated from the component markets. Let the basket be composed of k components with quantities q_1,q_2,...,q_k.

Cost to Create (Composite Ask): P_comp_ask(t)=∑_i=1 
k
 q_i⋅P_C_i,ask(t)

Value to Dismantle (Composite Bid): P_comp_bid(t)=∑_i=1 
k
 q_i⋅P_C_i,bid(t)

Arbitrage Conditions: An arbitrage opportunity exists if the cost of a round-trip transaction is negative (i.e., yields a profit).

Buy Arbitrage (Buy Basket, Sell Components): Execute if $P\_{B, \text{ask}}(t) \< P\_{\text{comp\_bid}}(t)$. The theoretical profit per basket is P_comp_bid(t)−P_B,ask(t).

Sell Arbitrage (Sell Basket, Buy Components): Execute if P_B,bid(t)P_comp_ask(t). The theoretical profit per basket is P_B,bid(t)−P_comp_ask(t).

🌋 Volcanic Rock Vouchers
Model: Black-Scholes-Merton Option Pricing

Mathematical Formulation: The voucher is treated as a European call option. Its theoretical price, C(S_t,t), is calculated using the Black-Scholes-Merton (BSM) formula. The risk-free rate r is assumed to be zero.

Volatility Estimation (σ): The volatility of the underlying asset (VOLCANIC_ROCK) is estimated from the standard deviation of historical logarithmic returns.

r 
t
​
 =ln( 
S 
t−1
​
 
S 
t
​
 
​
 )
σ 
daily
​
 =StDev(r 
t
​
 ,r 
t−1
​
 ,…,r 
t−N
​
 )
σ 
annual
​
 =σ 
daily
​
 ⋅ 
365

​
 
BSM Formula: The fair value of the call option is:

C(S 
t
​
 ,t)=S 
t
​
 Φ(d 
1
​
 )−Ke 
−r(T−t)
 Φ(d 
2
​
 )

where:

d 
1
​
 = 
σ 
T−t

​
 
ln(S 
t
​
 /K)+(r+ 
2
σ 
2
 
​
 )(T−t)
​
 
d 
2
​
 =d 
1
​
 −σ 
T−t

​
 
S_t: Current price of the underlying asset (VOLCANIC_ROCK).

K: Strike price of the voucher.

T−t: Time to expiration in years.

σ: Annualized volatility of the underlying.

r: Risk-free interest rate (assumed 0).

Φ(⋅): The cumulative distribution function (CDF) of the standard normal distribution. The code uses a high-precision polynomial approximation for this function.

Trading Logic: The bot compares the calculated fair value C(S_t,t) to the voucher's market mid-price. If the discrepancy exceeds a dynamic buffer, it places orders to capitalize on the perceived mispricing.

🍰 Magnificent Macarons
Model: Multivariate Regression with a Non-Linear Event Trigger

Mathematical Formulation: This strategy uses an econometric model to predict the fair price based on exogenous factors, but switches to a behavioral, event-driven model under specific conditions.

Fair Value Econometric Model: The fair price  
P
^
 ∗t is estimated using Ordinary Least Squares (OLS) regression. The model is:

P 
t
​
 =β 
0
​
 +β 
1
​
 X 
1,t
​
 +β 
2
​
 X 
2,t
​
 +ϵ 
t
​
 

where X∗1,t is sugarPrice and X_2,t is sunlightIndex. In matrix form, for a set of n observations, Y=Xβ+ϵ. The vector of estimated coefficients  
β
^
​
  is found via the normal equation:

β
^
​
 =(X 
T
 X) 
−1
 X 
T
 Y

The model's fit is evaluated using the coefficient of determination, R 
2
 :

R 
2
 =1− 
∑ 
i=1
n
​
 (y 
i
​
 − 
y
ˉ
​
 ) 
2
 
∑ 
i=1
n
​
 (y 
i
​
 − 
y
^
​
  
i
​
 ) 
2
 
​
 
Event-Driven Adjustment (CSI): The model incorporates a non-linear adjustment based on the Critical Sunlight Index (CSI). This captures the market's panic-buying behavior during a perceived supply shock. The adjusted fair price,  
P
^
  
′
 _t, is:

P
^
  
t
′
​
 ={ 
P
^
  
t
​
 +(CSI−X 
2,t
​
 )⋅∣ 
β
^
​
  
2
​
 ∣⋅k
P
^
  
t
​
 
​
  
if X 
2,t
​
 <CSI
otherwise
​
 

where k is a multiplier (k=2.5) that amplifies the price impact when sunlight is critically low. This transforms the linear model into a piecewise function, aggressively increasing the price target during a supply shock event.

Execution:

Below CSI: An aggressive long-only strategy is deployed to build a position up to a target size, anticipating a price spike.

Above CSI: The bot trades based on deviations from the regression-predicted fair price  
P
^
 _t. It also deploys a mean-reversion strategy to short the asset if it becomes significantly overvalued after a CSI event, fading the panic.

⚙️ Risk Management Framework
Position Limits: A hard constraint is applied to the inventory I_t for each product:

∣I 
t,j
​
 ∣≤I 
max,j
​
 ∀j∈{products}
Stop-Loss Orders: For the high-risk Macarons strategy, a path-dependent exit rule is implemented. A position is liquidated if the current price P_t breaches a threshold determined by the historical path of prices since the position was opened.

Trailing Stop: $P\_t \< \left(\max\_{i \in [t\_0, t]} P\_i\right) \cdot (1 - \text{SL}\_{%})$

Hard Stop: $P\_t \< P\_{\text{entry}} \cdot (1 - \text{SL}\_{\text{hard}%})$
where t_0 is the time of entry.

💻 Code Architecture
Trader class: The central controller that orchestrates all operations.

__init__(self): Initializes strategy_config (hyperparameters) and history (state variables).

run(self, state: TradingState): The main event loop that ingests market data, invokes the relevant quantitative models, and dispatches orders.

Strategy Functions: Private methods within Trader that contain the implementation of the mathematical models described above (e.g., _volcanic_voucher_strategy, _macarons_strategy).

🛠️ Dependencies
The project requires the following standard Python libraries for numerical computing:

Bash

pip install numpy pandas jsonpickle
