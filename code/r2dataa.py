import pandas as pd
import numpy as np


# Load data
df1=pd.read_csv('/Users/ankhanhnguyen/Downloads/round-2-island-data-bottle/prices_round_2_day_-1.csv',delimiter=";")
df2=pd.read_csv('/Users/ankhanhnguyen/Downloads/round-2-island-data-bottle/prices_round_2_day_-1.csv',delimiter=";")
df0=pd.read_csv('/Users/ankhanhnguyen/Downloads/round-2-island-data-bottle/prices_round_2_day_-1.csv',delimiter=";")
df1['timestamp']+=999900
df2['timestamp']+=2*999900
df=pd.concat([df0, df1, df2], ignore_index=True)
df.set_index(['day', 'timestamp', 'product'], inplace=True)

# First ensure we don't have duplicates - keep last entry if duplicates exist
df = df[~df.index.duplicated(keep='last')]

# Now unstack will work
unstacked_df = df.unstack(level='product')

# Define basket compositions
basket1_components = {
    'CROISSANTS': 6,
    'JAMS': 3,
    'DJEMBES': 1
}

basket2_components = {
    'CROISSANTS': 4,
    'JAMS': 2
}

# Calculate composite values for each basket
for basket, components in [('PICNIC_BASKET1', basket1_components),
                         ('PICNIC_BASKET2', basket2_components)]:
    
    # Composite ASK price (cost to buy components)
    unstacked_df[('composite_ask', basket)] = sum(
        qty * unstacked_df[('ask_price_1', product)]
        for product, qty in components.items()
    )
    
    # Composite BID price (value from selling components)
    unstacked_df[('composite_bid', basket)] = sum(
        qty * unstacked_df[('bid_price_1', product)]
        for product, qty in components.items()
    )

# Identify arbitrage opportunities
baskets = ['PICNIC_BASKET1', 'PICNIC_BASKET2']

for basket in baskets:
    # Buy signal: Basket ask < Composite bid (buy cheap basket, sell components)
    unstacked_df[('signal_buy', basket)] = (
        unstacked_df[('ask_price_1', basket)] < 
        unstacked_df[('composite_bid', basket)]
    )
    
    # Sell signal: Basket bid > Composite ask (sell expensive basket, buy components)
    unstacked_df[('signal_sell', basket)] = (
        unstacked_df[('bid_price_1', basket)] > 
        unstacked_df[('composite_ask', basket)]
    )
    
    # Profit potential
    unstacked_df[('profit_buy', basket)] = (
        unstacked_df[('composite_bid', basket)] -
        unstacked_df[('ask_price_1', basket)]
    )
    
    unstacked_df[('profit_sell', basket)] = (
        unstacked_df[('bid_price_1', basket)] -
        unstacked_df[('composite_ask', basket)]
    )

# Generate trade signals
trade_signals = []

for idx, row in unstacked_df.iterrows():
    day, timestamp = idx
    for basket in baskets:
        if row[('signal_buy', basket)]:
            trade_signals.append({
                'day': day,
                'timestamp': timestamp,
                'action': 'BUY_BASKET_SELL_COMPONENTS',
                'basket': basket,
                'basket_price': row[('ask_price_1', basket)],
                'composite_value': row[('composite_bid', basket)],
                'profit_potential': row[('profit_buy', basket)]
            })
            
        if row[('signal_sell', basket)]:
            trade_signals.append({
                'day': day,
                'timestamp': timestamp,
                'action': 'SELL_BASKET_BUY_COMPONENTS',
                'basket': basket,
                'basket_price': row[('bid_price_1', basket)],
                'composite_value': row[('composite_ask', basket)],
                'profit_potential': row[('profit_sell', basket)]
            })

trades_df = pd.DataFrame(trade_signals)

# Split into separate basket DataFrames and create multi-index
basket1_trades = (
    trades_df[trades_df['basket'] == 'PICNIC_BASKET1']
    .set_index(['day', 'timestamp'])
    .sort_index()
    .drop(columns=['basket'])
)

basket2_trades = (
    trades_df[trades_df['basket'] == 'PICNIC_BASKET2']
    .set_index(['day', 'timestamp'])
    .sort_index()
    .drop(columns=['basket'])
)

# Organize columns in logical order
column_order = [
    'action', 
    'basket_price', 
    'composite_value', 
    'profit_potential'
]

basket1_trades = basket1_trades[column_order]
basket2_trades = basket2_trades[column_order]

# Add metadata to distinguish
basket1_trades.columns = pd.MultiIndex.from_product([['PICNIC_BASKET1'], basket1_trades.columns])
basket2_trades.columns = pd.MultiIndex.from_product([['PICNIC_BASKET2'], basket2_trades.columns])

# Final organized structure
print("Basket 1 Trades:")
display(basket1_trades.head())

print("\nBasket 2 Trades:")
display(basket2_trades.head())