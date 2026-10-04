import pandas as pd
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
import matplotlib.pyplot as plt
import akshare as ak
import time

# 1. Load data
def fetch_stock_data(symbol, start_date, end_date, adjust='qfq'):
    df = ak.stock_zh_a_daily(
    symbol = symbol,
    start_date = start_date,
    end_date = end_date,
    adjust = adjust
    )
    df = df.rename(columns={'收盘': 'close_data'})
    return df

# 2. Calculate moving averages
def add_ma_columns(df, short_window=5, long_window=20):
    df = df.copy()
    df[f'ma{short_window}'] = df['close'].rolling(window= short_window).mean()   # 5-day moving average (fast line)
    df[f'ma{long_window}'] = df['close'].rolling(window= long_window).mean()
    df = df.dropna().reset_index(drop=True)  # 20-day moving average (slow line)
    return df

# 3. Backtest the moving average crossover strategy
def backtest_ma_strategy(df, short_window= 5, long_window= 20, take_profit_rate= 0.08, stop_loss_rate= -0.03):
    ma_fast_col = f'ma{short_window}'
    ma_slow_col = f'ma{long_window}'

    position = 0          # 0 = no position (empty), 1 = holding stock
    trade_result = []     # Store return rate of each completed trade
    entry_price = 0       # Price at which we bought


    # Loop through each day (start from index 1 to avoid i-1 out of bounds)
    for i in range(1, len(df)):
        ma5 = df[ma_fast_col].iloc[i]
        ma20 = df[ma_slow_col].iloc[i]
        close = df['close'].iloc[i]
        prev_ma_fast = df[ma_fast_col].iloc[i-1]
        prev_ma_slow = df[ma_slow_col].iloc[i-1]

        if position == 0:
            # --- Golden cross: fast MA crosses above slow MA → buy ---
            if ma5 > ma20 and prev_ma_fast <= prev_ma_slow:
                position = 1
                entry_price = close

        else:
            # --- Condition 1: Death cross (fast MA crosses below slow MA) ---
            cond_death_cross = (ma5 < ma20) and (prev_ma_fast >= prev_ma_slow)

            # --- Condition 2: Current position return rate ---
            current_return = (close - entry_price) / entry_price

            # --- Condition 3: Hit take profit or stop loss ---
            cond_take_profit = current_return >= take_profit_rate
            cond_stop_loss = current_return <= stop_loss_rate

            # Sell if ANY of the above conditions is true
            if cond_death_cross or cond_take_profit or cond_stop_loss:
                position = 0
                trade_result.append(current_return)
    return trade_result

# 4. Print backtest results
def print_backtest_result(trade_result):
    print("=" * 40)
    print("BACKTEST RESULT")
    print("=" * 40)
    print(f"Total trades: {len(trade_result)}")

    if len(trade_result) > 0:
        total_return = sum(trade_result)
        win_trades = sum(1 for r in trade_result if r > 0)
        win_rate = win_trades / len(trade_result)

        print(f"Each trade return: {[f'{r:.2%}' for r in trade_result]}")
        print(f"Total return rate: {total_return:.2%}")
        print(f"Win rate: {win_rate:.2%}")
        print(f"Max single profit: {max(trade_result):.2%}")
        print(f"Max single loss: {min(trade_result):.2%}")
        print("=" * 40)
    else:
        print("No complete buy-sell trade in this period!")

# 5. Plot candlestick / price chart with MAs
def plot_strategy_chart(df, short_window=5, long_window=20):
    ma_fast_col = f'ma{short_window}'
    ma_slow_col = f'ma{long_window}'

    plt.figure(figsize=(12, 6))
    plt.plot(df['close'], label='Close Price', color='black', linewidth=1)
    plt.plot(df[ma_fast_col], label=f'MA{short_window}', color='blue', linewidth=1.2)
    plt.plot(df[ma_slow_col], label=f'MA{long_window}', color='orange', linewidth=1.2)

    plt.title('MA Crossover Strategy Backtest')
    plt.xlabel('Days')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

def main():
    """Main entry point: run the full backtest pipeline."""
    # --- Parameters ---
    SYMBOL = 'sh600519'       # Kweichow Moutai
    START_DATE = '20230101'
    END_DATE = '20231231'
    SHORT_WINDOW = 2          # fast MA period
    LONG_WINDOW = 5           # slow MA period
    TAKE_PROFIT = 0.08        # +8% take profit
    STOP_LOSS = -0.03         # -3% stop loss

    # Step 1: Fetch data
    df = fetch_stock_data(SYMBOL, START_DATE, END_DATE)
    print("Columns:", df.columns.tolist())

    # Step 2: Calculate moving averages
    df = add_ma_columns(df, short_window=SHORT_WINDOW, long_window=LONG_WINDOW)

    # Step 3: Backtest the strategy
    trade_result = backtest_ma_strategy(
        df,
        short_window=SHORT_WINDOW,
        long_window=LONG_WINDOW,
        take_profit_rate=TAKE_PROFIT,
        stop_loss_rate=STOP_LOSS
    )

    # Step 4: Print results
    print_backtest_result(trade_result)

    # Step 5: Plot chart
    plot_strategy_chart(df, short_window=SHORT_WINDOW, long_window=LONG_WINDOW)


if __name__ == '__main__':
    main()