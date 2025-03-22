import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.widgets import CheckButtons
import random

def ma(df, column='Close', short_window=10, medium_window=50, long_window=100):
    ema_short = df[column].ewm(span=short_window, adjust=False).mean()
    sma_medium = df[column].rolling(window=medium_window).mean()
    sma_long = df[column].rolling(window=long_window).mean()
    return ema_short, sma_medium, sma_long

def bollinger_bands(df, column="Close", window=20, n_std=2):
    sma = df[column].rolling(window=window).mean()
    std = df[column].rolling(window=window).std()
    upper_band = sma + (n_std * std)
    lower_band = sma - (n_std * std)
    bb_vec = (df[column] - sma) / (n_std * std)
    return sma, upper_band, lower_band, bb_vec

def macd(df, column='Close', short_window=12, long_window=26, signal_window=9):
    short_ema = df[column].ewm(span=short_window, adjust=False).mean()
    long_ema = df[column].ewm(span=long_window, adjust=False).mean()
    macd = short_ema - long_ema
    signal = macd.ewm(span=signal_window, adjust=False).mean()   
    return macd, signal

def rsi(df, column="Close", window=14):
    delta = df[column].diff()
    delta = delta[1:]
    up, down = delta.clip(lower=0), delta.clip(upper=0).abs()
    roll_up, roll_down = up.ewm(alpha=1 / window).mean(), down.ewm(alpha=1 / window).mean()
    rs = roll_up / roll_down
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi

def bsl(df, column="Close"):
    # Bull Strength Level
    ema_short, sma_medium, sma_long = ma(df)
    bsl = 5 * ((ema_short - sma_medium) + (sma_medium - sma_long)) / df[column]
    return bsl

def prepare_indicators(df):
    df = df.copy()
    df["MACD"], df["Signal"] = macd(df)
    df['MACD_Slope'] = np.degrees(np.arctan(df['MACD'].diff()))
    df['Signal_Slope'] = np.degrees(np.arctan(df['Signal'].diff()))
    df["SMA_BB"], df["Upper_BB"], df["Lower_BB"], df["BB_Vec"] = bollinger_bands(df)
    df['RSI'] = rsi(df) 
    df['EMA_Short'], df['SMA_Medium'], df['SMA_Long'] = ma(df)
    df['BSL'] = bsl(df)
    df["BUY"] = False
    df["SELL"] = False
    df['EMA_SMA_Cross'] = ((df['EMA_Short'] > df['SMA_Medium']) & 
                           (df['EMA_Short'].shift(1) <= df['SMA_Medium'].shift(1)))
    return df

def plot_strategy(df):
    fig, ax = plt.subplots(
        5, 1, sharex=True, figsize=(8, 12), gridspec_kw={"height_ratios": [8, 3, 3, 3, 3]}
    )

    ax[0].plot(df["Upper_BB"], color="maroon", linewidth=0.8, linestyle="--", alpha=0.4, label="Upper BB")
    ax[0].plot(df["Lower_BB"], color="maroon", linewidth=0.8, linestyle="--", alpha=0.4, label="Lower BB")
    ax[0].plot(df["SMA_BB"], color="maroon", linewidth=1, alpha=0.4, label="SMA BB")
    ax[0].plot(df["Close"], color="teal", label="Index")
    ax[0].set_ylabel("Stock Price")
    ax[0].set_xlabel("Date")
    ax[0].grid()
    ax[0].legend()

    ax[1].plot(df["Signal"], color="orange", linewidth=0.8, label="Signal")
    ax[1].plot(df["MACD"], color="b", linewidth=0.8, label="MACD")
    ax[1].axhline(0, color="black")
    ax[1].set_ylabel("MACD, Signal")
    ax[1].set_xlabel("Date")
    ax[1].grid()
    ax[1].legend()

    ax[2].plot(df["BB_Vec"], color="maroon", linewidth=1, label="BB Vec")
    ax[2].plot(df["BSL"], color="gray", linewidth=1, label="BSL")
    ax[2].axhline(-1, linestyle="--", color="g")
    ax[2].axhline(1, linestyle="--", color="r")
    ax[2].set_ylabel("BB Vec")
    ax[2].set_xlabel("Date")
    ax[2].grid()
    ax[2].legend()

    ax[3].plot(df["RSI"], color="purple", linewidth=1, label="RSI")
    ax[3].axhline(70, linestyle="--", color="r")
    ax[3].axhline(30, linestyle="--", color="g")
    ax[3].set_ylabel("RSI")
    ax[3].set_xlabel("Date")
    ax[3].grid()
    ax[3].legend()

    ax[4].plot(df["EMA_Short"], color="deeppink", linewidth=1, label="Short EMA")
    ax[4].plot(df["SMA_Medium"], color="cyan", linewidth=1, label="Medium SMA")
    ax[4].plot(df["SMA_Long"], color="gold", linewidth=1, label="Long SMA")
    ax[4].set_ylabel("MA")
    ax[4].set_xlabel("Date")
    ax[4].grid()
    ax[4].legend()

    # Get dates where crossover happens
    bullish_dates = df.index[df["BUY"]]
    bearish_dates = df.index[df["SELL"]]

    for date in bullish_dates:
        ax[0].axvline(x=date, color="g", linestyle="--", linewidth=0.8, alpha=0.7)
        ax[1].axvline(x=date, color="g", linestyle="--", linewidth=0.8, alpha=0.7)
        ax[2].axvline(x=date, color="g", linestyle="--", linewidth=0.8, alpha=0.7)
        ax[3].axvline(x=date, color="g", linestyle="--", linewidth=0.8, alpha=0.7)
        ax[4].axvline(x=date, color="g", linestyle="--", linewidth=0.8, alpha=0.7)

    for date in bearish_dates:
        ax[0].axvline(x=date, color="r", linestyle="--", linewidth=0.8, alpha=0.7)
        ax[1].axvline(x=date, color="r", linestyle="--", linewidth=0.8, alpha=0.7)
        ax[2].axvline(x=date, color="r", linestyle="--", linewidth=0.8, alpha=0.7)
        ax[3].axvline(x=date, color="r", linestyle="--", linewidth=0.8, alpha=0.7)
        ax[4].axvline(x=date, color="r", linestyle="--", linewidth=0.8, alpha=0.7)

    plt.gcf().autofmt_xdate()
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b-%Y'))
    plt.tight_layout()
    plt.legend(loc="lower left")

    return fig

def apply_classic_macd_strategy(df):
    df = df.copy()
    # Find crossover points
    df["BUY"] = ((df["MACD"] > df["Signal"]) & 
                               (df["MACD"].shift(1) < df["Signal"].shift(1)) & 
                               (df["MACD"] < 0))  # Bullish
    df["SELL"] = ((df["MACD"] < df["Signal"]) & 
                               (df["MACD"].shift(1) > df["Signal"].shift(1)) & 
                               (df["MACD"] > 0))  # Bearish
    return df

def apply_advanced_macd_strategy(df):
    df = df.copy()
    # Identify bullish crossovers (MACD crosses above Signal with slope > 45°)
    df['BUY'] = ((df['MACD'] > df['Signal']) & 
                               (df['MACD'].shift(1) < df['Signal'].shift(1)) &
                               ((df["MACD"] < 0) | (df['MACD_Slope'] > 45)))
    # Identify bearish crossovers (MACD crosses below Signal)
    df['SELL'] = ((df['MACD'] < df['Signal']) & 
                               (df['MACD'].shift(1) > df['Signal'].shift(1)) &
                               (df["MACD"] > 0))

    return df

def appply_classic_bb_strategy(df):
    df = df.copy()
    llim = -1
    ulim = 1
    df['BUY'] = (df['BB_Vec'].shift(1) < llim) & (df['BB_Vec'] >= llim)
    df['SELL'] = (df['BB_Vec'].shift(1) > ulim) & (df['BB_Vec'] <= ulim)
    return df

def apply_my_strategy(df):
    df = df.copy()
    llim = -1
    ulim = 0.8
    df['BUY'] = ((df['BB_Vec'].shift(1) < llim) & 
                 (df['BB_Vec'] >= llim) &
                 (df['EMA_Short'] < df['SMA_Long']))
    # df['SELL'] = (df['BB_Vec'].shift(1) > ulim) & (df['BB_Vec'] <= ulim)
    df['SELL'] = ((df['MACD'] < df['Signal']) & 
                               (df['MACD'].shift(1) > df['Signal'].shift(1)) &
                               (df["MACD"] > 0))
    # # Create potential signals based on BB_Vec crossovers
    # potential_buy = (df['BB_Vec'].shift(1) < llim) & (df['BB_Vec'] >= llim)
    # potential_sell = (df['BB_Vec'].shift(1) > ulim) & (df['BB_Vec'] <= ulim)
    
    # # Initialize actual BUY/SELL columns with False
    # df['BUY'] = False
    # df['SELL'] = False
    
    # # Track our position state
    # in_position = False
    
    # # Iterate through the dataframe
    # for i in range(len(df)):
    #     if not in_position and potential_buy.iloc[i]:
    #         # If we're not in a position and we have a potential buy signal
    #         df.loc[df.index[i], 'BUY'] = True
    #         in_position = True
    #     elif in_position and potential_sell.iloc[i]:
    #         # If we're in a position and we have a potential sell signal
    #         df.loc[df.index[i], 'SELL'] = True
    #         in_position = False
    return df

def backtest_strategy(df, initial_investment=1000):
    total_invested = 0  # Track total money invested
    cash = 0            # Total cash after selling
    invested = 0        # Current shares holding
    last_buy_price = None  # Store the last buy price
    
    for i in range(len(df)):
        if df["BUY"].iloc[i]:  # Buy Signal
            last_buy_price = df["Close"].iloc[i]
            invested += initial_investment / last_buy_price  # Convert money to shares
            total_invested += initial_investment  # Track total invested amount
            # print(f"BUY at {df.index[i].date()} | Price: {last_buy_price:.2f} | Shares: {invested:.4f} | Total Invested: ₹{total_invested}")

        elif df["SELL"].iloc[i] and last_buy_price is not None:  # Sell Signal
            sell_price = df["Close"].iloc[i]
            cash += invested * sell_price  # Convert shares to cash
            # print(f"SELL at {df.index[i].date()} | Price: {sell_price:.2f} | Cash: ₹{cash:.2f}")
            invested = 0  # Reset investment
            last_buy_price = None  # Reset last buy price

    # If still holding shares at the end, sell at the last price
    if invested > 0:
        final_price = df["Close"].iloc[-1]
        cash += invested * final_price
        # print(f"FINAL SELL at {df.index[-1].date()} | Price: {final_price:.2f} | Total Cash: ₹{cash:.2f}")
    
    # Calculate profit/loss
    profit_or_loss = cash - total_invested
    # print(f"Total Invested: ₹{total_invested:.2f}")
    # print(f"Total Earned: ₹{cash:.2f}")
    # print(f"Profit/Loss: ₹{profit_or_loss:.2f} ({'Profit' if profit_or_loss > 0 else 'Loss'})")
    # print("---")

    return total_invested, cash, profit_or_loss  # Return key metrics

if __name__ == "__main__":  
    df = pd.read_csv("data/nifty50_histdata.csv", index_col="Date", parse_dates=True)
    ############ Single-stock Testing ############
    # symbol = "TATASTEEL"
    symbol = random.choice(df.columns)
    df = df[symbol]
    print(symbol)
    df = df.to_frame(name='Close')
    df = prepare_indicators(df)
    df_macd = apply_classic_macd_strategy(df)
    df_bb = appply_classic_bb_strategy(df)
    df_shri = apply_my_strategy(df)
    sd = "2024-03-20"
    ed = "2025-03-20"
    df_macd = df_macd.loc[sd:ed]
    df_bb = df_bb.loc[sd:ed]
    df_shri = df_shri.loc[sd:ed]
    # fig_macd = plot_strategy(df_macd)
    # fig_bb = plot_strategy(df_bb)
    fig_shri = plot_strategy(df_shri)
    total_invested_macd, total_earned_macd, profit_loss_macd = backtest_strategy(df_macd)
    total_invested_bb, total_earned_bb, profit_loss_bb = backtest_strategy(df_bb)
    total_invested_shri, total_earned_shri, profit_loss_shri = backtest_strategy(df_shri)
    print("MACD Strategy:") 
    print(f"Total Invested: ₹{total_invested_macd:.2f}")
    if(total_earned_macd):
        print(f"Profit/Loss: ₹{profit_loss_macd:.2f} ({abs(profit_loss_macd/total_invested_macd)*100:.2f}{'% Profit' if profit_loss_macd > 0 else '% Loss'})")
    print("---")
    print("BB Strategy:") 
    print(f"Total Invested: ₹{total_invested_bb:.2f}")
    if(total_earned_bb):
        print(f"Profit/Loss: ₹{profit_loss_bb:.2f} ({abs(profit_loss_bb/total_invested_bb)*100:.2f}{'% Profit' if profit_loss_bb > 0 else '% Loss'})")
    print("---")
    print("My Strategy:") 
    print(f"Total Invested: ₹{total_invested_shri:.2f}")
    if(total_earned_shri):
        print(f"Profit/Loss: ₹{profit_loss_shri:.2f} ({abs(profit_loss_shri/total_invested_shri)*100:.2f}{'% Profit' if profit_loss_shri > 0 else '% Loss'})")
    # fig_cl.savefig(f"plots/{symbol}_classic.png")
    # fig_adv.savefig(f"plots/{symbol}_advanced.png")
    # fig = stack_figures_side_by_side(fig_cl, fig_adv)
    plt.show()
    
    ############ Nifty50 Testing ############
    # grand_pl_macd = []
    # grand_pl_bb = []
    # grand_pl_shri = []
    # for i in range(20):
    #     df_result = pd.DataFrame(columns=['Symbol', 'PL_MACD', 'PL_BB', 'PL_Shri'])
    #     random_syms = random.sample(list(df.columns), 10)
    #     # for i, symbol in enumerate(df.columns):
    #     for i, symbol in enumerate(random_syms):
    #         df_symbol = df[symbol]
    #         df_symbol = df_symbol.to_frame(name='Close')
    #         df_symbol = prepare_indicators(df_symbol)
    #         df_macd = apply_classic_macd_strategy(df_symbol)
    #         df_bb = appply_classic_bb_strategy(df_symbol)
    #         df_shri = apply_my_strategy(df_symbol)
    #         sd = "2024-09-01"
    #         ed = "2025-03-20"
    #         df_macd = df_macd.loc[sd:ed]
    #         df_bb = df_bb.loc[sd:ed]
    #         df_shri = df_shri.loc[sd:ed]
    #         # fig_macd = plot_strategy(df_macd)
    #         # fig_bb = plot_strategy(df_bb)
    #         total_invested_macd, total_earned_macd, profit_loss_macd = backtest_strategy(df_macd)
    #         total_invested_bb, total_earned_bb, profit_loss_bb = backtest_strategy(df_bb)
    #         total_invested_shri, total_earned_shri, profit_loss_shri = backtest_strategy(df_shri)
    #         df_result.loc[i] = [symbol, profit_loss_macd, profit_loss_bb, profit_loss_shri]
    #         # print(symbol, profit_loss_cl, profit_loss_adv)

    #     # x = np.arange(len(df_result['Symbol']))
    #     # width = 0.35

    #     # plt.figure(figsize=(14, 7))
    #     # plt.bar(x - width/2, df_result['PL_Classic'], width, label='Classic', color='skyblue')
    #     # plt.bar(x + width/2, df_result['PL_Advanced'], width, label='Advanced', color='orange')

    #     # plt.xlabel('Symbols')
    #     # plt.ylabel('Profit/Loss')
    #     # plt.title('Strategy Comparison for 50 Symbols')
    #     # plt.xticks(x, df_result['Symbol'], rotation=90)
    #     # plt.legend()
    #     # plt.grid(True)
    #     # plt.tight_layout()
    #     # plt.show()

    #     pl_macd = df_result["PL_MACD"].sum()
    #     pl_bb = df_result["PL_BB"].sum()
    #     pl_shri = df_result["PL_Shri"].sum()
    #     # print("MACD PL: ", pl_macd)
    #     # print("BB PL: ", pl_bb)
    #     # print("Shri PL: ", pl_shri)
    #     # print("---")
    #     grand_pl_macd.append(pl_macd)
    #     grand_pl_bb.append(pl_bb)
    #     grand_pl_shri.append(pl_shri)

    # print("Grand MACD PL: ", int(sum(grand_pl_macd)))
    # print("Grand BB PL: ", int(sum(grand_pl_bb)))
    # print("Grand Shri PL: ", int(sum(grand_pl_shri)))

    # # plt.plot(grand_pl_macd, label="MACD")
    # # plt.plot(grand_pl_bb, label="BB")
    # # plt.plot(grand_pl_shri, label="Shri")
    # # plt.legend()
    # # plt.show()
    plt.close()
