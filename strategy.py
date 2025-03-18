import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random

def prepare_classic_macd_strategy(df):
    df = df.copy()
    short_window = 12
    long_window = 26
    signal_window = 9

    df["MACD"] = df["Close"].ewm(span=short_window, adjust=False).mean() - df["Close"].ewm(span=long_window, adjust=False).mean()
    df["Signal"] = df["MACD"].ewm(span=signal_window, adjust=False).mean()

    df['MACD_Slope'] = np.degrees(np.arctan(df['MACD'].diff()))
    df['Signal_Slope'] = np.degrees(np.arctan(df['Signal'].diff()))
    
    # Find crossover points
    df["Bullish_Crossover"] = (df["MACD"] > df["Signal"]) & (df["MACD"].shift(1) < df["Signal"].shift(1)) & (df["MACD"] < 0)  # Bullish
    df["Bearish_Crossover"] = (df["MACD"] < df["Signal"]) & (df["MACD"].shift(1) > df["Signal"].shift(1)) & (df["MACD"] > 0)  # Bearish

    return df

def prepare_advanced_macd_strategy(df):
    df = df.copy()
    short_window = 12
    long_window = 26
    signal_window = 9

    df["MACD"] = df["Close"].ewm(span=short_window, adjust=False).mean() - df["Close"].ewm(span=long_window, adjust=False).mean()
    df["Signal"] = df["MACD"].ewm(span=signal_window, adjust=False).mean()

    df['MACD_Slope'] = np.degrees(np.arctan(df['MACD'].diff()))
    df['Signal_Slope'] = np.degrees(np.arctan(df['Signal'].diff()))
    
    # Identify bullish crossovers (MACD crosses above Signal with slope > 45°)
    df['Bullish_Crossover'] = ((df['MACD'] > df['Signal']) & 
                               (df['MACD'].shift(1) < df['Signal'].shift(1)) &
                               ((df["MACD"] < 0) | (df['MACD_Slope'] > 45)))

    # Identify bearish crossovers (MACD crosses below Signal)
    df['Bearish_Crossover'] = ((df['MACD'] < df['Signal']) & 
                               (df['MACD'].shift(1) > df['Signal'].shift(1)) &
                               (df["MACD"] > 0))

    return df

def plot_strategy(df):
    fig, ax = plt.subplots(
        2, 1, sharex=True, figsize=(10, 8), gridspec_kw={"height_ratios": [6, 3]}
    )
    plt.title("Stock Price (Above) vs 12-26-9 MACD (Below)")
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

    # Get dates where crossover happens
    bullish_dates = df.index[df["Bullish_Crossover"]]
    bearish_dates = df.index[df["Bearish_Crossover"]]

    for date in bullish_dates:
        ax[0].axvline(x=date, color="g", linestyle="--", linewidth=0.8, alpha=0.7)
        ax[1].axvline(x=date, color="g", linestyle="--", linewidth=0.8, alpha=0.7)

    for date in bearish_dates:
        ax[0].axvline(x=date, color="r", linestyle="--", linewidth=0.8, alpha=0.7)
        ax[1].axvline(x=date, color="r", linestyle="--", linewidth=0.8, alpha=0.7)

    plt.gcf().autofmt_xdate()

    return fig

def stack_figures_side_by_side(fig1, fig2):
    # Create a new figure with two subplots side by side
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    # Draw fig1 on the first subplot
    for ax in fig1.get_axes():
        ax.figure = fig
        fig.axes.append(ax)
        fig.add_axes(ax)
        ax.change_geometry(1, 2, 1)

    # Draw fig2 on the second subplot
    for ax in fig2.get_axes():
        ax.figure = fig
        fig.axes.append(ax)
        fig.add_axes(ax)
        ax.change_geometry(1, 2, 2)

    plt.tight_layout()
    return fig

def backtest_strategy(df, initial_investment=1000):
    total_invested = 0  # Track total money invested
    cash = 0            # Total cash after selling
    invested = 0        # Current shares holding
    last_buy_price = None  # Store the last buy price
    
    for i in range(len(df)):
        if df["Bullish_Crossover"].iloc[i]:  # Buy Signal
            last_buy_price = df["Close"].iloc[i]
            invested += initial_investment / last_buy_price  # Convert money to shares
            total_invested += initial_investment  # Track total invested amount
            # print(f"BUY at {df.index[i].date()} | Price: {last_buy_price:.2f} | Shares: {invested:.4f} | Total Invested: ₹{total_invested}")

        elif df["Bearish_Crossover"].iloc[i] and last_buy_price is not None:  # Sell Signal
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
    ############ Single-stock Testing ############
    # symbol = "BAJAJ_AUTO"
    df = pd.read_csv("data/nifty50_histdata.csv", index_col="Date", parse_dates=True)
    symbol = random.choice(df.columns)
    df = df[symbol]
    print(symbol)
    df = df.to_frame(name='Close')
    df_cl = prepare_classic_macd_strategy(df)
    df_adv = prepare_advanced_macd_strategy(df)
    sd = "2024-11-30"
    ed = "2025-02-28"
    df_cl = df_cl.loc[sd:ed]
    df_adv = df_adv.loc[sd:ed]
    fig_cl = plot_strategy(df_cl)
    fig_adv = plot_strategy(df_adv)
    total_invested_cl, total_earned_cl, profit_loss_cl = backtest_strategy(df_cl)
    total_invested_adv, total_earned_adv, profit_loss_adv = backtest_strategy(df_adv)
    print(f"Classic Profit/Loss: ₹{profit_loss_cl:.2f} ({'Profit' if profit_loss_cl > 0 else 'Loss'})")
    print(f"Advanced Profit/Loss: ₹{profit_loss_adv:.2f} ({'Profit' if profit_loss_adv > 0 else 'Loss'})")
    print(f"{'Classic won!!!' if profit_loss_cl >= profit_loss_adv else 'Advanced won!!!'}")
    # fig_cl.savefig(f"plots/{symbol}_classic.png")
    # fig_adv.savefig(f"plots/{symbol}_advanced.png")
    # fig = stack_figures_side_by_side(fig_cl, fig_adv)
    plt.show()
    
    ############ Nifty50 Testing ############
    # df = pd.read_csv("data/nifty50_histdata.csv", index_col="Date", parse_dates=True)
    # for i in range(20):
    #     df_result = pd.DataFrame(columns=['Symbol', 'PL_Classic', 'PL_Advanced'])
    #     random_syms = random.sample(list(df.columns), 10)
    #     # for i, symbol in enumerate(df.columns):
    #     for i, symbol in enumerate(random_syms):
    #         df_symbol = df[symbol]
    #         df_symbol = df_symbol.to_frame(name='Close')
    #         df_classic = prepare_classic_macd_strategy(df_symbol)
    #         df_advanced = prepare_advanced_macd_strategy(df_symbol)
    #         sd = "2025-01-31"
    #         ed = "2025-02-28"
    #         df_classic = df_classic.loc[sd:ed]
    #         df_advanced = df_advanced.loc[sd:ed]
    #         total_invested_cl, total_earned_cl, profit_loss_cl = backtest_strategy(df_classic)
    #         total_invested_adv, total_earned_adv, profit_loss_adv = backtest_strategy(df_advanced)
    #         df_result.loc[i] = [symbol, profit_loss_cl, profit_loss_adv]
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

    #     pl_classic = df_result["PL_Classic"].sum()
    #     pl_advanced = df_result["PL_Advanced"].sum()
    #     print("Classic PL: ", pl_classic)
    #     print("Advanced PL: ", pl_advanced)
    #     print(f"({'Classic won!!!' if pl_classic > pl_advanced else 'Advanced won!!!'})")
    #     print("---")

    plt.close()