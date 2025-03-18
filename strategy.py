import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def prepare_strategy(df):
    short_window = 12
    long_window = 26
    signal_window = 9

    df["MACD"] = df["Close"].ewm(span=short_window, adjust=False).mean() - df["Close"].ewm(span=long_window, adjust=False).mean()
    df["Signal"] = df["MACD"].ewm(span=signal_window, adjust=False).mean()

    # Find crossover points
    df["Bullish_Crossover"] = (df["MACD"] > df["Signal"]) & (df["MACD"].shift(1) < df["Signal"].shift(1)) & (df["MACD"] < 0)  # Bullish
    df["Bearish_Crossover"] = (df["MACD"] < df["Signal"]) & (df["MACD"].shift(1) > df["Signal"].shift(1)) & (df["MACD"] > 0)  # Bearish

    return df

def plot_strategy(df):
    fig, ax = plt.subplots(
        2, 1, sharex=True, figsize=(10, 8), gridspec_kw={"height_ratios": [6, 3]}
    )
    plt.title("Sensex (Above) vs JPM 12-26-9 MACD (Below)")
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
    print(f"Total Invested: ₹{total_invested:.2f}")
    print(f"Total Earned: ₹{cash:.2f}")
    print(f"Profit/Loss: ₹{profit_or_loss:.2f} ({'Profit' if profit_or_loss > 0 else 'Loss'})")
    print("---")

    return total_invested, cash, profit_or_loss  # Return key metrics

if __name__ == "__main__":  
    symbol = "SHRIRAMFIN"
    df = pd.read_csv("data/nifty50_histdata.csv", index_col="Date", parse_dates=True)[symbol]    
    df = df.to_frame(name='Close')
    df = prepare_strategy(df)
    sd = "2024-01-01"
    ed = "2024-12-31"
    df = df.loc[sd:ed]
    fig = plot_strategy(df)
    fig.savefig(f"plots/{symbol}.png")
    total_invested, total_earned, profit_loss = backtest_strategy(df)
