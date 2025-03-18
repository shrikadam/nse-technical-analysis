import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from tvDatafeed import TvDatafeed, Interval

tv = TvDatafeed()

def plot_strategy(df):
    fig, ax = plt.subplots(
        2, 1, sharex=True, figsize=(10, 8), gridspec_kw={"height_ratios": [6, 3]}
    )
    plt.title("Sensex (Above) vs JPM 12-26-9 MACD (Below)")
    ax[0].plot(df["Close"], color="teal", label="Index")
    # ax[0].plot(df["EMA_Long"], color="y", linewidth=0.8, label="Long EMA (50)")
    # ax[0].plot(df["EMA_Short"], color="purple", linewidth=0.8, label="Short EMA (20)")
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

def prepare_strategy(df):
    short_window = 12
    long_window = 26
    signal_window = 9
    long_trend_window = 50
    short_trend_window = 20

    df["EMA_Long"] = df["Close"].ewm(span=long_trend_window, adjust=False).mean()
    df["EMA_Short"] = df["Close"].ewm(span=short_trend_window, adjust=False).mean()
    df["MACD"] = df["Close"].ewm(span=short_window, adjust=False).mean() - df["Close"].ewm(span=long_window, adjust=False).mean()
    df["Signal"] = df["MACD"].ewm(span=signal_window, adjust=False).mean()

    # Find crossover points
    df["Bullish_Crossover"] = (df["MACD"] > df["Signal"]) & (df["MACD"].shift(1) < df["Signal"].shift(1))# & (df["MACD"] < 0)  # Bullish
    df["Bearish_Crossover"] = (df["MACD"] < df["Signal"]) & (df["MACD"].shift(1) > df["Signal"].shift(1))# & (df["MACD"] > 0)  # Bearish

    return df

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

def plot_index_performance(categories, values):
    colors = ["green" if v > 0 else "red" for v in values]

    # Create figure
    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot bars
    ax.bar(categories, values, color=colors)
    x = np.arange(len(categories))
    ax.set_xticks(x)  
    ax.set_xticklabels(categories, rotation=90)
    ax.grid()

    plt.tight_layout()

    return fig

# sensex_df = pd.read_csv("data/BSE_Bankex_01011999_31122024.csv", index_col="Date", parse_dates=True)    
# sensex_df = sensex_df.loc["2024-07-01":"2024-12-31"]
# sensex_df = prepare_strategy(sensex_df)
# total_invested, total_earned, profit_loss = backtest_strategy(sensex_df)

# folder_path = 'data'
# # Dictionary to store plotted lines and their corresponding data
# categories = []
# values = []
# # Loop through all CSV files in the folder
# for filename in os.listdir(folder_path):
#     if filename.endswith('.csv'):
#         # Extract the name between the first and second underscores
#         legend_name = filename.split('_')[1]
#         categories.append(legend_name)
#         # Load the CSV file into a pandas DataFrame
#         file_path = os.path.join(folder_path, filename)
#         df = pd.read_csv(file_path, index_col="Date", parse_dates=True) 
#         sd = "2024-01-01"
#         ed = "2024-12-31"
#         df = df.loc[sd:ed]  
#         macd_df = prepare_strategy(df)
#         fig = plot_strategy(macd_df)
#         fig.savefig(f"plots/{legend_name}.png")
#         print(f"Index name: {legend_name}")
#         invested, cash, pl = backtest_strategy(macd_df)
#         values.append(pl)

# fig = plot_index_performance(categories, values)
# fig.savefig(f"plots/performance.png")
# total_pl = sum(values)
# print(f"Total Profit/Loss: ₹{total_pl:.2f} ({'Profit' if total_pl > 0 else 'Loss'})")

mazdock_df = tv.get_hist(symbol='MAZDOCK',exchange='NSE',interval=Interval.in_daily,n_bars=1000)
mazdock_df.rename(columns={'close': 'Close'}, inplace=True)
sd = "2024-01-01"
ed = "2024-12-31"
mazdock_df = mazdock_df.loc[sd:ed] 
mazdock_df = prepare_strategy(mazdock_df)
fig = plot_strategy(mazdock_df)
fig.savefig(f"plots/MAZDOCK.png")
total_invested, total_earned, profit_loss = backtest_strategy(mazdock_df)
print(f"Total Profit/Loss: ₹{profit_loss:.2f} ({'Profit' if profit_loss > 0 else 'Loss'})")
