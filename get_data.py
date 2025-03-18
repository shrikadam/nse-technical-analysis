import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from tvDatafeed import TvDatafeed, Interval

tv = TvDatafeed()

nifty50_df = pd.read_csv("data/nifty50list.csv")    
# Create an empty dictionary to store the industry-wise stocks
industry_dict = {}

# Iterate over each row in the DataFrame
for index, row in nifty50_df.iterrows():
    industry = row['Industry']
    symbol = row['Symbol']
    
    # If the industry already exists in the dictionary, append the symbol
    if industry in industry_dict:
        industry_dict[industry].append(symbol)
    else:
        # If the industry is not in the dictionary, create a new entry with the symbol in a list
        industry_dict[industry] = [symbol]

# Print the dictionary contents line by line
# for industry, symbols in industry_dict.items():
#     print(f"Industry: {industry}")
#     print(f"Stocks: {', '.join(symbols)}\n")

nifty50_stocks = []
n_bars = 500
for industry, symbols in industry_dict.items():
    plt.figure(figsize=(12, 6))  # Create a new figure for each industry
    for symbol in symbols:
        # Fetch historical data for each stock
        try:
            df = tv.get_hist(symbol=symbol, exchange='NSE', interval=Interval.in_daily, n_bars=n_bars)
            df_ = df[['close']].rename(columns={'close': symbol})
            nifty50_stocks.append(df_)
            # Normalize
            df['close'] /= df['close'].iloc[0]
            # Plot the close prices for each stock
            plt.plot(df.index, df['close'], label=symbol)
        except Exception as e:
            print(f"Error fetching data for {symbol}: {e}")
    
    # Add legend and labels
    plt.title(f"{industry} Stocks - Historical Data")
    plt.xlabel("Date")
    plt.ylabel("Close Price")
    plt.legend(title="Stocks", loc="upper left")
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b-%Y'))
    plt.grid()

    # Save the figure with the industry name
    plt.savefig(f"plots/industrywise_normed/{industry}.png")
    plt.close()  # Close the plot to avoid overlap in next iteration

nifty50_stocks_df = pd.concat(nifty50_stocks, axis=1)
nifty50_stocks_df.index = nifty50_stocks_df.index.date
nifty50_stocks_df.index.name = 'Date'  # Rename index to 'Date'
nifty50_stocks_df.to_csv('data/nifty50_histdata.csv', index=True)
