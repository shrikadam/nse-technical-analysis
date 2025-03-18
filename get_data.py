import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from tvDatafeed import TvDatafeed, Interval

tv = TvDatafeed()

nifty50_stocks_df = pd.read_csv("data/ind_nifty50list.csv")    
# Create an empty dictionary to store the industry-wise stocks
industry_dict = {}

# Iterate over each row in the DataFrame
for index, row in nifty50_stocks_df.iterrows():
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

# Function to fetch and plot data for each industry
def fetch_and_plot_data(industry_dict):
    for industry, symbols in industry_dict.items():
        plt.figure(figsize=(12, 6))  # Create a new figure for each industry
        for symbol in symbols:
            # Fetch historical data for each stock
            try:
                df = tv.get_hist(symbol=symbol, exchange='NSE', interval=Interval.in_daily, n_bars=500)
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
        plt.savefig(f"plots/{industry}_historical_data.png")
        plt.close()  # Close the plot to avoid overlap in next iteration

fetch_and_plot_data(industry_dict)

# stock_data = tv.get_hist(symbol='BAJAJ_AUTO',exchange='NSE',interval=Interval.in_daily,n_bars=500)
# print(stock_data.columns)
# stock_data["close"].plot()
# plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b-%Y'))
# plt.grid()
# plt.show()