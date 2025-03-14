import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Define the folder containing the CSV files
folder_path = 'data'

# Initialize the plot
fig, ax = plt.subplots(figsize=(10, 6))

# Dictionary to store plotted lines and their corresponding data
lines = []
data_dict = {}

# Loop through all CSV files in the folder
for filename in os.listdir(folder_path):
    if filename.endswith('.csv'):
        # Extract the name between the first and second underscores
        legend_name = filename.split('_')[1]
        
        # Load the CSV file into a pandas DataFrame
        file_path = os.path.join(folder_path, filename)
        df = pd.read_csv(file_path)

        # Ensure the 'Date' column is in datetime format
        df['Date'] = pd.to_datetime(df['Date'])

        # Plot the 'Close' column against the 'Date' column
        line, = ax.plot(df['Date'], df['Close'], label=legend_name)
        lines.append(line)
        data_dict[line] = (df['Date'].values, df['Close'].values)

# Add labels and title to the plot
ax.set_xlabel('Date')
ax.set_ylabel('Close')
ax.set_title('Daily Close Prices over Time for Different Indices')
ax.legend()
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.xticks(rotation=45)
plt.tight_layout()
plt.grid()

# Annotation box for hover info
annot = ax.annotate("", xy=(0, 0), xytext=(15, 15),
                    textcoords="offset points",
                    bbox=dict(boxstyle="round", fc="w"),
                    arrowprops=dict(arrowstyle="->"))
annot.set_visible(False)

def update_annot(line, ind):
    xdata, ydata = data_dict[line]
    index = ind[0]
    annot.xy = (xdata[index], ydata[index])
    annot.set_text(f"{xdata[index]:%Y-%m-%d}, {ydata[index]:.2f}")
    annot.get_bbox_patch().set_alpha(0.8)

def hover(event):
    vis = annot.get_visible()
    if event.inaxes == ax:
        for line in lines:
            cont, ind = line.contains(event)
            if cont:
                update_annot(line, ind["ind"])
                annot.set_visible(True)
                fig.canvas.draw_idle()
                return
    if vis:
        annot.set_visible(False)
        fig.canvas.draw_idle()

fig.canvas.mpl_connect("motion_notify_event", hover)
plt.show()
