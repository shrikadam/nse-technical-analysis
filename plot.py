import os
import pandas as pd
import matplotlib.pyplot as plt

# Define the folder containing the CSV files
folder_path = '.\\data'

# Initialize the plot
plt.figure(figsize=(10, 6))

# Loop through all CSV files in the folder
for filename in os.listdir(folder_path):
    if filename.endswith('.csv'):
        # Extract the name between the first and second underscores
        legend_name = filename.split('_')[1]
        
        # Load the CSV file into a pandas DataFrame
        file_path = os.path.join(folder_path, filename)
        df = pd.read_csv(file_path)

        # Make sure the 'date' column is in datetime format
        df['Date'] = pd.to_datetime(df['Date'])

        # Plot the 'close' column against the 'date' column
        plt.plot(df['Date'], df['Close'], label=legend_name)

# Add labels and title to the plot
plt.xlabel('Date')
plt.ylabel('Close')
plt.title('Close Prices over Time for Different CSVs')

# Add the legend
plt.legend()

# Show the plot
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
