import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import pymc3 as pm

# Read the CSV file
def visualize_data(file_path):
    try:
        # Load data
        data = pd.read_csv(file_path)

        # Check if required columns exist
        if 'time' not in data.columns or 'host' not in data.columns:
            raise ValueError("The CSV file must contain 'time' and 'host' columns.")

        # Define the exponential growth model
        def exponential_growth(t, y0, r):
            return y0 * np.exp(r * t)

        # Fit the model to the data
        popt, pcov = curve_fit(exponential_growth, data['time'], data['host'], p0=(data['host'].iloc[0], 0.1))

        # Extract fitted parameters
        y0, r = popt

        # Generate fitted values
        fitted_host = exponential_growth(data['time'], y0, r)

        # Plot the data and the fitted model
        plt.figure(figsize=(10, 6))
        plt.scatter(data['time'], data['host'], alpha=0.7, edgecolors='w', s=100, label='Data')
        plt.plot(data['time'], fitted_host, color='red', linewidth=2, label=f'Fit: y0={y0:.2f}, r={r:.4f}')
        plt.yscale('log')  # Log scale for y-axis
        plt.title('Exponential Growth Model Fit', fontsize=16)
        plt.xlabel('Time', fontsize=14)
        plt.ylabel('Host (Log Scale)', fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.show()

    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
    except pd.errors.EmptyDataError:
        print("Error: The file is empty.")
    except ValueError as ve:
        print(f"Error: {ve}")

# Example usage
# Replace 'your_file.csv' with the path to your CSV file
# visualize_data('your_file.csv')

