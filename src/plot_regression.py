import matplotlib
# 'TkAgg' is a backend that supports interactive plotting
matplotlib.use('TkAgg') 
import matplotlib.pyplot as plt
import numpy as np
from config import load_filenames, load_feature_and_parameters
from ascii_format import *
import pandas as pd
from price_prediction import estimate_price

def plot_regression(X, y, theta0, theta1):
    # Create a scatter plot of the data points
	print(f"{INFO} Creating a scatter plot of the data points...")

	# Set dark blue background
	plt.style.use('dark_background')

	plt.figure(figsize=(6, 4)) # Set the figure size
	"""
	s: size of the points
	alpha: transparency of the points
	"""
	plt.scatter(X, y, color='#096377', edgecolor='royalblue', s=100, alpha=0.7, marker='o', label='Data points')

	print(f"{DONE} Created a scatter plot of the data points\n")

	# Generate the regression line
	print(f"{INFO} Generating the regression line...")
	X_range = np.linspace(min(X), max(X), 100)  # Generate 100 points within the range of X
	# Predict the prices for all the mileage values in X_range
	y_pred = estimate_price(theta0, theta1, X, X_range)

	# Plot the regression line
	plt.plot(X_range, y_pred, color='#F550B5', linestyle='-', linewidth=1, label='Regression line')
	# This has the same data points but is wider and more transparent
	# so that it creates a glowing effect on the previous plot
	plt.plot(X_range, y_pred, color='#F550B5', linestyle='-', alpha=0.2, linewidth=5)
	
	print(f"{DONE} Plotted the regression line\n")

	# Add labels, title, and legend
	plt.xlabel('Mileage (km)', color='white', alpha=0.7)
	plt.ylabel('Price ($)', color='white', alpha=0.7)
	plt.title('Linear Regression: Mileage vs. Price', color='white', fontsize=11, alpha=0.7)

	# Set background color for the plot
	plt.gcf().set_facecolor('#202844')  # Overall figure background color
	plt.gca().set_facecolor('#202844')      # Axis background color

	plt.grid(True, which='both', linestyle='-', color='#293357', linewidth=0.5)
	# Remove the plot edges (spines) - No edges around the plot
	for spine in plt.gca().spines.values():
		spine.set_visible(False)

	# Customize the ticks on x and y axes
	fs = 10
	plt.xticks(fontsize=fs, color='white', alpha=0.7)
	plt.yticks(fontsize=fs, color='white', alpha=0.7)

	plt.legend(fontsize=10, loc='best', frameon=False, labelcolor="white")
    
	# Add annotations for some points (example: min and max price)
	plt.annotate(f'Min Price: {min(y)}', xy=(X[np.argmin(y)], min(y)), xytext=(X[np.argmin(y)] - 9500, min(y) + 300),
                 arrowprops=dict(facecolor='#58B6E6', edgecolor='#58B6E6', arrowstyle='->'), fontsize=9, color='#58B6E6', alpha=0.7)
	plt.annotate(f'Max Price: {max(y)}', xy=(X[np.argmax(y)], max(y)), xytext=(X[np.argmax(y)] - 10000, max(y) + 200),
                 arrowprops=dict(facecolor='#58B6E6', edgecolor='#58B6E6', arrowstyle='->'), fontsize=9, color='#58B6E6', alpha=0.7)

	# Show the plot
	plt.tight_layout()  # Adjust layout to prevent overlap
	plt.show()

def main():
	print(f"{INFO} This program will plot the data and the regression line on a graph.")
	print(f"{INFO} Loading data...")

	# Load filenames from the configuration file
	filenames = load_filenames()
	# Unpack the filenames into two separate variables
	if len(filenames) >= 2:
		thetaset_filename, dataset_filename = filenames
	else:
		print("{ERROR} Missing filename(s) in the configuration file.\n",
			file=sys.stderr)

	# Load parameters and feature and target values
	theta0, theta1, X, y = load_feature_and_parameters(thetaset_filename, dataset_filename)

	# Run the plotting
	print(f"{INFO} Preparing the plotting...")
	plot_regression(X, y, theta0, theta1)

if __name__ == "__main__":
	main()
