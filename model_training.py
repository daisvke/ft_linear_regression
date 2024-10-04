import sys
import numpy as np
import pandas as pd
import argparse
from price_prediction import estimate_price, read_thetas_from_file

def train_model(thetaset_filename, data, learning_rate):
	m = len(data)  # Number of observations
	try:
		theta0, theta1 = read_thetas_from_file(thetaset_filename)
	except Exception as e:
		print(f"An unexpected error occurred: {e}")
		sys.exit()
	iterations = 1000

	# Extract km and price values from the data
	km_values = np.array(data['km'], dtype=float)
	price_values = np.array(data['price'], dtype=float)

	for _ in range(iterations):
		error = 0

		for i in range(m):
			print(f"km: {km_values[i]}, price: {price_values[i]}")
			# Calculate predictions
			prediction = estimate_price(theta0, theta1, km_values[i])  # Assuming estimate_price takes km as input
			error += prediction - price_values[i]  # Calculate the error
			print(f"\033[31mPrediction: {prediction}, Actual Price: {price_values[i]}\033[0m")
			print(f"\033[33mError: {error}\033[0m")

		# Calculate the gradients
		tmp_theta0 = learning_rate * (1/m) * error # Gradient for theta0
		tmp_theta1 = learning_rate * (1/m) * error * km_values[i] # Gradient for theta1

		# Update the parameters
		theta0 -= tmp_theta0
		theta1 -= tmp_theta1

		print(f"\033[32mtheta0 = {theta0}, theta1 = {theta1}\033[0m")
		print(f"TMP theta0 = {tmp_theta0}, theta1 = {tmp_theta1}")

		if np.isnan(theta0) or np.isnan(theta1):
			print("Theta0 or 1 is NaN!!")
			sys.exit()
	return theta0, theta1

# Parse given arguments and get the thetaset filename
def parse_args():
	# Create the parser
	parser = argparse.ArgumentParser(description="""This program will be used to train
	your model. It will read your dataset file and perform a linear regression on
	the data. Once the linear regression has completed, theta0 and theta1 values will be
	saved for use in the price prediction program.
	It will be using the following formulas:
	tmp_theta0 = learningRate * (1/m) * sum(i=0 to m-1) (estimatePrice(mileage[i]) - price[i])
	tmp_theta1 = learningRate * (1/m) * sum(i=0 to m-1) (estimatePrice(mileage[i]) - price[i]) """)

	# Add arguments. There must be two arguments for filenames
	parser.add_argument('thetaset_file', type=str, help='the name of the thetaset file to read')
	parser.add_argument('dataset_file', type=str,
						help='a string with the name of the data file to read')

	# Parse the arguments
	args = parser.parse_args()

	# Access the two given arguments
	return args.thetaset_file, args.dataset_file

def main():
	# From the arguments, get the mileage to predict the price from
	thetaset_filename, dataset_filename = parse_args()

	# Load the dataset
	data = pd.read_csv(dataset_filename)
	# Describe data
	print(data.describe())

	# Set hyperparameters
	learning_rate = 0.01

	# Train the model
	theta0, theta1 = train_model(thetaset_filename, data, learning_rate)

	print(f"Updated parameters: theta0 = {theta0}, theta1 = {theta1}")

	# Save the parameters to a file
	with open('thetas.txt', 'w') as f:
		f.write(f"{theta0},{theta1}")

	print(f"Training complete. Parameters saved: theta0 = {theta0}, theta1 = {theta1}")

if __name__ == "__main__":
	main()
