import sys
import numpy as np
import pandas as pd
import argparse
from price_prediction import estimate_price, read_thetas_from_file, load_feature_and_parameters
from config import load_filenames

def save_parameters_to_file(thetaset_filename, theta0, theta1):
	"""Save the parameters (theta0 & theta1) to the corresponding file"""
	if np.isnan(theta0) or np.isnan(theta1):
		print("theta0 or theta1 is NaN!\n")
		sys.exit()

	try:
		with open(thetaset_filename, 'w') as f:
			f.write(f"{theta0},{theta1}")
	except Exception as e:
		print(f"An unexpected error occurred: {e}\n")

	print(f"\033[33mUpdated parameters: theta0 = {theta0}, theta1 = {theta1}\033[0m\n")

def train_model(thetaset_filename, dataset_filename):
	# Get thetaset, feature (mileage) and target (price) values
	theta0, theta1, X, y = load_feature_and_parameters(thetaset_filename, dataset_filename)

	m = len(y)  # Number of observations
	iterations = 1000
	learning_rate = 0.01

	# Normalize X
	X_mean = np.mean(X)
	X_std = np.std(X)
	X_normalized = (X - X_mean) / X_std

	prev_cost = float('inf')  # Set an initial large cost value
	tolerance = 1e-6          # Define a tolerance for cost improvement
	print(f"m: {X_mean}, std: {X_std}, X_nom: {X_normalized}")

	for _ in range(iterations):
		prediction = estimate_price(theta0, theta1, X, X) 
		error = (prediction - y)  # Error vector

		"""
		Set an early stopping condition for the loop. This approach can help
		detect when the model diverges due to an excessively large learning
		rate or when it's no longer making progress, potentially saving
		computation time.
		"""
		cost = np.sum(error ** 2) / (2 * m)
		print(f"Iteration {_}: Prediction = {prediction}, Price = {y}, Cost = {cost}")
		# Check for divergence or convergence
		if cost > prev_cost: # Set a Stopping Condition
			print(f"\033[33mCost increased at iteration {_}. Stopping early to prevent divergence.\033[0m")
			break
		elif abs(prev_cost - cost) < tolerance:
			print(f"\033[33mCost improvement below tolerance at iteration {_}. Converged.\033[0m")
			break

		prev_cost = cost

		tmp_theta0 = learning_rate * (1/m) * np.sum(error)
		tmp_theta1 = learning_rate * (1/m) * np.sum(error * X_normalized)
		theta0 -= tmp_theta0
		theta1 -= tmp_theta1

		save_parameters_to_file(thetaset_filename, theta0, theta1)

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
	parser.add_argument('thetaset_filename', type=str, help='the name of the thetaset file to read')
	parser.add_argument('dataset_filename', type=str,
						help='a string with the name of the data file to read')

	# Parse the arguments
	args = parser.parse_args()

	# Access the two given arguments
	return args.thetaset_filename, args.dataset_filename

def main():
	# From the arguments, get the mileage to predict the price from
	thetaset_filename, dataset_filename = parse_args()

	# Train the model
	train_model(thetaset_filename, dataset_filename)

	print(f"\n\033[32mTraining complete.\033[0m")

if __name__ == "__main__":
	main()
