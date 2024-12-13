import sys
import argparse
import pandas as pd
import numpy as np

def read_thetas_from_file(filename):
    # Load the dataset
	try:
		# Open the file and read the first line
		with open(filename, 'r') as file:
			buffer = file.readline().strip()  # Read the first line and remove any trailing whitespace

		# Split the line by the comma
		theta0, theta1 = buffer.split(',')
		if not theta0 or not theta1:
			raise ValueError("Found no value for theta0 and/or theta1")
		# Display the first few rows of the dataset
	except Exception as e:
		print(f"An unexpected error occurred: {e}")
		sys.exit()
	return float(theta0), float(theta1) if theta0 and theta1 else None

# Parse given arguments and get the thetaset filename
def parse_args():
	# Create the parser
	parser = argparse.ArgumentParser(description="""This program will be
	used to predict the price of a car for a given mileage. 
	When you launch the program, it should prompt you for a mileage, and then
	give you back the estimated price for that mileage. The program will use
	the following hypothesis to predict the price:
	estimateP rice(mileage) = θ0 + (θ1 ∗ mileage)""")

    # Add arguments. There must be two arguments for filenames
	parser.add_argument('thetaset_filename', type=str, help='the name of the thetaset file to read')
	parser.add_argument('dataset_filename', type=str, help='the name of the dataset file to read')

	# Parse the arguments
	args = parser.parse_args()

    # Return the two filenames
	return args.thetaset_filename, args.dataset_filename

def estimate_price(theta0, theta1, X, mileage):	
	"""
	theta0 is the intercept (constant term).
	theta1 is the slope (how much y changes with x).
	"""
	print(f"\ntheta0 = {theta0} / theta1 = {theta1} / mileage = {mileage}\n")

	# Normalize feature values
	X_mean = np.mean(X)
	X_std = np.std(X)
	mileage_normalized = (mileage - X_mean) / X_std
	# Return estimation
	return theta0 + (mileage_normalized * theta1)

def get_feature_and_parameters(thetaset_filename, dataset_filename): 
	# Get the theta values from the corresponding file 
	try:
		theta0, theta1 = read_thetas_from_file(thetaset_filename)
		# Load the dataset
		data = pd.read_csv(dataset_filename)
		# Describe data
		print(data.describe())

	except Exception as e:
		print(f"An unexpected error occurred: {e}")
		sys.exit()

	return theta0, theta1, data['km'].values, data['price'].values

def main():
	# From the arguments, get the mileage to predict the price from
	thetaset_filename, dataset_filename = parse_args()

	# Prompt the user for mileage
	mileage = int(input("Enter the mileage of the car (in km): "))

	# Get thetaset and feature values
	theta0, theta1, X, _ = get_feature_and_parameters(thetaset_filename, dataset_filename)

	# Predict the price of the car
	predicted_price = estimate_price(theta0, theta1, X, mileage)

	# Print the predicted price
	print(f"\033[33mPredicted price: {predicted_price}\033[0m")

if __name__ == "__main__":
	main()
