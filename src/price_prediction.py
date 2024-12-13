import sys
import argparse
import numpy as np
from config import load_filenames, load_feature_and_parameters

'''
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
'''

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

def main():
	# Load filenames from the configuration file
	filenames = load_filenames()
	# Unpack the filenames into two separate variables
	if len(filenames) >= 2:
		thetaset_filename, dataset_filename = filenames
	else:
		print("Missing filename(s) in the configuration file.\n",
			file=sys.stderr)

	# Prompt the user for mileage
	try:
		mileage = int(input("Enter the mileage of the car (in km): "))
	except ValueError:
		print("\033[31mInvalid input. Please enter a number.\033[0m",
			file=sys.stderr)
		sys.exit(1)

	# Get thetaset and feature values
	theta0, theta1, X, _ = load_feature_and_parameters(thetaset_filename, dataset_filename)

	# Predict the price of the car
	predicted_price = estimate_price(theta0, theta1, X, mileage)

	# Print the predicted price
	print(f"\033[33mPredicted price: {predicted_price}\033[0m")

if __name__ == "__main__":
	main()
