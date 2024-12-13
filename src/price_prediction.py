import sys
import argparse
import numpy as np
from config import load_filenames, load_feature_and_parameters
from ascii_format import *

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

def estimate_price(theta0, theta1, X, mileage, verbose=False):	
	"""
	theta0 is the intercept (constant term).
	theta1 is the slope (how much y changes with x).
	"""

	# Normalize feature values
	if verbose:
		print(f"{INFO} Normalizing feature values...")
	X_mean = np.mean(X)
	X_std = np.std(X)
	mileage_normalized = (mileage - X_mean) / X_std
	# Return estimation
	if verbose:
		print(f"{INFO} Making estimation...")
		print(f"{INFO} Applied hypothesis: estimatePrice(mileage) = θ0 + (θ1 * mileage)")
	return theta0 + (mileage_normalized * theta1)

def main():
	# Load filenames from the configuration file
	filenames = load_filenames()
	# Unpack the filenames into two separate variables
	if len(filenames) >= 2:
		thetaset_filename, dataset_filename = filenames
	else:
		print(f"{ERROR} Missing filename(s) in the configuration file.\n",
			file=sys.stderr)
	print("This program will predict the price of a car from its mileage.\n")
	
	try: # Prompt the user for mileage
		mileage = int(input("├── Enter the mileage of the car (in km): "))
	except ValueError:
		print(f"{ERROR} Invalid input. Please enter a number.",
			file=sys.stderr)
		sys.exit(1)

	# Get thetaset and feature values
	theta0, theta1, X, _ = load_feature_and_parameters(thetaset_filename, dataset_filename)

	# Predict the price of the car
	predicted_price = estimate_price(theta0, theta1, X, mileage, True)
	print(f"{DONE} Predicted price: {BG_YELLOW}{RED}{predicted_price}{RESET}\n")

if __name__ == "__main__":
	main()
