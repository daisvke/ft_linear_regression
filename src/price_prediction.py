import sys
import argparse
import numpy as np
from config import load_filenames, load_feature_and_parameters
from ascii_format import *

"""
This program will be used to predict the price of a car for a given mileage.
When you launch the program, it should prompt you for a mileage, and then
give you back the estimated price for that mileage.

The program will use the following hypothesis to predict the price:
	estimatePrice(mileage) = θ0 + θ1 ∗ mileage
"""

def estimate_price(theta0, theta1, X, mileage, norm=False, verbose=False):	
	"""
	theta0 is the intercept (constant term).
	theta1 is the slope (how much y changes with x).
	"""

	if verbose:
		print(f"{INFO} Normalizing feature values...")

	"""
	Normalization:
	- The mean is the average of a set of numbers, representing the
		central value of the dataset.
	- The standard deviation measures the spread or dispersion of data
		points from the mean.
	"""
	X_mean = np.mean(X)
	X_std = np.std(X)
	mileage_normalized = (mileage - X_mean) / X_std

	if verbose:
		print(f"{INFO} Making estimation...")
		print(f"{INFO} Using hypothesis: estimatePrice(mileage) = θ0 + (θ1 * mileage)")
	return theta0 + (mileage_normalized * theta1)

def main():
	print(f"{INFO} This program will predict the price of a car from its mileage.\n")
	print(f"{INFO} Loading data...")

	# Load filenames from the configuration file
	filenames = load_filenames()
	# Unpack the filenames into two separate variables
	if len(filenames) >= 2:
		thetaset_filename, dataset_filename = filenames
	else:
		print(f"{ERROR} Missing filename(s) in the configuration file.\n",
			file=sys.stderr)

	try: # Prompt the user for mileage
		mileage = int(input("├── Enter the mileage of the car (in km): "))
		if mileage < 0: raise ValueError()
	except ValueError:
		print(f"{ERROR} Invalid input. Please enter a positive number.",
			file=sys.stderr)
		sys.exit(1)

	# Get thetaset and feature values
	theta0, theta1, X, _ = load_feature_and_parameters(thetaset_filename, dataset_filename)

	# Predict the price of the car. We only return positive values as
	# prices should not be negative
	predicted_price = max(0, estimate_price(theta0, theta1, X, mileage, True))
	print(f"{DONE} Predicted price: {BG_YELLOW}{RED} {predicted_price} {RESET}\n")

if __name__ == "__main__":
	main()
