import sys
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score	

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
	parser.add_argument('thetaset_file', type=str, help='the name of the thetaset file to read')

	# Parse the arguments
	args = parser.parse_args()

    # Return the two filenames
	return args.thetaset_file

def estimate_price(theta0, theta1, mileage):	
	print(f"estimate >> theta0: {theta0}\ttheta1: {theta1}\tmileage: {mileage}")

	# Return estimation
	return theta0 + (float(mileage) * theta1)

def main():
	# From the arguments, get the mileage to predict the price from
	thetaset_filename = parse_args()

	# Prompt the user for mileage
	mileage = int(input("Enter the mileage of the car (in km): "))

	# Get the theta values from the corresponding file 
	try:
		theta0, theta1 = read_thetas_from_file(thetaset_filename)
	except Exception as e:
		print(f"An unexpected error occurred: {e}")
		sys.exit()

	# Predict the price of the car
	predicted_price = estimate_price(theta0, theta1, mileage)

	# Print the predicted price
	print(f"\nPredicted price: {predicted_price}")

if __name__ == "__main__":
	main()
