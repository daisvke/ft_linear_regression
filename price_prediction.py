import sys
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score	

def read_data_from_file(filename):
    # Load the dataset
	try:
		data = pd.read_csv(filename)
		# Display the first few rows of the dataset
	except Exception as e:
		print(f"An unexpected error occurred: {e}")
		sys.exit()
	return data

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
    parser.add_argument('thetaset_file', type=str, help='the name of the dataset file to read')
    parser.add_argument('dataset_file', type=str, help='the name of the thetaset file to read')

    # Parse the arguments
    args = parser.parse_args()

    # Return the two filenames
    return args.file1, args.file2

def estimate_price(thetaset_filename, mileage):
	# Get the theta values from the corresponding file 
	thetaset = read_data_from_file(thetaset_filename)
	
	# Extracting the independent and dependent variables
	theta0 = thetaset['theta0'].values[0]
	theta1 = thetaset['theta1'].values[0]
	
	print(f"theta0: {theta0}\ntheta1: {theta1}")

	# Return estimation
	return theta0 + (mileage * theta1)

def main():
	# From the arguments, get the mileage to predict the price from
	thetaset_filename = parse_args()

	# Prompt the user for mileage
	mileage = int(input("Enter the mileage of the car (in km): "))

	# Predict the price of the car
	predicted_price = estimate_price(thetaset_filename, mileage)

	# Print the predicted price
	print(f"\nPredicted price: {predicted_price}")

if __name__ == "__main__":
	main()
