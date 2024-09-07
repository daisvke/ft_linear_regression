import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score	

def read_data_from_file():
    # Load the dataset
    data = pd.read_csv('data.csv')

    # Display the first few rows of the dataset
    print(data.head())

def parse_args():
	# Create the parser
	parser = argparse.ArgumentParser(description="Process some integers.")

	# Add arguments
	parser.add_argument('args', metavar='N', type=str, nargs='+',
						help='an integer for the accumulator')

	# Parse the arguments
	args = parser.parse_args()

	# Access the arguments
	for arg in args.args:
		print(arg)

def main():
    parse_args();
    read_data_from_file();

if __name__ == "__main__":
	main()
