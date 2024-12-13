import pandas as pd
from ascii_format import *
import sys

def load_filenames(filename='params/filenames.txt'):
	"""
	This function loads the filenames used in the project
	It loads all lines from the file containing the filenames
	as a list and each line (=1 filename) is stripped and added
	to the returned list
	"""
	with open(filename, 'r') as file:
		return [line.strip() for line in file.readlines()]

def read_parameters_from_file(filename):
    # Load the dataset
	# Open the file and read the first line
	with open(filename, 'r') as file:
		buffer = file.readline().strip()  # Read the first line and remove any trailing whitespace

	# Split the line by the comma
	theta0, theta1 = buffer.split(',')
	if not theta0 or not theta1:
		raise ValueError("Found no value for theta0 and/or theta1")
	# Display the first few rows of the dataset

	return float(theta0), float(theta1) if theta0 and theta1 else None

def load_feature_and_parameters(thetaset_filename, dataset_filename): 
	# Get the theta values from the corresponding file 
	try:
		theta0, theta1 = read_parameters_from_file(thetaset_filename)

		# Display current parameter values
		print("│\n├── Parameters:")
		print(f"│   ├── theta0 = {theta0}\n│   └── theta1 = {theta1}\n│")

		# Load the dataset
		data = pd.read_csv(dataset_filename)

		# Display general informations about the data
		print("└── Data:")
		print(data.describe(), "\n")

	except Exception as e:
		print(f"{ERROR} An unexpected error occurred: {e}", file=sys.stderr)
		sys.exit(1)

	return theta0, theta1, data['km'].values, data['price'].values
