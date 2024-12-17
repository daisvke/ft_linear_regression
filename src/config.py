import pandas as pd
from ascii_format import ERROR
import sys
from typing import Tuple

def load_filenames(filename: str = 'params/filenames.txt') -> list:
    """
    This function loads the filenames used in the project
    It loads all lines from the file containing the filenames
    as a list and each line (=1 filename) is stripped and added
    to the returned list
    """
    with open(filename, 'r', newline='') as file:
        return [line.strip() for line in file.readlines()]


def read_parameters_from_file(filename: str) -> Tuple[float, float]:
    """Load the parameters (thetas) from an extern file"""
    # Open the file and read the first line
    with open(filename, 'r', newline='') as file:
        # Read the first line and remove any trailing whitespace
        buffer = file.readline().strip()

    # Split the line by the comma
    theta0, theta1 = buffer.split(',')
    if not theta0 or not theta1:
        raise ValueError("Found no value for theta0 and/or theta1")
    # Display the first few rows of the dataset

    return float(theta0), float(theta1)


def load_data(thetaset_filename, dataset_filename):
    """Load all the necessary data for the regression.

    Parameters
    ----------
     - thetaset_filename: the filename of the file containing the parameters
     - dataset_filename: the filename of the CSV file containing the data

    Returns
    ------
     - X and y labels
     - theta0 and theta1 values
     - X and y values
    """
    try:
        # Get the theta values from the corresponding file
        theta0, theta1 = read_parameters_from_file(thetaset_filename)

        # Display current parameter values
        print("│\n├── Parameters:")
        print(f"│   ├── theta0 = {theta0}\n│   └── theta1 = {theta1}\n│")

        # Load the dataset
        data = pd.read_csv(dataset_filename)
        # Get the column names (= X and y labels)
        X_label, y_label = data.columns.tolist()

        # Display general informations about the data
        print("└── Data:")
        print(data.describe(), "\n")

        # Convert columns to numeric, invalid parsing will be set to NaN
        data[X_label] = pd.to_numeric(data[X_label], errors='coerce')
        data[y_label] = pd.to_numeric(data[y_label], errors='coerce')

        # Check if there are any NaN values
        if data[X_label].isnull().any() or data[y_label].isnull().any():
            raise ValueError("There are non-numeric values in the data")
    except Exception as e:
        print(f"{ERROR} An unexpected error occurred: {e}", file=sys.stderr)
        sys.exit(1)

    return (
        X_label, y_label,
        theta0, theta1,
        data[X_label].values, data[y_label].values
        )
