import numpy as np
import pandas as pd
from price_prediction import estimate_price

def train_model(thetaset_filename, data, learning_rate, iterations):
    m = len(data)  # Number of observations
    theta0 = 0.0
    theta1 = 0.0

    for _ in range(iterations):
        # Calculate predictions
        predictions = theta0 + (theta1 * data['km'])
        
        # Calculate the gradients
        tmp_theta0 = learning_rate * (1/m) * np.sum(estimate_price(thetaset_filename, data['km']) - data['price'])
        tmp_theta1 = learning_rate * (1/m) * np.sum((estimate_price(thetaset_filename, data['km'] - data['price']) * data['km'])
        
        # Update the parameters
        theta0 -= tmp_theta0
        theta1 -= tmp_theta1

    return theta0, theta1

# Parse given arguments and get the thetaset filename
def parse_args():
	# Create the parser
	parser = argparse.ArgumentParser(description="""This program will be
	used to predict the price of a car for a given mileage. 
	When you launch the program, it should prompt you for a mileage, and then
	give you back the estimated price for that mileage. The program will use
	the following hypothesis to predict the price:
	estimateP rice(mileage) = θ0 + (θ1 ∗ mileage)""")

	# Add arguments. There must be one argumentdescription_text
	parser.add_argument('args', metavar='f', type=str, nargs='+',
						help='a string with the name of the data file to read')

	# Parse the arguments
	args = parser.parse_args()

	# Access the first given argument
	return args.args[0], args.args[1]

def main():
	# From the arguments, get the mileage to predict the price from
	thetaset_filename, dataset_filename = parse_args()

    # Load the dataset
    data = pd.read_csv('data.csv')

    # Set hyperparameters
    learning_rate = 0.01
    iterations = 1000

    # Train the model
    theta0, theta1 = train_model(data, learning_rate, iterations)

    # Save the parameters to a file
    with open('thetas.txt', 'w') as f:
        f.write(f"{theta0},{theta1}")

    print(f"Training complete. Parameters saved: theta0 = {theta0}, theta1 = {theta1}")

if __name__ == "__main__":
    main()
