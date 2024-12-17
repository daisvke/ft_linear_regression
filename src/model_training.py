import sys
import numpy as np
from estimate import estimate
from config import load_filenames, load_data
from ascii_format import ERROR, INFO, WARNING, DONE

"""
This program will be used to train the model.
It will read the dataset file and perform a linear regression on the data.

Once the linear regression has completed, theta0 and theta1 values will be
saved for use in the price prediction program.

It will be using the following formulas:

    tmp_theta0 =
        learningRate * (1/m) * sum(i=0 to m-1) (estimatePrice(x[i]) - y[i])

    tmp_theta1 =
        learningRate * (1/m)
            * sum(i=0 to m-1) (
                (estimatePrice(x[i]) - y[i]) * normalized_mileage
                )
"""


def save_parameters_to_file(
    thetaset_filename: str, theta0: float, theta1: float
        ) -> None:
    """Save the parameters (theta0 & theta1) to the corresponding file"""
    if np.isnan(theta0) or np.isnan(theta1):
        print("{ERROR} theta0 or theta1 is NaN\n")
        sys.exit()

    try:
        with open(thetaset_filename, 'w') as f:
            f.write(f"{theta0},{theta1}")
    except Exception as e:
        print(f"{ERROR} An unexpected error occurred: {e}\n")

    print(f"{INFO} Updated parameters: theta0 = {theta0}, theta1 = {theta1}\n")


def train_model(thetaset_filename: str, dataset_filename: str) -> None:
    # Get labels, thetaset, feature (mileage) and target (price) values
    X_label, y_label, theta0, theta1, X, y = load_data(
        thetaset_filename, dataset_filename
        )

    m = len(y)  # Number of observations
    iterations = 1000
    learning_rate = 0.01

    """
    Normalization:
    - The mean is the average of a set of numbers, representing the
        central value of the dataset.
    - The standard deviation measures the spread or dispersion of data
        points from the mean.
    """
    X_mean = np.mean(X)
    X_std = np.std(X)
    X_normalized = (X - X_mean) / X_std

    print(
        f"{INFO} Normalization: "
        f"mean = {X_mean}, standard deviation = {X_std}\n"
        )

    prev_cost = float("inf")    # Set an initial large cost value
    tolerance = 1e-10            # Define a tolerance for cost improvement

    for i in range(iterations):
        prediction = estimate(theta0, theta1, X, X)
        error = (prediction - y)

        """
        A cost (or loss) is a metric that quantifies how well a model's
        predictions match the actual data.

        Using that cost, we will set an early stopping condition for the loop.
        This approach can help detect when the model diverges due to an
        excessively large learning rate or when it's no longer making progress,
        potentially saving computation time.

        """
        cost = np.sum(error ** 2) / (2 * m)

        # Log output every 10 iterations and at the last iteration
        if i % 10 == 0 or i == iterations - 1:
            print(f"{INFO} Iteration {i}:")
            print(
                f"{INFO} Prediction = {prediction[0]}, "
                f"Target {y_label} value = {y[0]}, Cost = {cost}"
                )
        # Check for divergence or convergence
        if cost > prev_cost:  # Set a Stopping Condition
            print(
                f"{WARNING} Cost increased at iteration {i}. "
                "Stopping early to prevent divergence."
                )
            break
        elif abs(prev_cost - cost) < tolerance:
            print(
                f"{WARNING} Cost improvement below tolerance at "
                f"iteration {i}. Converged.")
            break

        prev_cost = cost

        # Compute temporary parameters
        tmp_theta0 = learning_rate * (1/m) * np.sum(error)
        tmp_theta1 = learning_rate * (1/m) * np.sum(error * X_normalized)
        # Update parameters
        theta0 -= tmp_theta0
        theta1 -= tmp_theta1

        # Save parameters every 10 iterations and at the last iteration
        if i % 10 == 0 or i == iterations - 1:
            save_parameters_to_file(thetaset_filename, theta0, theta1)


def main():
    print(f"{INFO} This program will train the model")
    print(f"{INFO} Loading data...")

    try:
        # Load filenames from the configuration file
        filenames = load_filenames()
        # Unpack the filenames into two separate variables
        if len(filenames) >= 2:
            thetaset_filename, dataset_filename = filenames
        else:
            print(
                "{ERROR} Missing filename(s) in the configuration file.\n",
                file=sys.stderr
                )

        # Train the model
        train_model(thetaset_filename, dataset_filename)

        print(f"\n{DONE} Training complete.")
    except KeyboardInterrupt:
        print(f"\n{INFO} Exiting...")


if __name__ == "__main__":
    main()
