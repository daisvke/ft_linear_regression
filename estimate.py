#!/usr/bin/env python3

import sys
import numpy as np
from src.config import load_filenames, load_data
from src.ascii_format import INFO, ERROR, DONE, BG_YELLOW, RED, RESET
from typing import Any

"""
This program will be used to predict the y value for a given X value.
When you launch the program, it should prompt you for the X value, and then
give you back the estimated y value for that X value.

The program will use the following hypothesis to predict the y value:
    estimateY(X_value) = θ0 + θ1 ∗ X_value
"""


def estimate(
    theta0: float, theta1: float, X: float, x_input: float, verbose=False
        ) -> Any:
    """
    'theta0' and 'theta1' are the parameters.
    'theta0' is the intercept (constant term).
    'theta1' is the slope (how much y changes with X).
    'input_x' is the input feature (an individual measurable property).
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
    x_input_normalized = (x_input - X_mean) / X_std

    if verbose:
        print(f"{INFO} Making estimation...")
        print(f"{INFO} Using hypothesis: estimateY(x) = θ0 + (θ1 * x)")
    return theta0 + (x_input_normalized * theta1)


def main():
    print(f"{INFO} Loading data...")

    # Load filenames from the configuration file
    filenames = load_filenames()
    # Unpack the filenames into two separate variables
    if len(filenames) >= 2:
        thetaset_filename, dataset_filename = filenames
    else:
        print(
            f"{ERROR} Missing filename(s) in the configuration file.\n",
            file=sys.stderr
            )

    # Get thetaset and feature values
    X_label, y_label, theta0, theta1, X, _ = load_data(
        thetaset_filename, dataset_filename
        )

    print(
        f"{INFO} This program will predict the y value ({y_label}) "
        f"from its X value ({X_label}).\n"
        )

    try:  # Prompt the user for mileage
        input_x = int(input(f"├── Enter the X value ({X_label}): "))
        if input_x < 0:
            raise ValueError()
    except ValueError:
        print(
            f"{ERROR} Invalid input. Please enter a positive number.",
            file=sys.stderr
            )
        sys.exit(1)

    # Predict the y value. We only return positive values as
    # y values should not be negative
    predicted_y = max(0, estimate(theta0, theta1, X, input_x, True))
    print(
        f"{DONE} Predicted {y_label}: "
        f"{BG_YELLOW}{RED} {predicted_y} {RESET}\n"
        )


if __name__ == "__main__":
    main()
