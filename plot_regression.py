import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from src.config import load_filenames, load_data
from src.ascii_format import INFO, DONE
from estimate import estimate
from sys import stderr


# 'TkAgg' is a backend that supports interactive plotting
matplotlib.use('TkAgg')

"""
This module creates a scatter plot of the data points
and a plot of the linear regression on an external window.
"""


def plot_regression(
    X_label: str, y_label: str,
    X: float, y: float,
    theta0: float, theta1: float
        ) -> None:
    """Create a scatter plot of the data points"""
    print(f"{INFO} Creating a scatter plot of the data points...")

    # Set dark blue background
    plt.style.use('dark_background')

    plt.figure(figsize=(6, 4))  # Set the figure size
    """
    s: size of the points
    alpha: transparency of the points
    """
    plt.scatter(
        X, y, color='#096377',
        edgecolor='royalblue',
        s=100, alpha=0.7, marker='o',
        label='Data points'
        )

    print(f"{DONE} Ready.\n")

    # Generate the regression line
    print(f"{INFO} Generating the regression line...")
    # Generate 100 points within the range of X
    X_range = np.linspace(np.min(X), np.max(X), 100)
    # Predict the y values for all the X values in X_range
    y_pred = estimate(theta0, theta1, X, X_range)

    # Plot the regression line
    plt.plot(
        X_range, y_pred,
        color='#F550B5',
        linestyle='-', linewidth=1,
        label='Regression line'
        )
    # This has the same data points but is wider and more transparent
    # so that it creates a glowing effect on the previous plot
    plt.plot(
        X_range, y_pred,
        color='#F550B5',
        linestyle='-', linewidth=5,
        alpha=0.2,
        )

    print(f"{DONE} Plotted the regression line.\n")

    # Add labels, title, and legend
    plt.xlabel(X_label, color='white', alpha=0.7)
    plt.ylabel(y_label, color='white', alpha=0.7)
    plt.title(
        f'Linear Regression: {X_label} vs. {y_label}',
        color='white',
        fontsize=11,
        alpha=0.7
        )

    # Set background color for the plot
    plt.gcf().set_facecolor('#202844')  # Overall figure background color
    plt.gca().set_facecolor('#202844')  # Axis background color

    plt.grid(True, which='both', linestyle='-', color='#293357', linewidth=0.5)
    # Remove the plot edges (spines) - No edges around the plot
    for spine in plt.gca().spines.values():
        spine.set_visible(False)

    # Customize the ticks on x and y axes
    fs = 10
    plt.xticks(fontsize=fs, color='white', alpha=0.7)
    plt.yticks(fontsize=fs, color='white', alpha=0.7)

    plt.legend(fontsize=fs, loc='best', frameon=False, labelcolor="white")

    # Show the plot
    plt.tight_layout()  # Adjust layout to prevent overlap
    plt.show()


def main():
    print(
        f"{INFO} This program will plot the data "
        "and the regression line on a graph."
        )
    print(f"{INFO} Loading data...")

    # Load filenames from the configuration file
    filenames = load_filenames()
    # Unpack the filenames into two separate variables
    if len(filenames) >= 2:
        thetaset_filename, dataset_filename = filenames
    else:
        print(
            "{ERROR} Missing filename(s) in the configuration file.\n",
            file=stderr
            )

    # Load parameters and feature and target values
    X_label, y_label, theta0, theta1, X, y = load_data(
        thetaset_filename, dataset_filename
        )

    # Run the plotting
    print(f"{INFO} Preparing the plotting...")
    plot_regression(X_label, y_label, X, y, theta0, theta1)


if __name__ == "__main__":
    main()
