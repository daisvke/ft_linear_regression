import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from estimate import estimate
from src.config import load_filenames, load_data
from src.ascii_format import INFO, ERROR, DONE, BG_YELLOW, RED, RESET
from typing import Any
from sys import stderr


"""
* Mean Squared Error (MSE)
 - MSE is the average of the squared differences between predicted and
  actual values.
 - Lower MSE indicates better model performance.
 - The unit of MSE is the square of the target variable
  (e.g., square of price).

* Root Mean Squared Error (RMSE)
 - RMSE is the square root of MSE, making it easier to interpret because
  it has the same unit as the target variable (e.g., price in dollars).
 - An RMSE of 667.57 means that, on average, the model's predictions
  are off by about 667.57 units.
 - Whether this is "good" depends on the range and scale of your
  target variable.
 - For example:
    - If car prices in the dataset range between $3,000 and $8,000, an error
     of $667 is ~10% of the range, which might be acceptable.
    - If prices are closer together (e.g., $5,000-$6,000), this error may be
     too high.

* R-squared (R²)
 - R2 measures how well the model explains the variance in the data.
 - A value of 0.733 means the model explains 73.3% of the variability in
  the target variable.
 - R2 ranges from 0 to 1:
    - 1 indicates a perfect model.
    - 0 means the model does no better than simply predicting the mean of
     the target.

 - For many real-world problems, an R2R2 value above 0.7 is considered "good",
   but this depends on the field:
    - In social sciences, even 0.3 might be acceptable.
    - In physical sciences, models often achieve R² > 0.9.
"""


def calculate_precision(y_target: float, y_pred: float) -> Any:
    """
    Calculate precision metrics: MSE, RMSE, and R².

    Parameters
    ----------
     - The target y value
     - The predicted y value

    Returns
    -------
     - The precision values
    """
    mse = mean_squared_error(y_target, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_target, y_pred)
    r2 = max(0, r2)  # r2 has a minimum value of 0
    return mse, rmse, r2


def main():
    print(f"{INFO} This program will calculate the precision of the model.")
    print(f"{INFO} Loading data...")

    # Load filenames from the configuration file
    filenames = load_filenames()
    # Unpack the filenames into two separate variables
    if len(filenames) >= 2:
        thetaset_filename, dataset_filename = filenames
    else:
        print(
            f"{ERROR} Missing filename(s) in the configuration file.\n",
            file=stderr
            )

    # Get labels, thetaset and feature values
    X_label, y_label, theta0, theta1, X, y = load_data(
        thetaset_filename, dataset_filename
        )

    # Predict y values
    print(f"{INFO} Estimating {y_label}s using thetaset...")
    y_pred = estimate(theta0, theta1, X, X)

    # Calculate precision metrics
    print(f"{INFO} Calculating precision...")
    mse, rmse, r2 = calculate_precision(y, y_pred)

    # Print results
    print(f"\n{DONE} Computed model precision.\n")
    print(f"Mean Squared Error (MSE):\t{BG_YELLOW}{RED}{mse}{RESET}")
    print(f"Root Mean Squared Error (RMSE):\t{BG_YELLOW}{RED}{rmse}{RESET}")
    print(f"R-squared (R²):\t\t\t{BG_YELLOW}{RED}{r2}{RESET}\n")


if __name__ == "__main__":
    main()
