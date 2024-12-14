# ft_linear_regression

## Description
This project implements a simple linear regression with a single feature - in this case, the mileage of the car.<br />

To do so, we have two programs :<br />
* The first program will be used to predict the price of a car for a given mileage.<br />
- When you launch the program, it should prompt you for a mileage, and then give
you back the estimated price for that mileage. The program will use the following
hypothesis to predict the price :<br />

`estimatePrice(mileage) = θ0 + (θ1 * mileage)`<br />

- Before the run of the training program, theta0 and theta1 will be set to 0.<br />
- screenshot:
<img src="screenshots/training.png" />

* The second program will be used to train the model. It will read the dataset file
and perform a linear regression on the data.<br />
Once the linear regression has completed, the variables theta0 and theta1 will be saved in an external file for use in the first program.<br />

* We will be using the following formulas:
<img src="screenshots/formulas.png" />

* The third program will plot the data and the regression line on a graph using the matplotlib library. It will be displayed on an external window. It is a great way to visualize how well the model fits the data.<br />
<img src="screenshots/plot.png" />

* The fourth program will calculate the precision of the model
<img src="screenshots/precision.png" />

## Requirements
* Some version of python
* Tkinter for python
```
// Install Tkinter (for Ubuntu/Linux)
sudo apt-get update
sudo apt-get install python3-tk
```

## Commands
```
// Set up the environment
make

// Launch the training
make re
or
python src/model_training.py

// Make a prediction
make estim
or
python src/price_prediction.py

// Plotting the data and the regression line on a graph.
make plot
or
python src/plot_regression.py

// Calculate the precision of the model
make precis
or
python src/model_precision.py

```

## Model precision
* Mean Squared Error (MSE)
 - MSE is the average of the squared differences between predicted and actual values.
 - Lower MSE indicates better model performance.
 - The unit of MSE is the square of the target variable (e.g., square of price).

* Root Mean Squared Error (RMSE)
 - RMSE is the square root of MSE, making it easier to interpret because
  it has the same unit as the target variable (e.g., price in dollars).
 - An RMSE of 667.57 means that, on average, the model's predictions
  are off by about 667.57 units.
 - Whether this is "good" depends on the range and scale of your target variable.
 - For example:
    - If car prices in the dataset range between $3,000 and $8,000, an error of $667 is ~10% of the range, which might be acceptable.
    - If prices are closer together (e.g., $5,000-$6,000), this error may be too high.

* R-squared (R²)
 - R2 measures how well the model explains the variance in the data.
 - A value of 0.733 means the model explains 73.3% of the variability in the target variable.
 - R2 ranges from 0 to 1:
    - 1 indicates a perfect model.
    - 0 means the model does no better than simply predicting the mean of the target.

 - For many real-world problems, an R2R2 value above 0.7 is considered "good",
   but this depends on the field:
    - In social sciences, even 0.3 might be acceptable.
    - In physical sciences, models often achieve R² > 0.9.
