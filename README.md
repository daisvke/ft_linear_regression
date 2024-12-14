# ft_linear_regression

## Description
This project implements a simple linear regression with a single feature - in this case, the mileage of the car.<br />

To do so, we have two programs :<br />
* The first program will be used to predict the price of a car for a given mileage.<br />
When you launch the program, it should prompt you for a mileage, and then give
you back the estimated price for that mileage. The program will use the following
hypothesis to predict the price :<br />

`estimatePrice(mileage) = θ0 + (θ1 * mileage)`<br />

Before the run of the training program, theta0 and theta1 will be set to 0.<br />

* The second program will be used to train the model. It will read the dataset file
and perform a linear regression on the data.<br />
Once the linear regression has completed, the variables theta0 and theta1 will be saved in an external file for use in the first program.<br />

* We will be using the following formulas:
<img src="screenshots/formulas.png" />

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
```
