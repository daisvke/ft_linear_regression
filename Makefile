.PHONY: all train estim clean re

all:
	# Create a virtual environment
	python3 -m venv venv	
	# Activate the virtual environment on Unix
	source venv/bin/activate
	# Install packages
	pip install -r requirements

train:
	python3 model_training.py thetas.txt data.csv

estim:
	python3 src/price_prediction.py

# Lists all the installed packages in the current environment along with their versions in requirements.txt.
freeze:
	pip freeze > requirements.txt

clean:
	echo "0.0,0.0" > thetas.txt # Init parameters

fclean: clean
	deactivate # Deactivate the current environment
	
re: clean train
