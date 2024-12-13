.PHONY: all train estim clean re

all:
	# Create a virtual environment
	python3 -m venv venv	
	# Activate the virtual environment on Unix
	source lr_env/bin/activate
	# Install packages
	pip install -r requirements

train:
	python3 model_training.py thetas.txt data.csv

estim:
	python3 price_prediction.py thetas.txt data.csv	

# Lists all the installed packages in the current environment along with their versions in requirements.txt.
freeze:
	pip freeze > requirements.txt

clean:
	echo "0.0,0.0" > thetas.txt
	
re: clean train
