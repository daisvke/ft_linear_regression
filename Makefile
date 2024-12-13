.PHONY: all train estim freeze clean re

PYTHON_VERS	= python3
PIP_VERS	= pip
VENV_NAME	= venv
PARAMS_PATH	= params/thetas.txt

all:
	# Create a virtual environment
	$(PYTHON_VERS) -m venv $(VENV)
	# Activate the virtual environment on Unix
	source venv/bin/activate
	# Install packages
	$(PIP_VERS) install -r requirements

train:
	$(PYTHON_VERS) src/model_training.py

estim:
	$(PYTHON_VERS) src/price_prediction.py

# Lists all the installed packages in the current environment along with their versions in requirements.txt.
freeze:
	$(PIP_VERS) freeze > requirements.txt

clean:
	echo "0.0,0.0" > $(PARAMS_PATH) # Init parameters

fclean: clean
	deactivate # Deactivate the current environment
	
re: clean train
