.PHONY: all train estim clean re

all:
	python3 -m venv lr_env 	
	source lr_env/bin/activate
	pip install -r requirements

train:
	python3 model_training.py thetas.txt data.csv

estim:
	python3 price_prediction.py thetas.txt data.csv	

clean:
	echo "0.0,0.0" > thetas.txt
	
re: clean train
