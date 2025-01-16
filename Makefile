VENV_DIR	= venv
PYTHON_OS	= python3
PYTHON		= $(VENV_DIR)/bin/python3
PIP			= $(VENV_DIR)/bin/pip
PARAMS_PATH	= params/thetas.txt

# ANSI escape codes for stylized output
RESET 		= \033[0m
GREEN		= \033[32m
YELLOW		= \033[33m
RED			= \033[31m
# Logs levels
INFO 		= $(YELLOW)[INFO]$(RESET)
ERROR		= $(RED)[ERROR]$(RESET)
DONE		= $(GREEN)[DONE]$(RESET)

.PHONY: all install setup train estim freeze clean re

all: setup

# Create the python virtual environment
$(VENV_DIR):
	@echo "$(INFO) Creating virtual environment..."
	$(PYTHON_OS) -m venv $(VENV_DIR)

# Upgrade pip command, then install the packages
#
# There is an animated progress spinner at the bottom of the terminal.
# Its frame is updated every slept seconds. The 'frames' variable contains
#  all the spinner frames.
# The 'while kill -0 $$pid' loop checks if the process ($$pid) is still running.
# The \r carriage return ensures the spinner stays on the same line.
install: $(VENV_DIR)
	@echo "$(INFO) Upgrading pip..."
	$(PIP) install --upgrade pip

	@if [ -f "requirements.txt" ]; then \
		( \
			$(PIP) install -r requirements.txt & \
			pid=$$!; \
			frames="/ - \\ |"; \
			while kill -0 $$pid 2>/dev/null; do \
				for frame in $$frames; do \
					printf "\r$(YELLOW)[Installing] $$frame $(RESET)"; \
					sleep 0.2; \
				done; \
			done; \
			wait $$pid; \
			printf "\r$(DONE) Installation complete.$(RESET)\n"; \
		); \
	else \
		echo "$(ERROR) No requirements.txt file found."; \
	fi

setup: $(VENV_DIR) install
	@echo "$(DONE) Setup complete. Virtual environment is ready to use."

train:
	@$(PYTHON) model_training.py

estimate:
	@$(PYTHON) estimate.py

plot:
	@$(PYTHON) plot_regression.py

precision:
	@$(PYTHON) model_precision.py

# Lists all the installed packages in the current environment along with their versions in requirements.txt.
freeze:
	$(PIP) freeze > requirements.txt

clean:
	@echo "$(INFO) Initializing parameters..."
	echo "0.0,0.0" > $(PARAMS_PATH)

fclean: clean
	@echo "$(INFO) Removing virtual environment..."
	rm -rf $(VENV_DIR)
	@echo "$(DONE) Virtual environment removed."

re: clean train
