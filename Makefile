VENV_DIR	= venv
PYTHON_OS	= python3
PYTHON		= $(VENV_DIR)/bin/python3
PIP			= $(VENV_DIR)/bin/pip
PARAMS_PATH	= params/thetas.txt

# ANSI escape codes for stylized output
RESET 		= \033[0m
BOLD		= \033[1m
CYAN		= \033[36m
GREEN		= \033[32m
YELLOW		= \033[33m
RED			= \033[31m

.PHONY: all install setup train estim freeze clean re

all: setup

# Create the python virtual environment
$(VENV_DIR):
	@echo "$(CYAN)Creating virtual environment...$(RESET)"
	$(PYTHON_OS) -m venv $(VENV_DIR)

# Upgrade pip command, then install the packages
#
# There is an animated progress spinner at the bottom of the terminal.
# Its frame is updated every slept seconds. The 'frames' variable contains
#  all the spinner frames.
# The 'while kill -0 $$pid' loop checks if the process ($$pid) is still running.
# The \r carriage return ensures the spinner stays on the same line.
install: $(VENV_DIR)
	@echo "$(YELLOW)Upgrading pip...$(RESET)"
	$(PIP) install --upgrade pip

	@if [ -f "requirements.txt" ]; then \
		( \
			$(PIP) install -r requirements.txt & \
			pid=$$!; \
			frames="/ | \\ -"; \
			while kill -0 $$pid 2>/dev/null; do \
				for frame in $$frames; do \
					printf "\r$(YELLOW)[Installing] $$frame $(RESET)"; \
					sleep 0.2; \
				done; \
			done; \
			wait $$pid; \
			printf "\r$(GREEN)[Done]         $(RESET)\n"; \
		); \
		else \
		echo "No requirements.txt file found."; \
	fi

setup: $(VENV_DIR) install
	@echo "$(BOLD)$(GREEN)Setup complete. Virtual environment is ready to use.$(RESET)"

train:
	@$(PYTHON) src/model_training.py

estim:
	@$(PYTHON) src/price_prediction.py

# Lists all the installed packages in the current environment along with their versions in requirements.txt.
freeze:
	$(PIP) freeze > requirements.txt

clean:
	@echo "$(CYAN)Initializing parameters...$(RESET)"
	echo "0.0,0.0" > $(PARAMS_PATH)

fclean: clean
	@echo "$(YELLOW)Removing virtual environment...$(RESET)"
	rm -rf $(VENV_DIR)
	@echo "$(GREEN)Virtual environment removed.$(RESET)"

re: clean train
