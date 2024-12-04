.DEFAULT_GOAL := help

CURRENT_DIR := $(CURDIR)

PP=$(CURRENT_DIR)/portfolio-project

PP_DRAW=drawing.py

BASIC_CV2=$(CURRENT_DIR)/basic-setup

BANKNOTE=$(CURRENT_DIR)/bank-notes

PUPPY=$(CURRENT_DIR)/puppy-colors


.PHONY: help
help:
	@grep -E '^[a-zA-Z0-9_-]+:.*?## .*' $(MAKEFILE_LIST) | sort

.PHONY: pp-setup
pp-setup: ## setup dependencies and precursors for portfolio project
	@echo "pp: setting up portfolio project virtual env"
	@cd $(PP) && python3 -m venv venv && \
		. venv/bin/activate && \
		pip install --upgrade pip && \
		pip install -r requirements.txt

.PHONY: pp-draw
pp-draw: ## executes portfolio project Annotation Draw
	@echo "pp: starting portfolio project annotation drawing"
	@cd $(PP) && \
		. venv/bin/activate && \
		$(PP)/$(PP_DRAW)
	@echo "pp: completed portfolio project annotation drawing"

.PHONY: basic-cv2-setup
basic-cv2-setup: ## setup bsic cv2 project
	@cd $(BASIC_CV2) && python3 -m venv venv && \
		. venv/bin/activate && \
		pip install --upgrade pip && \
		pip install -r requirements.txt

.PHONY: basic-cv2
basic-cv2: ## executes basic cv2 setup
	@cd $(BASIC_CV2) && \
		. venv/bin/activate && \
		./app.py

.PHONY: banknote-setup
banknote-setup: ## setup banknote project
	@cd $(BANKNOTE) && python3 -m venv venv && \
		. venv/bin/activate && \
		pip install --upgrade pip && \
		pip install -r requirements.txt

.PHONY: banknote
banknote: ## executes banknote project
	@cd $(BANKNOTE) && \
		. venv/bin/activate && \
		jupyter notebook

.PHONY: puppy-colors-setup
puppy-colors-setup: ## setup puppy color project
	@cd $(PUPPY) && python3 -m venv venv && \
		. venv/bin/activate && \
		pip install --upgrade pip && \
		pip install -r requirements.txt

.PHONY: puppy-colors
puppy-colors: ## executes puppy color project
	@cd $(PUPPY) && \
		. venv/bin/activate && \
		jupyter notebook
	