.DEFAULT_GOAL := help

CURRENT_DIR := $(CURDIR)

PP=$(CURRENT_DIR)/portfolio-project

BASIC_CV2=$(CURRENT_DIR)/basic-setup


.PHONY: help
help:
	@grep -E '^[a-zA-Z0-9_-]+:.*?## .*' $(MAKEFILE_LIST) | sort

.PHONY: pp-setup
pp-setup: ## setup dependencies and precursors for portfolio project
	@echo "pp: setting up portfolio project virtual env"

.PHONY: pp-test
pp-test: ## executes test portfolio project
	@echo "pp: testing portfolio project dependencies"

.PHONY: pp
pp: ## executes portfolio project
	@echo "checking if gpu libs are available"; make pp-test
	@echo "pp: starting portfolio project"
	@cd $(PP) && \
		. venv/bin/activate && \
		$(PP)/$(PP_APP)
	@echo "pp: completed portfolio project"

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
