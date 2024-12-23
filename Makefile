

# CONFIG


.DEFAULT_GOAL := help

CURRENT_DIR := $(CURDIR)

PP=$(CURRENT_DIR)/portfolio-project
PP_DRAW=drawing.py

BASIC_CV2=$(CURRENT_DIR)/basic-setup

BANKNOTE=$(CURRENT_DIR)/bank-notes

PUPPY=$(CURRENT_DIR)/puppy-colors

FILTERING=$(CURRENT_DIR)/filtering
FILTERING_APP=app.py

MORPHING=$(CURRENT_DIR)/morph-text
MORPHING_APP=app.py

ADAPTIVE_THRESHOLDING=$(CURRENT_DIR)/adaptive-thresholding
ADAPTIVE_THRESHOLDING_APP=app.py

## PYTHON CONFIG


# ubuntu

# PYTHON_CONFIG=python3
# PYTHON_PIP_CONFIG=pip
# VNV_ACTIVATE=venv/bin/activate

# windows

PYTHON_CONFIG=py.exe
PYTHON_PIP_CONFIG=py.exe -m pip
VNV_ACTIVATE=venv/Scripts/activate


# TARGETS STARTS


.PHONY: help
help:
	@grep -E '^[a-zA-Z0-9_-]+:.*?## .*' $(MAKEFILE_LIST) | sort

.PHONY: pp-setup
pp-setup: ## setup dependencies and precursors for portfolio project
	@echo "pp: setting up portfolio project virtual env"
	@cd $(PP) && $(PYTHON_CONFIG) -m venv venv && \
		. $(VNV_ACTIVATE) && \
		$(PYTHON_PIP_CONFIG) install --upgrade $(PYTHON_PIP_CONFIG) && \
		$(PYTHON_PIP_CONFIG) install -r requirements.txt

.PHONY: pp-draw
pp-draw: ## executes portfolio project Annotation Draw
	@echo "pp: starting portfolio project annotation drawing"
	@cd $(PP) && \
		. $(VNV_ACTIVATE) && \
		$(PYTHON_CONFIG) $(PP)/$(PP_DRAW)
	@echo "pp: completed portfolio project annotation drawing"

.PHONY: basic-cv2-setup
basic-cv2-setup: ## setup bsic cv2 project
	@cd $(BASIC_CV2) && python3 -m venv venv && \
		. $(VNV_ACTIVATE) && \
		$(PYTHON_PIP_CONFIG) install --upgrade pip && \
		$(PYTHON_PIP_CONFIG) install -r requirements.txt

.PHONY: basic-cv2
basic-cv2: ## executes basic cv2 setup
	@cd $(BASIC_CV2) && \
		. $(VNV_ACTIVATE) && \
		./app.py

.PHONY: banknote-setup
banknote-setup: ## setup banknote project
	@cd $(BANKNOTE) && python3 -m venv venv && \
		. $(VNV_ACTIVATE) && \
		$(PYTHON_PIP_CONFIG) install --upgrade pip && \
		$(PYTHON_PIP_CONFIG) install -r requirements.txt

.PHONY: banknote
banknote: ## executes banknote project
	@cd $(BANKNOTE) && \
		. $(VNV_ACTIVATE) && \
		jupyter notebook

.PHONY: puppy-colors-setup
puppy-colors-setup: ## setup puppy color project
	@cd $(PUPPY) && python3 -m venv venv && \
		. $(VNV_ACTIVATE) && \
		$(PYTHON_PIP_CONFIG) install --upgrade pip && \
		$(PYTHON_PIP_CONFIG) install -r requirements.txt

.PHONY: puppy-colors
puppy-colors: ## executes puppy color project
	@cd $(PUPPY) && \
		. $(VNV_ACTIVATE) && \
		jupyter notebook

.PHONY: filtering-setup
filtering-setup: ## setup filtering project
	@cd $(FILTERING) && $(PYTHON_CONFIG) -m venv venv && \
		. $(VNV_ACTIVATE) && \
		$(PYTHON_PIP_CONFIG) install --upgrade pip && \
		$(PYTHON_PIP_CONFIG) install -r requirements.txt

.PHONY: filtering
filtering: ## executes filtering project
	@cd $(FILTERING) && \
		. $(VNV_ACTIVATE) && \
		$(PYTHON_CONFIG) $(FILTERING_APP)

.PHONY: morph-setup
morph-setup: ## setup morphing project
	@cd $(MORPHING) && $(PYTHON_CONFIG) -m venv venv && \
		. $(VNV_ACTIVATE) && \
		$(PYTHON_PIP_CONFIG) install --upgrade pip && \
		$(PYTHON_PIP_CONFIG) install -r requirements.txt

.PHONY: morph
morph: ## executes morphing project
	@cd $(MORPHING) && \
		. $(VNV_ACTIVATE) && \
		$(PYTHON_CONFIG) $(MORPHING_APP)

.PHONY: adaptive-thresholding-setup
adaptive-thresholding-setup: ## setup adaptive thresholding
	@cd $(ADAPTIVE_THRESHOLDING) && $(PYTHON_CONFIG) -m venv venv && \
		. $(VNV_ACTIVATE) && \
		$(PYTHON_PIP_CONFIG) install --upgrade pip && \
		$(PYTHON_PIP_CONFIG) install -r requirements.txt

.PHONY: adaptive-thresholding
adaptive-thresholding: ## executes adaptive thresholding
	@cd $(ADAPTIVE_THRESHOLDING) && \
		. $(VNV_ACTIVATE) && \
		$(PYTHON_CONFIG) $(ADAPTIVE_THRESHOLDING_APP)
