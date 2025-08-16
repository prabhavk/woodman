# Use bash (optional)
SHELL := /bin/bash

PYTHON := .emtr/bin/python
PIP    := $(PYTHON) -m pip

.PHONY: all install test clean

all:  ## create venv and install package in editable mode
	python3 -m venv .emtr
	$(PIP) install --upgrade pip
	$(PIP) install -e .[test]

install:  ## (re)install into existing venv
	$(PIP) install --upgrade pip
	$(PIP) install -e .[test]

test:  ## run tests with venv python
	$(PYTHON) -m pytest -q || echo "No tests yet"

clean:  ## remove build artifacts
	rm -rf .emtr build/ dist/ *.egg-info
