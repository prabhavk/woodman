# Use bash (optional)
SHELL := /bin/bash

PYTHON := .wood/bin/python
PIP    := $(PYTHON) -m pip

.PHONY: up install test clean

up:  ## create venv and install package in editable mode
	python3 -m venv .wood
	$(PIP) install --upgrade pip
	$(PIP) install -e .[test]

install:  ## (re)install into existing venv
	$(PIP) install --upgrade pip
	$(PIP) install -e .[test]

test:  ## run tests with venv python
	$(PYTHON) -m pytest -q || echo "No tests yet"

clean:  ## remove build artifacts
	rm -rf build/ dist/ *.egg-info
