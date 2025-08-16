.PHONY: up

up:
	python3 -m venv .wood
	. .wood/bin/activate; pip install -U pip; pip install -e .
