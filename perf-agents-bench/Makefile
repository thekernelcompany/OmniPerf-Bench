.PHONY: venv install help

VENV := .venv
PY := $(VENV)/bin/python
PIP := $(VENV)/bin/pip

help:
	@echo "Targets: venv, install"

venv:
	python3 -m venv $(VENV)
	$(PIP) install -U pip

install: venv
	$(PIP) install -e .
	@echo "Installed perf-agents-bench into $(VENV)"