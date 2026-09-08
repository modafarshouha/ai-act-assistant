PYTHON := $(if $(wildcard .venv/Scripts/python.exe),.venv/Scripts/python.exe,.venv/bin/python)

install:
	python -m venv .venv
	$(PYTHON) -m pip install -r requirements-dev.txt

ingest:
	$(PYTHON) scripts/ingest.py

test:
	$(PYTHON) -m pytest

lint:
	$(PYTHON) -m ruff check app scripts tests

ui:
	$(PYTHON) -m streamlit run app/ui.py

evaluate:
	$(PYTHON) scripts/evaluate.py --json eval/results/eval.json

loadtest:
	$(PYTHON) scripts/loadtest.py -n 100 -c 1 --json eval/results/perf-c1.json
	$(PYTHON) scripts/loadtest.py -n 100 -c 4 --json eval/results/perf-c4.json

.PHONY: install ingest test lint ui evaluate loadtest
