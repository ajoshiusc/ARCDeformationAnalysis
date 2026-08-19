PYTHON ?= python
CONFIG ?= config/local.toml
REFERENCE := results/reference

.PHONY: install test lint audit reproduce report paper clean-paper

install:
	$(PYTHON) -m pip install -e '.[dev]'

test:
	$(PYTHON) -m pytest

lint:
	$(PYTHON) -m ruff check src tests

audit:
	$(PYTHON) -m arc_deformation audit --config $(CONFIG)

reproduce:
	$(PYTHON) -m arc_deformation reproduce --config $(CONFIG)

report:
	$(PYTHON) -m arc_deformation report \
		--results-dir $(REFERENCE) \
		--output-dir paper/generated

paper: report
	latexmk -pdf -interaction=nonstopmode -halt-on-error -cd paper/main.tex

clean-paper:
	latexmk -C -cd paper/main.tex
