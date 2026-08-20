PYTHON ?= python
CONFIG ?= config/local.toml
REFERENCE := results/reference

.PHONY: install test lint audit reproduce report paper submission clean-paper

install:
	$(PYTHON) -m pip install -e '.[dev]'

test:
	PYTHONPATH=src $(PYTHON) -m pytest

lint:
	$(PYTHON) -m ruff check src tests

audit:
	PYTHONPATH=src $(PYTHON) -m arc_deformation audit --config $(CONFIG)

reproduce:
	PYTHONPATH=src $(PYTHON) -m arc_deformation reproduce --config $(CONFIG)

report:
	PYTHONPATH=src $(PYTHON) -m arc_deformation report \
		--results-dir $(REFERENCE) \
		--output-dir paper/generated

paper: report
	latexmk -pdf -interaction=nonstopmode -halt-on-error -cd paper/main.tex

submission: paper
	latexmk -pdf -interaction=nonstopmode -halt-on-error -cd paper/supplement.tex
	latexmk -pdf -interaction=nonstopmode -halt-on-error -cd paper/cover_letter.tex

clean-paper:
	latexmk -C -cd paper/main.tex
	latexmk -C -cd paper/supplement.tex
	latexmk -C -cd paper/cover_letter.tex
