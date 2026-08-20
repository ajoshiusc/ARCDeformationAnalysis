PYTHON ?= python
CONFIG ?= config/local.toml
REFERENCE := results/reference

.PHONY: install test lint audit hodge-sensitivity reproduce report ants-report paper ants-supplement submission clean-paper

install:
	$(PYTHON) -m pip install -e '.[dev]'

test:
	PYTHONPATH=src $(PYTHON) -m pytest

lint:
	$(PYTHON) -m ruff check src tests ants_mni152

audit:
	PYTHONPATH=src $(PYTHON) -m arc_deformation audit --config $(CONFIG)

hodge-sensitivity:
	PYTHONPATH=src $(PYTHON) -m arc_deformation hodge-sensitivity --config $(CONFIG) \
		--primary-hodge-manifest results/runs/hodge/hodge_manifest.csv

reproduce:
	PYTHONPATH=src $(PYTHON) -m arc_deformation reproduce --config $(CONFIG)

report:
	PYTHONPATH=src $(PYTHON) -m arc_deformation report \
		--results-dir $(REFERENCE) \
		--output-dir paper/generated

ants-report:
	PYTHONPATH=src $(PYTHON) ants_mni152/generate_supplement.py \
		--reference-dir ants_mni152/results/reference \
		--generated-dir ants_mni152/paper/generated

paper: report ants-report
	latexmk -pdf -interaction=nonstopmode -halt-on-error -cd paper/main.tex

ants-supplement: ants-report
	latexmk -pdf -interaction=nonstopmode -halt-on-error -cd ants_mni152/paper/supplement_ants.tex

submission: paper ants-supplement
	latexmk -pdf -interaction=nonstopmode -halt-on-error -cd paper/supplement.tex
	latexmk -pdf -interaction=nonstopmode -halt-on-error -cd paper/cover_letter.tex
	pdfunite paper/main.pdf paper/supplement.pdf \
		ants_mni152/paper/supplement_ants.pdf paper/submission.pdf

clean-paper:
	latexmk -C -cd paper/main.tex
	latexmk -C -cd paper/supplement.tex
	latexmk -C -cd paper/cover_letter.tex
	latexmk -C -cd ants_mni152/paper/supplement_ants.tex
	$(RM) paper/submission.pdf
