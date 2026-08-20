# Manuscript build

The manuscript is organized as an Imaging Neuroscience Research Article.
Generate all submission PDFs from the repository root with:

```bash
make submission
```

The reporting commands first regenerate `paper/generated/results.tex`, model,
paired-comparison, association, adjusted-association, and Hodge-sensitivity
tables, plus the model-comparison, association, adjustment, and numerical
sensitivity figures from frozen aggregate CSV/JSON files. The independent
ANTs reporter separately regenerates its QC, pipeline-agreement, association, and
prediction assets from `ants_mni152/results/reference/`, which contains no
participant rows or paths. `latexmk` then builds:

- `main.pdf`;
- `supplement.pdf`;
- `ants_mni152/paper/supplement_ants.pdf`;
- `cover_letter.pdf`.

The submission target also creates `submission.pdf`, an all-in-one
first-submission file containing the main manuscript followed by both complete
supplements. The separate supplements remain available for a revision or for
Editorial Manager if requested.

The main text uses author-year APA-style citations, a single-paragraph abstract,
six keywords, numbered sections, page numbers, and line numbers. The paper
includes the journal-required availability, contribution, competing-interest,
funding, and AI-use sections.

Do not upload until every item under “Author confirmation required” in
`SUBMISSION_CHECKLIST.md` is resolved. Scientific results are frozen and
generated; author order, funding, and conflicts are factual metadata outside
the analysis and remain explicit placeholders.
