# Manuscript build

The manuscript is organized as an Imaging Neuroscience Research Article.
Generate all submission PDFs from the repository root with:

```bash
make submission
```

The reporting command first regenerates `paper/generated/results.tex`, model
tables, the model-comparison figure, and the association figure from frozen
aggregate CSV/JSON files. `latexmk` then builds:

- `main.pdf`;
- `supplement.pdf`;
- `cover_letter.pdf`.

The main text uses author-year APA-style citations, a single-paragraph abstract,
six keywords, numbered sections, page numbers, and line numbers. The paper
includes the journal-required availability, contribution, competing-interest,
funding, and AI-use sections.

Do not upload until every item under “Author confirmation required” in
`SUBMISSION_CHECKLIST.md` is resolved. Scientific results are frozen and
generated; author order, funding, and conflicts are factual metadata outside
the analysis and remain explicit placeholders.
