# Manuscript build

From the repository root:

```bash
make paper
```

This regenerates `paper/generated/results.tex`, `model_table.tex`, and the model
comparison figure from the frozen aggregate CSV/JSON files, then compiles
`paper/main.pdf` with `latexmk`.

The default manuscript is anonymized for blinded review. Before a non-blinded
submission, replace the author block and declarations with the confirmed author
list, affiliations, contributions, funding, acknowledgments, and competing
interest statements. Those facts are intentionally not inferred by code.
