# Imaging Neuroscience submission checklist

Checked against the journal's official Guide for Authors on 2026-08-19.

## Complete

- Research Article structure with numbered Introduction, Materials and Methods,
  Results, Discussion, and Conclusion sections.
- Single-paragraph abstract and six keywords.
- Author identity and affiliation visible for single-blind review.
- Integrated, numbered figures and tables; page and line numbering enabled.
- APA-style author-year bibliography; DOI-bearing references verified against
  publisher, PubMed, or institutional publication records.
- Ethics/consent statement and public-dataset exemption identifier.
- Data and Code Availability, Author Contributions, Funding, Declaration of
  Competing Interests, and generative-AI disclosure sections present.
- Main manuscript, supplement, and cover-letter sources build with
  `make submission`.
- All numerical claims are generated from aggregate result files; no outcome is
  manually entered into the manuscript.
- Pressure and physical-velocity claims are explicitly excluded.

## Author confirmation required before upload

These are factual declarations and are deliberately not inferred by software:

- confirm the complete author list, order, affiliation, and corresponding
  author;
- confirm the provisional CRediT contribution statement;
- replace the Funding placeholder with the exact grant statement or a confirmed
  “no specific funding” declaration;
- replace the Competing Interests placeholder with the author's exact
  declaration;
- confirm the cover-letter originality/exclusive-consideration statement;
- decide whether acknowledgments beyond the generative-AI disclosure are needed;
- create a tagged archival release (for example, Zenodo) and insert its DOI if
  one is available before submission.

## Files to upload

- `paper/main.pdf` (main manuscript);
- `paper/supplement.pdf` (supplementary material);
- figure PDFs/PNGs from `paper/generated/` and
  `results/reference/figures/` if requested separately;
- `paper/cover_letter.pdf` after the confirmations above.

Official requirements:
<https://direct.mit.edu/imag/pages/guide_for_authors>
