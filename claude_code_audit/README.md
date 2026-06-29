# Claude Code audit

This folder documents the use of Claude Code (Anthropic) in developing the analysis
pipeline, the verification performed, and the issues identified and resolved during
development. It accompanies the Code Availability statement in the manuscript.

## Disclosure

Portions of this pipeline were drafted and refactored with Claude Code (Anthropic,
Claude Opus 4.8). The model was used to: implement and clean the harmonization and
analysis scripts, generate the figures, parameterize paths for reproducibility, run
reproducibility checks, and reconcile the manuscript's reported numbers against the
code's outputs. All analytic decisions (which operationalizations to test, model
specifications, inclusion criteria, framing) were made by the authors, and all code
was reviewed by the corresponding author.

## Contents

- `reproducibility.md` - what was run and which manuscript numbers reproduced
- `findings.md` - issues identified during development, how each was resolved, and
  known limitations / open items
- `changelog.md` - chronological log of substantive edits and analytic decisions

## How the code was checked

Every number reported in the manuscript is collated by `harmonize/11_paper_numbers.py`
into `paper_numbers.json` from pipeline outputs only (nothing is hand-entered). The
figures read the same derivatives. Reproducibility was confirmed by running the
analysis and figure scripts from a clean checkout with paths set through `config.py`;
results are in `reproducibility.md`.
