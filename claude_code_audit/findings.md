# Findings: issues identified, resolutions, and limitations

Issues surfaced while developing and auditing the pipeline, each checked against the
data, and how it was resolved. Items that change manuscript results are marked.

## Resolved

1. **Two divergent code copies.** The public repository originally contained only the
   four-script dataset *build*; the analysis that produces the manuscript's results
   lived in a separate working copy. The full pipeline (harmonization + analysis +
   figures + literature review) was consolidated, de-duplicated, and parameterized.

2. **"Ever-met" excluded full remission.** The ever-met case definition counted
   present, past, and partial-remission states but not full remission. In this battery
   full remission is recorded only for tic disorders (~250 cases) and a small number of
   ADHD cases. No effect on the manuscript's results: all analyses are fixed at the
   baseline wave, where zero diagnoses are recorded as in full remission (remitted
   cases appear only at later waves). The definition was updated to include full
   remission for completeness. *(No change to reported numbers.)*

3. **Tic sub-disorder mapping (dead code).** The sub-disorder dictionary matched tic
   tokens by exact key and did not catch the remission-coded variants. This had no
   effect on any output, because those rows are excluded upstream by the ever-met
   status filter regardless. Flagged as inert; no behavioral change.

4. **Inferential models strengthened.** Brain-connectivity associations were initially
   estimated without imaging quality control or multi-site/family adjustment. The
   models now apply ABCD-standard RSFC inclusion (imgincl flag, mean FD < 0.5 mm,
   >= 375 retained frames), adjust for study site (and scanner + mean FD for imaging),
   and use family cluster-robust standard errors. Non-imaging estimates were
   essentially unchanged; brain associations became appropriately more fragile once
   motion and site were handled. *(Updated the correlate-stability numbers: significant
   in every spec 34% -> 27%; reversed 21% -> 26%; least-stable measure within-salience
   -> default-mode-frontoparietal.)*

5. **Prevalence framing.** The headline prevalence range was aligned to the three
   primary decisions the manuscript foregrounds (timeframe, informant, grouping) at
   full criteria: 1.9-49.9%. The separate threshold/subthreshold decision extends the
   full defensible range to 56.8% and is reported as such. *(Figure 1 and text use the
   three-decision range.)*

## Known limitations

- The inferential models are logistic GLMs with site fixed effects and family
  cluster-robust SEs, not full generalized linear mixed models.
- Under RSFC quality control, eating disorders has too few cases to estimate
  connectivity models and is omitted from the neuroimaging domain (other domains
  retain all six constructs).
- Build scripts that ingest the access-controlled raw ABCD tables were not
  re-executed in this audit (raw data are not redistributable).

## Open items for author confirmation

- **Psychosis sub-disorder scope.** The released sub-disorder breakdown covers the
  schizophrenia-spectrum diagnoses (schizophrenia, schizoaffective, schizophreniform).
  Attenuated psychosis, delusional disorder, and isolated hallucination diagnoses are
  counted in the category-level Psychosis variable but are not given separate
  sub-disorder columns. No cases are lost. Whether to add delusional disorder as its
  own column is a taxonomy decision pending author review.
- **Single source of truth vs. headline.** `paper_numbers.json` reports the full-grid
  any-disorder range (1.9-56.8%, all operationalizations including threshold); the
  manuscript foregrounds the three-decision subset (1.9-49.9%). These are consistent
  (the latter is a subset); both are documented.
