# Analytical workflow and supported execution

## Validated Mode A

The README commands need only this repository and its locked Python packages. `pytest.ini` restricts default discovery to `tests/`; historical real-data tests are not silently treated as passing. Exposure fixtures test first-shock chronology, missing/negative wealth and recovery boundaries, all three cohort search windows, and alternative thresholds. FI fixtures test the 27/30 and 20/22 rules, observation windows and missing interviews. Tests check the 42-column model design, a metafor meta-analysis fixture, synthetic primary-model fitting, random-slope scaling, spline derivatives, public aggregates and portable path failures.

The final Supplementary Materials PDF and S1–S12 source directory were explicitly confirmed by the author during release preparation. `verify_supplement.py` compares actual table cells at their published four- or six-decimal precision. It handles mathematical minus markup and negative zero without changing numbers. It verifies counts in Tables S9, S11 and S12 in addition to coefficient/SE/CI cells. Table S10's C(8) values are checked as eight times the average annual contrasts, as specified; they are not independent evidence.

## Authorized-data Mode B

These analytical implementations are included for inspection and controlled adaptation. Their full historical orchestration is not independently packaged or validated. The intended order is:

1. Verify exact dataset/codebook identities, mappings, scope rules and source columns.
2. Build Spine v2 → Candidate Intervals v3 → Eligibility Base v2 → Outcome Components v2 → FI Panel v2 → Wealth v1 → Exposure v1 → Covariate v1 → Weight/Design v1.
3. Build Landmark Ledger v1 → Final Eligibility v1 → Analysis Panel v1 using actual interview timing. Preserve first-shock selection, landmark FI, later-FI requirements and complete-case eligibility.
4. Fit primary cohort-specific ML random-intercept models with `run_section_3_2.py` scientific functions `prepare_frame`, `fit_model` and `predictions`. `prepare_frame` checks the study's cohort/sample anchors; the function is not an arbitrary new-cohort modeling API. The exact formula and standardization rules are retained.
5. Combine cohort estimates using `meta_analysis`: REML heterogeneity and modified Knapp–Hartung inference. Do not pool participant records across cohorts.
6. Use `run_section_3_3.py`, `run_weighted_sensitivity.R`, `run_mi_imputation.R` and the `wealth_measurement`, `model_robustness`, `landmark_fi` and `shock_period` modules for their respective specifications. This includes cohort-specific supported bases, raw-loss transformation/restrictions, RS01/NL01, PO01/LM01 and SP01.
7. Generate tables and Figure 3 from authorized local model inputs. `code/restricted/build_figure_s1.py` retains the original flowchart helper, requiring local integrity assets plus XeLaTeX/Poppler/fonts. `code/plot_authorized_diagnostics.py` can reconstruct the residual-vs-time/fitted and normal-Q–Q content described for Figure S2 from authorized local residuals; its layout is not pixel-identical to the final figure. Never commit those individual-level residual inputs.

### Integrity assets and limits

The original end-to-end runners require data-dependent panel manifests, frozen table anchors, parameter references and hash-bound run contracts. These operational assets are not bundled, and their absence is an explicit unsupported boundary, not a reason to access an undisclosed private repository. The source checks remain active. A researcher must build and review integrity records for the licensed input versions in a private work area before using those full-run wrappers. This candidate does not supply a validated automatic rebaselining tool for those records. No historical freeze has been rewritten or represented as byte-identical after redaction.

`config/scientific_contracts/` contains explicitly labeled **projections** of the scientific fields of original contracts, together with source-contract hashes. They exclude internal approvals and run authorization. They are documentation, not replacements accepted by historical hash validators. The original primary and basic sensitivity scientific contracts and the metafor fixture are retained where they contain no private operational records. See `dependency_classes.csv` for dependency classification.

Thus, merely supplying the three raw files does not guarantee that all historical `build`, `fit`, `freeze` or presentation commands will execute. Mode B has not been validated from raw files to the publication; no such claim is made. Full implementation of that orchestration remains a documented limitation of this data-free release. The numerical kernels, mappings and specification documentation are included; unsupported operational steps must fail rather than waive sample, version or integrity requirements.

### R

Use R 4.5.1. In an independent environment, install `renv` and restore `renv.lock` (`renv::restore(lockfile="renv.lock", prompt=FALSE)`). The lock includes WeMix and mice. R is unavailable in the current validation environment, so package restoration, synthetic R bridge tests and weighted/MI cohort execution are NOT VERIFIED. No R packages or vendor source are redistributed here.

### Numerical exceptions

The final approved HRS PO01 and SP01 summaries are retained as `PASS_WITH_CAVEAT`, with their original Stage2 `HOLD_NOT_ESTIMABLE` flags intact. SP01 also retains `AUTHOR_ACCEPTED_NUMERICAL_EXCEPTION`. These flags are preserved in the public aggregate CSVs and tested. They are not generic numerical convergence passes and cannot justify silently changing models. Gaussian residual limitations and sensitivity of HRS random-slope inference remain relevant.

### Figure 3 and other non-reproduced items

The primary runner contains standardization/prediction and Figure 3 code. No respondent predictions or unreviewed prediction grid is included. Figure 3 therefore requires restricted data (or a separately disclosure-cleared aggregate grid). Selected `.tex` files are author-supplied reference text; shipping them is not independent recalculation. Table S5 is not distributed because its covariate-missingness counts include cells of 1–6; repository redistribution of those cells has not been cleared. This exclusion does not change the author-confirmed final Supplement. Table 1 and Tables S5–S6/S8 have not been independently re-estimated. Figure 1 graphics remain excluded. See the coverage matrix for each item.
