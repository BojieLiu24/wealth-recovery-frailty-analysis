# Wealth recovery and subsequent frailty progression

Code and aggregate verification materials accompanying **Wealth Recovery and Subsequent Frailty Progression after a Major Wealth Shock: A Longitudinal Study across Three National Ageing Cohorts**, by Bojie Liu, Jia Guo, Dong Zhang, Zaiying Zhou and Yichen Dong. Accepted by *Healthcare* on 16 November 2026, manuscript healthcare-4603924. A DOI has not been verified.

The study compares frailty progression after substantial wealth recovery versus persistent loss or limited recovery in HRS, ELSA and KLoSA. A qualifying shock is a ≥75% loss; substantial recovery means regaining ≥50% of that shock-related loss. The primary longitudinal outcome includes the landmark FI and subsequent eligible FI assessments.

## Mode A: no cohort data required

Requires Python 3.14.7 and [uv](https://docs.astral.sh/uv/). From the repository root:

```sh
uv sync --locked --group dev
uv run --locked python -B -m pytest -q
uv run --locked python -B code/verify_public.py
uv run --locked python -B code/verify_supplement.py
```

These commands run synthetic scientific tests, check primary cohort counts and estimates, recompute the secondary meta-analysis from three aggregate estimates, regenerate Figure 2, and compare 231 displayed numerical cells in Supplementary Tables S7 and S9–S12 against the included aggregate summaries. Generated figures and `verification.json` are written to `outputs/generated/`.

**This is aggregate-based verification, not independent re-estimation of the cohort models.** The tests additionally exercise the original primary LMM implementation on explicitly synthetic observations. No external SSD, private repository, private storage package or individual-level dataset is required for Mode A.

## Contents

| Directory | Contents |
|---|---|
| `code/` | Portable aggregate verification and figure commands; restricted-data figure helpers |
| `tests/` | Default data-free synthetic and interface tests |
| `02_code/01_panel_build/` | Cohort-specific construction functions and portable source-location interface |
| `02_code/03_results/` | Original primary and sensitivity statistical implementations |
| `00_admin/` | Scientific registers, source versions and variable mappings |
| `config/` | Local path template and explicitly identified scientific contract projections |
| `outputs/published_aggregates/` | Non-identifying cohort-level estimate tables, Figure 2 input and provenance |
| `docs/` | Coverage matrix, definitions, data access and analytical limitations |

## Mode B: authorized individual-level data

Obtain the specified HRS, ELSA and KLoSA releases independently, then configure `config/paths.example.json` as described in [data access](docs/data_access.md). No raw or derived participant datasets are distributed. The public code preserves cohort-specific mappings, formulas, timing, integrity checks and numerical exception information.

**Mode B is a documented collection of analytical implementations, not a validated one-command raw-to-publication pipeline.** Original full-run wrappers require locally rebuilt integrity manifests, approved run contracts and data-dependent table/model inputs that are not distributed. Missing integrity evidence causes failure; it is not silently bypassed. See [the supported workflow and remaining limits](docs/reproducibility.md) before executing data-dependent commands.

The [coverage matrix](docs/coverage_matrix.csv) distinguishes aggregate verification, restricted-data execution, documented-only items and unavailable assets. Figure 1's third-party graphics are excluded; a [text description](docs/study_design.md) is provided. Figure 3's prediction grid and individual-level diagnostic data are excluded pending disclosure review. Selected supplementary reference tables are author-supplied scientific text, not newly estimated results. Table S5 is excluded because its covariate-missingness panel contains small cells that have not been cleared for repository redistribution.

## Environment and validation

Python versions are locked in `uv.lock`. R 4.5.1 and the packages in `renv.lock` are needed for weighting and imputation. R restoration and actual cohort-model execution have not been verified in this release. Windows data-free execution, including a relocated copy that blocks access to the private research directory and private storage imports, is tested; macOS/Linux execution remains unverified. Fonts may differ where Arial is unavailable; Figure 2 falls back to DejaVu Sans without altering estimates.

No participant records, IDs, derived panels, respondent predictions, BLUPs, residual arrays, internal correspondence, journal PDFs or binary model objects are included. Publication-level model summaries are separated from cohort data. HRS PO01 and SP01 retain their original numerical HOLD status alongside the accepted result caveat; they are not relabeled ordinary numerical passes.

## Citation, contact and rights

See `CITATION.cff`. Contact: Bojie Liu, lbj24@mails.tsinghua.edu.cn; corresponding author Yichen Dong, ycdong@bjtu.edu.cn.

The authors have confirmed the necessary authorization to publish the selected materials. No open-source software license has been granted for this release; public availability does not imply MIT licensing or unrestricted reuse. MIT remains a proposal subject to explicit approval. Third-party software retains its own licenses. Cohort access and data-use conditions are independent of code licensing.
