# Data access and portable configuration

Required versions are RAND HRS Longitudinal File 2022 V1, Gateway Harmonized ELSA Version H and Gateway Harmonized KLoSA Version F, with matching codebooks. Different harmonized releases must not be substituted silently. Scientific version labels and original source SHA-256 values remain in `00_admin/shared_source_registry.csv`; no replacement freeze hashes are invented.

Obtain data independently through the cohort providers. Read the [HRS conditions of use](https://hrsdata.isr.umich.edu/data-products/conditions-of-use), [HRS replication policy](https://hrs.isr.umich.edu/sites/default/files/HRSReplicationRules_v5b.pdf), [UK Data Service access conditions](https://ukdataservice.ac.uk/find-data/access-conditions/) and [Gateway acknowledgement/data-access guidance](https://g2aging.org/help/faqs). The [UKDS handling guide](https://ukdataservice.ac.uk/app/uploads/cd171-researchdatahandling.pdf) also addresses non-disclosive output. Repository code does not confer cohort access or permission to redistribute cohort files. Follow dataset acknowledgements and any agreement applicable to your own access.

Copy `config/paths.example.json` to a private location outside the repository and supply your own paths. Relative locations are resolved against the configuration file's directory, not your shell directory. The JSON interface uses only the Python standard library. It supports ordinary Windows, macOS and Linux paths. Data and codebooks must remain outside this Git tree.

PowerShell:

```powershell
$env:WF_PATHS_FILE = (Resolve-Path ../private-paths.json).Path
uv run --locked python -B 02_code/01_panel_build/portable_data_paths.py --verify-hashes
```

macOS/Linux shell:

```sh
export WF_PATHS_FILE="../private-paths.json"
uv run --locked python -B 02_code/01_panel_build/portable_data_paths.py --verify-hashes
```

Configure the exact six source/codebook asset keys in the template. If an explicit asset path is absent, only its registered relative location under `data_root` is tried. Unknown keys, absent inputs, paths inside the repository and source-hash mismatches fail clearly. There is no search for a similar filename and no fallback to the author's environment. Licensed downloads differing in bytes require an explicit source-version review, not editing the expected hash to make a test pass.

`panel_core.resolve_cohort_dataset_path` and `resolve_cohort_codebook_path` use this interface. The R extraction callback in `run_stage.py` uses `Rscript` from PATH or the explicit `WF_RSCRIPT` environment variable. It never invokes a private storage launcher. Codebook identity is checked at the configured external location; redundant copying of a codebook into the repository is unnecessary.

The source-location preflight is available, but full raw-to-model analytical reproduction is NOT VERIFIED and requires rebuilding the integrity assets documented in `reproducibility.md`. Mode A never reads this configuration or checks restricted-data hashes.
