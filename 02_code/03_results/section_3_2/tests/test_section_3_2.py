import sys
from pathlib import Path

import numpy as np
import pandas as pd
import patsy


HERE = Path(__file__).resolve().parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import run_section_3_2 as section


def test_primary_formula_has_42_columns_when_all_levels_are_present():
    rows = []
    rng = np.random.default_rng(42)
    levels = {
        "education": ["High", "Medium", "Low"],
        "marital_status": ["Married/partnered", "Separated/divorced", "Widowed", "Never married"],
        "labour_force_status": ["Working", "Unemployed", "Retired or partly retired", "Other economically inactive"],
        "wealth_quintile": ["5", "1", "2", "3", "4"],
    }
    for i in range(100):
        rows.append(
            {
                "fi_primary": 0.2,
                "recovery": int(rng.integers(0, 2)),
                "time_years": float(rng.uniform(0, 8)),
                "age_z": float(rng.normal()),
                "household_size_z": float(rng.normal()),
                "household_income_rank_z": float(rng.normal()),
                "wealth_loss_fraction_z": float(rng.normal()),
                "fi_t_z": float(rng.normal()),
                "sex": rng.choice(["Male", "Female"]),
                "education": rng.choice(levels["education"]),
                "marital_status": rng.choice(levels["marital_status"]),
                "labour_force_status": rng.choice(levels["labour_force_status"]),
                "current_smoking": rng.choice(["True", "False"]),
                "wealth_quintile": rng.choice(levels["wealth_quintile"]),
            }
        )
    _, x = patsy.dmatrices(section.FORMULA, pd.DataFrame(rows), return_type="dataframe")
    assert x.shape[1] == 42
    assert np.linalg.matrix_rank(x.to_numpy()) == 42


def test_meta_analysis_equal_effects_has_zero_heterogeneity():
    primary = pd.DataFrame(
        {
            "cohort": ["HRS", "ELSA", "KLoSA"],
            "estimate": [0.001, 0.001, 0.001],
            "se": [0.0005, 0.0007, 0.0006],
            "is_primary_estimand": [True, True, True],
        }
    )
    result = section.meta_analysis(primary)
    assert abs(result["estimate"] - 0.001) < 1e-10
    assert result["tau2"] < 1e-8
    assert result["i2_percent"] == 0.0


def test_meta_analysis_matches_locked_metafor_fixture():
    fixture = __import__("json").loads((section.ROOT / "00_admin/model_manifests/primary_lmm/metafor_reml_fixture_v1.json").read_text(encoding="utf-8"))
    primary = pd.DataFrame(
        {
            "cohort": ["HRS", "ELSA", "KLoSA"],
            "estimate": fixture["yi"],
            "se": fixture["sei"],
            "is_primary_estimand": [True, True, True],
        }
    )
    result = section.meta_analysis(primary)
    for field, expected in fixture["expected"].items():
        if field == "degrees_of_freedom":
            continue
        assert abs(result[field] - expected) <= 1e-12
    assert section.verify_metafor_fixture(result, primary)["status"] == "PASS"
