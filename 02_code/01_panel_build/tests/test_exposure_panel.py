"""Exposure Panel v1 的两阶段 chronology、恢复边界与冻结测试。"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from openpyxl import load_workbook

MODULE_DIR = Path(__file__).resolve().parents[1]
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

from exposure_panel import (
    AUDIT_ONLY_THRESHOLDS,
    ARCHIVED_FREEZE_PATH,
    EXPECTED_RELEASE_COUNTS,
    EXPOSURE_PANEL_COLUMNS,
    EXPOSURE_PANEL_DTYPES,
    FREEZE_PATH,
    PRIMARY_RECOVERY_THRESHOLD,
    REVIEW_PATH,
    ROOT,
    SENSITIVITY_THRESHOLDS,
    SHOCK_SEARCH_WINDOWS,
    WITHDRAWAL_PATH,
    build_exposure_panel,
    build_shock_pass,
    derive_recovery_indicator,
    select_index_exposure,
    specification,
    specification_hash,
    validate_exposure_panel,
    verify_locked_design,
)
from panel_core import sha256_file


def _candidate_rows(rows: list[dict[str, object]]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    return frame.astype(
        {
            "cohort": "string",
            "person_id": "string",
            "candidate_interval_order": "Int16",
            "tminus1_wave": "Int16",
            "candidate_shock_wave": "Int16",
            "tplus1_wave": "Int16",
            "tminus1_nominal_year": "Int16",
            "shock_wave_nominal_year": "Int16",
            "tplus1_nominal_year": "Int16",
        }
    )


def _eligibility(candidate: pd.DataFrame) -> pd.DataFrame:
    frame = candidate.loc[
        :, ["cohort", "person_id", "candidate_shock_wave"]
    ].copy()
    frame["structural_eligibility_status"] = pd.Series(
        ["pass"] * len(frame), dtype="string"
    )
    return frame


def _wealth_rows(
    rows: list[tuple[str, int, float | None]], cohort: str = "HRS"
) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "cohort": [cohort] * len(rows),
            "person_id": [row[0] for row in rows],
            "wave": [row[1] for row in rows],
            "wealth_total_real_equiv": [row[2] for row in rows],
            "wealth_equiv_divisor": [1.0] * len(rows),
            "wealth_source_flag": [pd.NA] * len(rows),
        }
    )
    return frame.astype(
        {
            "cohort": "string",
            "person_id": "string",
            "wave": "int16",
            "wealth_total_real_equiv": "float64",
            "wealth_equiv_divisor": "Float64",
            "wealth_source_flag": "string",
        }
    )


def _fixture() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    candidate_rows: list[dict[str, object]] = []
    for person_id in ("A", "B", "C", "D", "E", "F"):
        candidate_rows.append(
            {
                "cohort": "HRS",
                "person_id": person_id,
                "candidate_interval_order": 1,
                "tminus1_wave": 2,
                "candidate_shock_wave": 3,
                "tplus1_wave": 4,
                "tminus1_nominal_year": 2002,
                "shock_wave_nominal_year": 2004,
                "tplus1_nominal_year": 2006,
            }
        )
    candidate_rows.extend(
        [
            {
                "cohort": "HRS",
                "person_id": "G",
                "candidate_interval_order": 1,
                "tminus1_wave": 1,
                "candidate_shock_wave": 2,
                "tplus1_wave": 3,
                "tminus1_nominal_year": 2000,
                "shock_wave_nominal_year": 2002,
                "tplus1_nominal_year": 2004,
            },
            {
                "cohort": "HRS",
                "person_id": "G",
                "candidate_interval_order": 2,
                "tminus1_wave": 2,
                "candidate_shock_wave": 3,
                "tplus1_wave": 4,
                "tminus1_nominal_year": 2002,
                "shock_wave_nominal_year": 2004,
                "tplus1_nominal_year": 2006,
            },
            {
                "cohort": "HRS",
                "person_id": "H",
                "candidate_interval_order": 1,
                "tminus1_wave": 13,
                "candidate_shock_wave": 14,
                "tplus1_wave": 15,
                "tminus1_nominal_year": 2016,
                "shock_wave_nominal_year": 2018,
                "tplus1_nominal_year": 2020,
            },
            {
                "cohort": "HRS",
                "person_id": "I",
                "candidate_interval_order": 1,
                "tminus1_wave": 15,
                "candidate_shock_wave": 16,
                "tplus1_wave": pd.NA,
                "tminus1_nominal_year": 2020,
                "shock_wave_nominal_year": 2022,
                "tplus1_nominal_year": pd.NA,
            },
        ]
    )
    candidate = _candidate_rows(candidate_rows)
    wealth_rows: list[tuple[str, int, float | None]] = []
    recovery_wealth = {
        "A": 40.0,   # R = 0.25
        "B": 60.0,   # R = 0.50
        "C": 100.0,  # R = 1.00
        "D": 0.0,    # R = -0.25
        "E": 120.0,  # R = 1.25
        "F": None,   # t+1 财富缺失
    }
    for person_id, tplus1_wealth in recovery_wealth.items():
        wealth_rows.extend(
            [(person_id, 2, 100.0), (person_id, 3, 20.0), (person_id, 4, tplus1_wealth)]
        )
    wealth_rows.extend(
        [
            ("G", 1, 100.0),
            ("G", 2, 20.0),
            ("G", 3, 1.0),
            ("G", 4, 100.0),
            ("H", 1, 100.0),
            ("H", 13, 100.0),
            ("H", 14, 20.0),
            ("H", 15, None),
            ("I", 15, 100.0),
            ("I", 16, 20.0),
        ]
    )
    return candidate, _eligibility(candidate), _wealth_rows(wealth_rows)


def test_specification_and_design_register_are_locked() -> None:
    spec = specification()
    assert len(EXPOSURE_PANEL_COLUMNS) == 13
    assert PRIMARY_RECOVERY_THRESHOLD == 0.50
    assert SENSITIVITY_THRESHOLDS == (0.25, 1.00)
    assert AUDIT_ONLY_THRESHOLDS == (0.75,)
    assert spec["raw_fraction_policy"] == "preserve without clipping or winsorisation"
    assert spec["continuous_linear_sensitivity"] is False
    assert spec["outcome_or_fi_used"] is False
    assert SHOCK_SEARCH_WINDOWS == {"HRS": (3, 14), "ELSA": (2, 8), "KLoSA": (2, 7)}
    assert len(specification_hash()) == 64
    locked = verify_locked_design()
    assert locked["recovery_threshold"]["recommended_value"] == "0.50"
    assert locked["recovery_threshold"]["selected_value"] == "0.50"
    assert locked["shock_search_window_HRS"]["selected_value"] == "W3-W14"
    assert locked["shock_search_window_ELSA"]["selected_value"] == "W2-W8"
    assert locked["shock_search_window_KLoSA"]["selected_value"] == "W2-W7"


def test_recovery_boundaries_raw_tails_missing_and_first_shock() -> None:
    candidate, eligibility, wealth = _fixture()
    panel, _internal = build_exposure_panel(candidate, eligibility, wealth, "HRS")
    validate_exposure_panel(panel, candidate, eligibility, "HRS")
    assert list(panel.columns) == list(EXPOSURE_PANEL_COLUMNS)
    assert {column: str(panel[column].dtype) for column in panel.columns} == EXPOSURE_PANEL_DTYPES

    index = select_index_exposure(panel).set_index("person_id")
    expected = {
        "A": (0.25, "persistent_or_limited_recovery"),
        "B": (0.50, "substantial_recovery"),
        "C": (1.00, "substantial_recovery"),
        "D": (-0.25, "persistent_or_limited_recovery"),
        "E": (1.25, "substantial_recovery"),
    }
    for person_id, (fraction, status) in expected.items():
        assert index.loc[person_id, "recovery_fraction_raw"] == pytest.approx(fraction)
        assert index.loc[person_id, "recovery_status"] == status
    assert pd.isna(index.loc["F", "recovery_fraction_raw"])
    assert pd.isna(index.loc["F", "recovery_status"])
    assert pd.isna(index.loc["H", "wealth_tplus1"])
    assert pd.isna(index.loc["H", "recovery_status"])
    assert "I" not in index.index

    person_g = panel.loc[panel["person_id"].eq("G")].sort_values(
        "candidate_shock_wave"
    )
    assert person_g["major_shock"].tolist() == [True, True]
    assert person_g["is_index_shock"].tolist() == [False, True]
    assert pd.isna(person_g.iloc[0]["wealth_tplus1"])
    assert person_g.iloc[1]["wealth_tplus1"] == pytest.approx(100.0)

    person_i = panel.loc[panel["person_id"].eq("I")].iloc[0]
    assert bool(person_i["major_shock"]) is True
    assert bool(person_i["is_index_shock"]) is False
    assert pd.isna(person_i["wealth_tplus1"])


def test_pass_a_cannot_read_tplus1_or_fi_and_is_invariant_to_tplus1() -> None:
    candidate = _candidate_rows(
        [
            {
                "cohort": "HRS",
                "person_id": "Z",
                "candidate_interval_order": 1,
                "tminus1_wave": 2,
                "candidate_shock_wave": 3,
                "tplus1_wave": 4,
                "tminus1_nominal_year": 2002,
                "shock_wave_nominal_year": 2004,
                "tplus1_nominal_year": 2006,
            }
        ]
    )
    eligibility = _eligibility(candidate)
    wealth_a = _wealth_rows([("Z", 2, 100.0), ("Z", 3, 20.0), ("Z", 4, 0.0)])
    wealth_b = wealth_a.copy()
    wealth_b.loc[wealth_b["wave"].eq(4), "wealth_total_real_equiv"] = 999.0
    wealth_a["fi_score_should_never_be_read"] = 0.1
    wealth_b["fi_score_should_never_be_read"] = 0.9

    pass_a = build_shock_pass(candidate, eligibility, wealth_a, "HRS")
    pass_b = build_shock_pass(candidate, eligibility, wealth_b, "HRS")
    pd.testing.assert_frame_equal(pass_a, pass_b, check_exact=True)
    assert "wealth_tplus1" not in pass_a.columns
    assert not any("fi" in column.lower() for column in pass_a.columns)
    assert pass_a.loc[0, "is_index_shock"]


@pytest.mark.parametrize("cohort", ["HRS", "ELSA", "KLoSA"])
def test_cohort_window_boundaries_and_outside_major_shocks(cohort: str) -> None:
    first_wave, last_wave = SHOCK_SEARCH_WINDOWS[cohort]

    def row(person_id: str, order: int, shock_wave: int) -> dict[str, object]:
        return {
            "cohort": cohort,
            "person_id": person_id,
            "candidate_interval_order": order,
            "tminus1_wave": shock_wave - 1,
            "candidate_shock_wave": shock_wave,
            "tplus1_wave": shock_wave + 1,
            "tminus1_nominal_year": 1990 + 2 * (shock_wave - 1),
            "shock_wave_nominal_year": 1990 + 2 * shock_wave,
            "tplus1_nominal_year": 1990 + 2 * (shock_wave + 1),
        }

    candidate = _candidate_rows(
        [
            row("LOW", 1, first_wave - 1),
            row("LOW", 2, first_wave),
            row("HIGH", 1, last_wave),
            row("HIGH", 2, last_wave + 1),
            row("OUTSIDE", 1, last_wave + 1),
        ]
    )
    wealth = _wealth_rows(
        [
            ("LOW", first_wave - 2, 100.0),
            ("LOW", first_wave - 1, 20.0),
            ("LOW", first_wave, 1.0),
            ("LOW", first_wave + 1, 100.0),
            ("HIGH", last_wave - 1, 100.0),
            ("HIGH", last_wave, 20.0),
            ("HIGH", last_wave + 1, 1.0),
            ("HIGH", last_wave + 2, 100.0),
            ("OUTSIDE", last_wave, 100.0),
            ("OUTSIDE", last_wave + 1, 20.0),
            ("OUTSIDE", last_wave + 2, 100.0),
        ],
        cohort,
    )
    panel, _internal = build_exposure_panel(
        candidate, _eligibility(candidate), wealth, cohort
    )
    low = panel.loc[panel["person_id"].eq("LOW")].sort_values("candidate_shock_wave")
    high = panel.loc[panel["person_id"].eq("HIGH")].sort_values("candidate_shock_wave")
    outside = panel.loc[panel["person_id"].eq("OUTSIDE")].iloc[0]
    assert low["major_shock"].tolist() == [True, True]
    assert low["is_index_shock"].tolist() == [False, True]
    assert high["major_shock"].tolist() == [True, True]
    assert high["is_index_shock"].tolist() == [True, False]
    assert bool(outside["major_shock"]) is True
    assert bool(outside["is_index_shock"]) is False
    assert pd.isna(outside["wealth_tplus1"])


def test_sensitivity_indicators_are_dynamic_and_audit_threshold_is_rejected() -> None:
    candidate, eligibility, wealth = _fixture()
    panel, _internal = build_exposure_panel(candidate, eligibility, wealth, "HRS")
    index = select_index_exposure(panel).set_index("person_id")
    partial = derive_recovery_indicator(index, 0.25)
    primary = derive_recovery_indicator(index, 0.50)
    full = derive_recovery_indicator(index, 1.00)
    assert bool(partial.loc["A"]) is True
    assert bool(primary.loc["A"]) is False
    assert bool(primary.loc["B"]) is True
    assert bool(full.loc["B"]) is False
    assert bool(full.loc["C"]) is True
    assert pd.isna(primary.loc["F"])
    with pytest.raises(ValueError, match="正式 recovery threshold"):
        derive_recovery_indicator(index, 0.75)


@pytest.mark.skipif(not FREEZE_PATH.exists(), reason="Exposure Panel v1 尚未构建")
def test_frozen_release_assets_schema_hashes_counts_and_upstream_contract() -> None:
    freeze = json.loads(FREEZE_PATH.read_text(encoding="utf-8"))
    assert freeze["freeze_status"] == "FROZEN"
    assert freeze["specification_sha256"] == specification_hash()
    assert freeze["physical_columns"] == list(EXPOSURE_PANEL_COLUMNS)
    assert freeze["protected_inputs_before_and_after_identical"] is True
    assert freeze["fi_or_health_outcome_is_not_a_build_input"] is True
    assert freeze["sensitivity_physical_columns_created"] is False
    assert freeze["next_stage_status"] == "COVARIATE_NOT_STARTED"
    assert sha256_file(ARCHIVED_FREEZE_PATH) == freeze[
        "controlled_pre_window_v1_withdrawal"
    ]["archived_freeze_sha256"]
    assert sha256_file(WITHDRAWAL_PATH) == freeze[
        "controlled_pre_window_v1_withdrawal"
    ]["withdrawal_record_sha256"]
    assert sha256_file(REVIEW_PATH) == freeze["review_workbook"]["sha256"]
    dictionary = ROOT / freeze["data_dictionary"]["path"]
    assert sha256_file(dictionary) == freeze["data_dictionary"]["sha256"]

    for cohort, record in freeze["cohorts"].items():
        parquet = ROOT / record["parquet_path"]
        manifest_path = ROOT / record["manifest_path"]
        assert sha256_file(parquet) == record["parquet_sha256"]
        assert sha256_file(manifest_path) == record["manifest_sha256"]
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert set(manifest["build_inputs"]) == {
            "candidate_intervals_v3",
            "eligibility_base_v2",
            "wealth_panel_v1",
            "design_register",
        }
        assert manifest["key"] == ["cohort", "person_id", "candidate_shock_wave"]
        assert manifest["key_unique"] is True
        assert manifest["output"]["sha256"] == record["parquet_sha256"]
        assert manifest["invariants"]["later_shock_substitution_forbidden"] is True
        assert manifest["invariants"]["missing_tplus1_not_comparator"] is True
        assert manifest["invariants"]["index_shock_restricted_to_locked_analytic_window"] is True
        assert manifest["invariants"]["threshold_counts_nested"] is True
        assert manifest["divisor_change_qc"][
            "inferential_sensitivity_automatically_triggered"
        ] is False
        assert [row["role"] for row in manifest["threshold_support"]] == [
            "sensitivity",
            "primary",
            "audit_only",
            "sensitivity",
        ]

        panel = pd.read_parquet(parquet)
        assert list(panel.columns) == list(EXPOSURE_PANEL_COLUMNS)
        assert {column: str(panel[column].dtype) for column in panel.columns} == EXPOSURE_PANEL_DTYPES
        assert len(panel) == record["row_count"]
        assert not panel.duplicated(["cohort", "person_id", "candidate_shock_wave"]).any()
        assert panel.loc[~panel["is_index_shock"], "wealth_tplus1"].isna().all()
        assert panel.loc[~panel["is_index_shock"], "recovery_fraction_raw"].isna().all()
        assert not any(
            column.startswith(("fi_", "outcome_", "covariate_", "weight_"))
            for column in panel.columns
        )
        index = select_index_exposure(panel)
        expected = EXPECTED_RELEASE_COUNTS[cohort]
        assert len(index) == expected["index_shocks"]
        assert int(index["recovery_fraction_raw"].notna().sum()) == expected["classifiable"]
        assert int(index["recovery_status"].eq("substantial_recovery").sum()) == expected[
            "primary_recovery"
        ]
        assert int(index["recovery_status"].eq("persistent_or_limited_recovery").sum()) == expected[
            "primary_comparator"
        ]
        for threshold in (0.25, 0.75, 1.00):
            assert int(index["recovery_fraction_raw"].ge(threshold).sum()) == expected[
                f"recovery_ge_{threshold:.2f}_n"
            ]
        per_person = panel.groupby("person_id")["is_index_shock"].sum()
        assert int(per_person.max()) <= 1
        assert int(panel["is_index_shock"].sum()) == int(index["person_id"].nunique())
        first_wave, last_wave = SHOCK_SEARCH_WINDOWS[cohort]
        assert index["candidate_shock_wave"].between(first_wave, last_wave).all()
        nested = [
            int(index["recovery_fraction_raw"].ge(1.00).sum()),
            int(index["recovery_fraction_raw"].ge(0.75).sum()),
            int(index["recovery_fraction_raw"].ge(0.50).sum()),
            int(index["recovery_fraction_raw"].ge(0.25).sum()),
            int(index["recovery_fraction_raw"].notna().sum()),
        ]
        assert nested == sorted(nested)


@pytest.mark.skipif(not REVIEW_PATH.exists(), reason="Exposure review workbook 尚未构建")
def test_review_workbook_has_only_three_aggregate_reader_sheets() -> None:
    workbook = load_workbook(REVIEW_PATH, data_only=False)
    assert workbook.sheetnames == [
        "1_Definition Contract",
        "2_Shock Flow",
        "3_Recovery QC",
    ]
    for sheet in workbook.worksheets:
        assert sheet.freeze_panes == "A5"
        assert len(sheet.tables) == (2 if sheet.title == "3_Recovery QC" else 1)
        for row in sheet.iter_rows():
            for cell in row:
                if isinstance(cell.value, str):
                    assert not cell.value.startswith("=")
    shock_sheet = workbook["2_Shock Flow"]
    assert shock_sheet["A1"].value == (
        "First-shock identification within the prespecified analytic shock window"
    )
    assert shock_sheet.max_row == 29
    recovery_sheet = workbook["3_Recovery QC"]
    headers = [cell.value for cell in recovery_sheet[4]]
    assert "Median R (IQR)" in headers
    assert "Divisor stable: substantial / comparator" not in headers
