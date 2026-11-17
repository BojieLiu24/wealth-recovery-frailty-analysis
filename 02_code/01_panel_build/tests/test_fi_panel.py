"""正式 FI Panel v1/v2 的 specification、构建边界与冻结测试。"""

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

from fi_panel import (
    ACTIVE_SPEC,
    COMMON22_THRESHOLD,
    FI_PANEL_COLUMNS,
    FORMAL_WINDOWS,
    PRIMARY_THRESHOLD,
    ROOT,
    V1_SPEC,
    V2_SPEC,
    build_fi_panel,
    fi_specification,
    specification_hash,
    validate_fi_panel,
    verify_locked_design,
    with_fi_completeness_flags,
)
from outcome_components import COMMON_22, PRIMARY_ITEMS
from panel_core import sha256_file


V1_FREEZE_PATH = ROOT / "00_admin/panel_manifests/fi_panel/fi_panel_v1_freeze.json"
V2_FREEZE_PATH = ROOT / "00_admin/panel_manifests/fi_panel/fi_panel_v2_freeze.json"


def _synthetic_spine() -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "cohort": ["HRS"] * 8,
            "person_id": ["1"] * 4 + ["2"] * 4,
            "wave": [1, 2, 3, 4] * 2,
            "observed_in_wave": [True, True, True, True, True, True, True, False],
        }
    )
    return frame.astype(
        {
            "cohort": "string",
            "person_id": "string",
            "wave": "int16",
            "observed_in_wave": "boolean",
        }
    )


def _synthetic_outcome(release: str) -> pd.DataFrame:
    spine = _synthetic_spine()
    output = spine.loc[:, ["cohort", "person_id", "wave"]].copy()
    for item in PRIMARY_ITEMS["HRS"]:
        output[item] = pd.Series(0.5, index=output.index, dtype="Float64")
    if release == "v1":
        # person 1 恰好 24/30 与 18/22；person 2 为 23/30 与 17/22。
        exact_common_missing = list(COMMON_22[-4:])
        exact_extension_missing = list(PRIMARY_ITEMS["HRS"][-2:])
        below_common_missing = list(COMMON_22[-5:])
        below_extension_missing = list(PRIMARY_ITEMS["HRS"][-2:])
    elif release == "v2":
        # person 1 恰好 27/30 与 20/22；person 2 为 26/30 与 19/22。
        exact_common_missing = list(COMMON_22[-2:])
        exact_extension_missing = list(PRIMARY_ITEMS["HRS"][-1:])
        below_common_missing = list(COMMON_22[-3:])
        below_extension_missing = list(PRIMARY_ITEMS["HRS"][-1:])
    else:
        raise ValueError(release)
    output.loc[2, exact_common_missing + exact_extension_missing] = pd.NA
    output.loc[6, below_common_missing + below_extension_missing] = pd.NA
    return output


def test_specifications_are_exact_and_versioned() -> None:
    assert ACTIVE_SPEC == V2_SPEC
    assert FORMAL_WINDOWS == {"HRS": (3, 16), "ELSA": (1, 10), "KLoSA": (1, 9)}
    assert PRIMARY_THRESHOLD == 27
    assert COMMON22_THRESHOLD == 20
    assert len(COMMON_22) == 22
    assert all(len(PRIMARY_ITEMS[cohort]) == 30 for cohort in PRIMARY_ITEMS)
    assert fi_specification(V1_SPEC)["qc90_reference"]["creates_alternative_fi"] is False
    assert fi_specification(V2_SPEC)["qc80_reference"]["creates_alternative_fi"] is False
    assert len(specification_hash(V2_SPEC)) == 64
    frozen_v1 = json.loads(V1_FREEZE_PATH.read_text(encoding="utf-8"))
    assert specification_hash(V1_SPEC) == frozen_v1["specification_sha256"]


def test_design_register_has_exact_locked_v2_decisions() -> None:
    rows = verify_locked_design(V2_SPEC)
    assert len(rows) == 6
    assert rows["fi_primary_definition"]["selected_value"] == "cohort_specific_primary30"
    assert rows["fi_common22_definition"]["selected_value"] == "common22_sensitivity"
    assert rows["fi_completeness_rule"]["selected_value"] == "at_least_90_percent"


@pytest.mark.parametrize(
    ("release_spec", "release", "exact_primary", "exact_common", "below_primary", "below_common"),
    [
        (V1_SPEC, "v1", 24, 18, 23, 17),
        (V2_SPEC, "v2", 27, 20, 26, 19),
    ],
)
def test_build_boundaries_and_opportunity_mask(
    release_spec,
    release: str,
    exact_primary: int,
    exact_common: int,
    below_primary: int,
    below_common: int,
) -> None:
    spine = _synthetic_spine()
    outcome = _synthetic_outcome(release)
    panel = build_fi_panel(outcome, spine, "HRS", release_spec)
    validate_fi_panel(panel, spine, outcome, "HRS", release_spec)
    assert list(panel.columns) == list(FI_PANEL_COLUMNS)
    assert panel.loc[
        panel["wave"].isin([1, 2]), list(FI_PANEL_COLUMNS[3:])
    ].isna().all(axis=None)

    exact = panel.loc[(panel["person_id"] == "1") & (panel["wave"] == 3)].iloc[0]
    assert exact["fi_primary_n_valid"] == exact_primary
    assert exact["fi_common22_n_valid"] == exact_common
    assert exact["fi_primary"] == pytest.approx(0.5)
    assert exact["fi_common22"] == pytest.approx(0.5)

    below = panel.loc[(panel["person_id"] == "2") & (panel["wave"] == 3)].iloc[0]
    assert below["fi_primary_n_valid"] == below_primary
    assert below["fi_common22_n_valid"] == below_common
    assert pd.isna(below["fi_primary"])
    assert pd.isna(below["fi_common22"])

    noninterview = panel.loc[
        (panel["person_id"] == "2") & (panel["wave"] == 4)
    ].iloc[0]
    assert noninterview[list(FI_PANEL_COLUMNS[3:])].isna().all()


def test_complete_flags_are_dynamic_specification_aware_and_not_physical() -> None:
    panel = build_fi_panel(
        _synthetic_outcome("v2"), _synthetic_spine(), "HRS", V2_SPEC
    )
    assert "fi_primary_complete" not in panel
    assert "fi_common22_complete" not in panel
    derived = with_fi_completeness_flags(panel, V2_SPEC)
    exact = derived.loc[(derived["person_id"] == "1") & (derived["wave"] == 3)].iloc[0]
    below = derived.loc[(derived["person_id"] == "2") & (derived["wave"] == 3)].iloc[0]
    outside = derived.loc[(derived["person_id"] == "1") & (derived["wave"] == 2)].iloc[0]
    assert bool(exact["fi_primary_complete"])
    assert bool(exact["fi_common22_complete"])
    assert not bool(below["fi_primary_complete"])
    assert not bool(below["fi_common22_complete"])
    assert pd.isna(outside["fi_primary_complete"])
    assert pd.isna(outside["fi_common22_complete"])


@pytest.mark.skipif(not V1_FREEZE_PATH.exists(), reason="FI Panel v1 尚未构建")
def test_frozen_v1_hashes_remain_valid() -> None:
    freeze = json.loads(V1_FREEZE_PATH.read_text(encoding="utf-8"))
    assert freeze["freeze_status"] == "FROZEN"
    assert freeze["specification_sha256"] == specification_hash(V1_SPEC)
    assert sha256_file(ROOT / freeze["review_workbook"]["path"]) == freeze["review_workbook"]["sha256"]
    assert sha256_file(ROOT / freeze["data_dictionary"]["path"]) == freeze["data_dictionary"]["sha256"]
    for cohort, record in freeze["cohorts"].items():
        assert sha256_file(ROOT / record["parquet_path"]) == record["parquet_sha256"]
        assert sha256_file(ROOT / record["manifest_path"]) == record["manifest_sha256"]
        assert cohort in PRIMARY_ITEMS


@pytest.mark.skipif(not V2_FREEZE_PATH.exists(), reason="FI Panel v2 尚未构建")
def test_frozen_v2_hashes_schema_dependencies_and_top_tail() -> None:
    freeze = json.loads(V2_FREEZE_PATH.read_text(encoding="utf-8"))
    assert freeze["freeze_status"] == "FROZEN"
    assert freeze["publication_sync_status"] == "NOT_STARTED_REQUIRES_USER_APPROVAL"
    assert freeze["qc_context"]["candidate_and_eligibility_are_not_build_inputs"] is True
    assert freeze["qc_context"]["top_tail_audit_is_not_a_build_input"] is True
    assert freeze["specification_sha256"] == specification_hash(V2_SPEC)
    assert freeze["supersedes"]["fi_panel_v1_freeze_sha256"] == sha256_file(V1_FREEZE_PATH)
    expected_schema = [
        {"name": column, "dtype": dtype, "nullable": nullable}
        for column, dtype, nullable in (
            ("cohort", "string", False),
            ("person_id", "string", False),
            ("wave", "int16", False),
            ("fi_primary", "Float64", True),
            ("fi_primary_n_valid", "Int16", True),
            ("fi_common22", "Float64", True),
            ("fi_common22_n_valid", "Int16", True),
        )
    ]
    expected_counts = {
        "HRS": (260535, 261392, 5094),
        "ELSA": (97399, 97387, 757),
        "KLoSA": (69272, 69272, 917),
    }
    for cohort, record in freeze["cohorts"].items():
        parquet = ROOT / record["parquet_path"]
        manifest_path = ROOT / record["manifest_path"]
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert sha256_file(parquet) == record["parquet_sha256"]
        assert sha256_file(manifest_path) == record["manifest_sha256"]
        assert set(manifest["build_inputs"]) == {"outcome_components", "spine"}
        assert manifest["schema"] == expected_schema
        assert manifest["top_tail_qc"]["decision"] == "PASS"
        assert manifest["top_tail_qc"]["raw_rebuild_component_cell_differences"] == 0
        assert manifest["top_tail_qc"]["direct_raw_score_cell_differences"] == 0
        assert manifest["top_tail_qc"]["tagged_cells_scored_nonmissing"] == 0
        assert manifest["top_tail_qc"]["unrecognised_numeric_code_count"] == 0
        panel = pd.read_parquet(parquet)
        expected_primary, expected_common, expected_tail = expected_counts[cohort]
        assert int(panel["fi_primary"].notna().sum()) == expected_primary
        assert int(panel["fi_common22"].notna().sum()) == expected_common
        assert manifest["top_tail_qc"]["v2_retained_tail_n"] == expected_tail
        assert "fi_primary_complete" not in panel
        assert "fi_common22_complete" not in panel


@pytest.mark.skipif(not V2_FREEZE_PATH.exists(), reason="FI Panel v2 尚未构建")
def test_real_v2_panels_match_spine_and_are_v1_subsets() -> None:
    freeze_v1 = json.loads(V1_FREEZE_PATH.read_text(encoding="utf-8"))
    freeze_v2 = json.loads(V2_FREEZE_PATH.read_text(encoding="utf-8"))
    for cohort, record_v2 in freeze_v2["cohorts"].items():
        panel_v2 = pd.read_parquet(ROOT / record_v2["parquet_path"])
        panel_v1 = pd.read_parquet(ROOT / freeze_v1["cohorts"][cohort]["parquet_path"])
        spine = pd.read_parquet(
            ROOT / f"01_data/derived/panel_components/{cohort}/01_spine/{cohort.lower()}_spine.parquet"
        )
        outcome = pd.read_parquet(
            ROOT / f"01_data/derived/panel_components/{cohort}/04_outcome_components/{cohort.lower()}_outcome_components_v2.parquet"
        )
        validate_fi_panel(panel_v2, spine, outcome, cohort, V2_SPEC)
        pd.testing.assert_series_equal(
            panel_v2["fi_primary_n_valid"], panel_v1["fi_primary_n_valid"], check_exact=True
        )
        pd.testing.assert_series_equal(
            panel_v2["fi_common22_n_valid"], panel_v1["fi_common22_n_valid"], check_exact=True
        )
        for score in ("fi_primary", "fi_common22"):
            assert panel_v2[score].notna().le(panel_v1[score].notna()).all()
            overlap = panel_v2[score].notna()
            np.testing.assert_allclose(
                panel_v2.loc[overlap, score].astype(float),
                panel_v1.loc[overlap, score].astype(float),
                rtol=0,
                atol=0,
            )
        assert not any(
            column.startswith(("wealth_", "recovery_", "weight_", "eligibility_"))
            for column in panel_v2.columns
        )


@pytest.mark.skipif(not V2_FREEZE_PATH.exists(), reason="FI Panel v2 尚未构建")
def test_v2_review_workbook_has_three_visible_annotated_sheets() -> None:
    workbook_path = ROOT / "04_collaboration/fi_finalisation_review_v2.xlsx"
    workbook = load_workbook(workbook_path, read_only=False, data_only=False)
    assert workbook.sheetnames == ["1_Underweight_QC", "2_Completeness", "3_FI_QC"]
    assert all(workbook[name].sheet_state == "visible" for name in workbook.sheetnames)
    assert [cell.value for cell in workbook["1_Underweight_QC"][1]] == [
        "Wave", "Valid %", "Underweight %", "Age pattern", "Decision"
    ]
    assert [cell.value for cell in workbook["2_Completeness"][1]] == [
        "Cohort",
        "Wave",
        "Observed N",
        "Primary >=80%, n (%) [QC reference]",
        "Primary >=90%, n (%) [formal]",
        "Common-22 >=80%, n (%) [QC reference]",
        "Common-22 >=90%, n (%) [formal]",
    ]
    assert [cell.value for cell in workbook["3_FI_QC"][1]] == [
        "Cohort", "Wave", "Primary N", "Mean", "SD", "Median", "P99",
        "Correlation with common-22", "Review flag"
    ]
    assert all(workbook[name].freeze_panes == "A2" for name in workbook.sheetnames)
    text_by_sheet = {
        name: " ".join(
            str(cell.value)
            for row in workbook[name].iter_rows()
            for cell in row
            if cell.value is not None
        )
        for name in workbook.sheetnames
    }
    assert "KLoSA only" in text_by_sheet["1_Underweight_QC"]
    assert "90% columns are the formal v2 rule" in text_by_sheet["2_Completeness"]
    assert "Absolute FI levels are not direct country-health comparisons" in text_by_sheet["3_FI_QC"]
    workbook.close()
