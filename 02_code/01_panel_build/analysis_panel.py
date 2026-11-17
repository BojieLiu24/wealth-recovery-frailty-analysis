"""构建 Analysis Panel v1 人工复核候选。"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import analysis_cohort_ledger as ledger_source
import final_eligibility as final_source
from panel_core import RUNNER_VERSION, sha256_file


ROOT = Path(__file__).resolve().parents[2]
MANIFEST_ROOT = ROOT / "00_admin" / "panel_manifests"
ANALYSIS_MANIFEST_ROOT = MANIFEST_ROOT / "analysis_panel"
DESIGN_REGISTER_PATH = ROOT / "00_admin" / "panel_design_register.csv"
REVIEW_PATH = ROOT / "04_collaboration" / "analysis_panel_review_v1.csv"
DICTIONARY_PATH = ANALYSIS_MANIFEST_ROOT / "analysis_panel_data_dictionary.md"
RELEASE_CANDIDATE_PATH = (
    ANALYSIS_MANIFEST_ROOT / "analysis_panel_v1_release_candidate.json"
)
FORMAL_FREEZE_PATH = ANALYSIS_MANIFEST_ROOT / "analysis_panel_v1_freeze.json"

COHORTS = ("HRS", "ELSA", "KLoSA")
RELEASE_VERSION = "v1"
SCHEMA_VERSION = "base_pool_timed_person_wave_29_columns_v1"
BUILDER_VERSION = "analysis_panel_builder_v1.0.0"

PANEL_COLUMNS = (
    "cohort",
    "person_id",
    "tminus1_wave",
    "index_shock_wave",
    "landmark_wave",
    "wave",
    "fi_time_role",
    "interview_year",
    "interview_month",
    "elapsed_months",
    "time_years",
    "fi_primary",
    "fi_common22",
    "wealth_loss_fraction",
    "recovery_fraction_raw",
    "recovery_status",
    "fi_t",
    "age",
    "sex",
    "education",
    "marital_status",
    "household_size",
    "labour_force_status",
    "household_income_rank",
    "current_smoking",
    "wealth_quintile",
    "landmark_weight_raw",
    "primary_complete_case_eligible",
    "weighted_sensitivity_eligible",
)

PANEL_DTYPES = {
    "cohort": "string",
    "person_id": "string",
    "tminus1_wave": "Int16",
    "index_shock_wave": "Int16",
    "landmark_wave": "Int16",
    "wave": "Int16",
    "fi_time_role": "string",
    "interview_year": "Int16",
    "interview_month": "Int8",
    "elapsed_months": "Int16",
    "time_years": "Float64",
    "fi_primary": "Float64",
    "fi_common22": "Float64",
    "wealth_loss_fraction": "Float64",
    "recovery_fraction_raw": "Float64",
    "recovery_status": "string",
    "fi_t": "Float64",
    "age": "Float64",
    "sex": "string",
    "education": "string",
    "marital_status": "string",
    "household_size": "Int16",
    "labour_force_status": "string",
    "household_income_rank": "Float64",
    "current_smoking": "boolean",
    "wealth_quintile": "Int8",
    "landmark_weight_raw": "Float64",
    "primary_complete_case_eligible": "boolean",
    "weighted_sensitivity_eligible": "boolean",
}

NULLABLE_COLUMNS = {
    "fi_common22",
    "fi_t",
    "age",
    "sex",
    "education",
    "marital_status",
    "household_size",
    "labour_force_status",
    "household_income_rank",
    "current_smoking",
    "wealth_quintile",
    "landmark_weight_raw",
}

ADJUSTMENT_COLUMNS = (
    "fi_t",
    "age",
    "sex",
    "education",
    "marital_status",
    "household_size",
    "labour_force_status",
    "household_income_rank",
    "current_smoking",
    "wealth_quintile",
)

FIXED_PERSON_COLUMNS = (
    "tminus1_wave",
    "index_shock_wave",
    "landmark_wave",
    "wealth_loss_fraction",
    "recovery_fraction_raw",
    "recovery_status",
    *ADJUSTMENT_COLUMNS,
    "landmark_weight_raw",
    "primary_complete_case_eligible",
    "weighted_sensitivity_eligible",
)

REQUIRED_DESIGN = {
    "design_id": "wealth_recovery_frailty_analysis_v1",
    "analysis_panel_schema": SCHEMA_VERSION,
    "recovery_landmark_wave": "t+1",
    "baseline_outcome_wave": "t+1",
    "first_followup_outcome_wave": "t+2_onward",
    "landmark_fi_role": "first_repeated_outcome_at_time_zero",
    "time_scale": "individual_elapsed_years_from_interview_year_month",
    "recovery_threshold": "0.50",
    "weight_definition": (
        "cohort_specific_raw_cross_sectional_person_weight_at_exact_tplus1_v1"
    ),
    "primary_weighting_hierarchy": (
        "unweighted_primary_exact_tplus1_weighted_sensitivity"
    ),
    "covariate_missingness": "complete_case_primary_covariate_mi_sensitivity",
}

EXPECTED = {
    "HRS": {
        "shock_experienced_participants": 8204,
        "base_participants": 5222,
        "base_fi_rows": 23846,
        "base_landmark_rows": 5222,
        "base_later_rows": 18624,
        "primary_participants": 5114,
        "primary_fi_rows": 23411,
        "primary_landmark_rows": 5114,
        "primary_later_rows": 18297,
        "weighted_participants": 5097,
        "weighted_fi_rows": 23361,
        "substantial_recovery": 1703,
        "persistent_or_limited_recovery": 3411,
    },
    "ELSA": {
        "shock_experienced_participants": 1284,
        "base_participants": 772,
        "base_fi_rows": 2886,
        "base_landmark_rows": 772,
        "base_later_rows": 2114,
        "primary_participants": 699,
        "primary_fi_rows": 2584,
        "primary_landmark_rows": 699,
        "primary_later_rows": 1885,
        "weighted_participants": 676,
        "weighted_fi_rows": 2508,
        "substantial_recovery": 271,
        "persistent_or_limited_recovery": 428,
    },
    "KLoSA": {
        "shock_experienced_participants": 3125,
        "base_participants": 2507,
        "base_fi_rows": 10781,
        "base_landmark_rows": 2507,
        "base_later_rows": 8274,
        "primary_participants": 2502,
        "primary_fi_rows": 10762,
        "primary_landmark_rows": 2502,
        "primary_later_rows": 8260,
        "weighted_participants": 2502,
        "weighted_fi_rows": 10762,
        "substantial_recovery": 943,
        "persistent_or_limited_recovery": 1559,
    },
}

EXPECTED_SOURCE_WIDE_TIMING_EXCLUSIONS = {
    "HRS": 493,
    "ELSA": 106,
    "KLoSA": 139,
}


def _relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def _json_write(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def _output_path(cohort: str) -> Path:
    return (
        ROOT
        / "01_data"
        / "derived"
        / "panel_components"
        / cohort
        / "12_analysis_panel"
        / f"{cohort.lower()}_analysis_panel_v1.parquet"
    )


def _manifest_path(cohort: str) -> Path:
    return ANALYSIS_MANIFEST_ROOT / f"{cohort.lower()}_analysis_panel_v1.json"


def _release_paths() -> list[Path]:
    return [
        *(_output_path(cohort) for cohort in COHORTS),
        *(_manifest_path(cohort) for cohort in COHORTS),
        REVIEW_PATH,
        DICTIONARY_PATH,
        RELEASE_CANDIDATE_PATH,
    ]


def _verify_release_absent() -> None:
    existing = [str(path) for path in [*_release_paths(), FORMAL_FREEZE_PATH] if path.exists()]
    if existing:
        raise RuntimeError("Analysis Panel v1拒绝覆盖既有资产：" + "; ".join(existing))


def _read_design_register() -> dict[str, dict[str, str]]:
    with DESIGN_REGISTER_PATH.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    by_id = {row["decision_id"]: row for row in rows}
    for decision_id, selected in REQUIRED_DESIGN.items():
        row = by_id.get(decision_id)
        if row is None:
            raise RuntimeError(f"design register缺少决策：{decision_id}")
        if row["status"] != "locked" or row["selected_value"] != selected:
            raise RuntimeError(
                f"design register未锁定：{decision_id}={row.get('selected_value')}"
            )
    return by_id


def protected_snapshot() -> dict[str, Any]:
    """验证Final Eligibility freeze及其冻结输入。"""
    freeze = final_source.verify_final_eligibility_freeze()
    return {
        "final_eligibility_v1": {
            "freeze_path": _relative(final_source.FORMAL_FREEZE_PATH),
            "freeze_sha256": sha256_file(final_source.FORMAL_FREEZE_PATH),
            "source_release_candidate": freeze["source_release_candidate"],
            "cohorts": freeze["cohorts"],
            "review_summary": freeze["review_summary"],
            "data_dictionary": freeze["data_dictionary"],
        },
        "final_eligibility_protected_inputs": freeze["protected_input_snapshot"],
    }


def _load_final_eligibility(cohort: str) -> pd.DataFrame:
    freeze = final_source.verify_final_eligibility_freeze()
    record = freeze["cohorts"][cohort]
    frame = pd.read_parquet(ROOT / record["parquet_path"])
    final_source.validate_final_eligibility(frame, cohort)
    return frame


def _load_panel(name: str, cohort: str, columns: list[str]) -> pd.DataFrame:
    return ledger_source._load_frozen_panel(name, cohort, columns)


def _exact_merge(
    left: pd.DataFrame,
    right: pd.DataFrame,
    *,
    on: list[str],
    validate: str,
    label: str,
) -> pd.DataFrame:
    merged = left.merge(
        right,
        on=on,
        how="left",
        validate=validate,
        indicator=True,
    )
    if not merged["_merge"].eq("both").all():
        raise RuntimeError(f"{label}无法精确连接")
    return merged.drop(columns="_merge")


def _cast(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.loc[:, PANEL_COLUMNS].copy()
    for column, dtype in PANEL_DTYPES.items():
        output[column] = output[column].astype(dtype)
    return output


def _participant_metrics(
    frame: pd.DataFrame, cohort: str, shock_experienced_participants: int
) -> dict[str, Any]:
    primary = frame["primary_complete_case_eligible"]
    weighted = frame["weighted_sensitivity_eligible"]
    landmarks = frame["fi_time_role"].eq("landmark")
    later = frame["fi_time_role"].eq("later")
    primary_people = frame.loc[primary, ["person_id", "recovery_status"]].drop_duplicates()
    weighted_people = frame.loc[weighted, "person_id"].nunique()
    return {
        "shock_experienced_participants": int(shock_experienced_participants),
        "base_participants": int(frame["person_id"].nunique()),
        "base_fi_rows": int(len(frame)),
        "base_landmark_rows": int(landmarks.sum()),
        "base_later_rows": int(later.sum()),
        "primary_participants": int(frame.loc[primary, "person_id"].nunique()),
        "primary_fi_rows": int(primary.sum()),
        "primary_landmark_rows": int((primary & landmarks).sum()),
        "primary_later_rows": int((primary & later).sum()),
        "weighted_participants": int(weighted_people),
        "weighted_fi_rows": int(weighted.sum()),
        "substantial_recovery": int(
            primary_people["recovery_status"].eq("substantial_recovery").sum()
        ),
        "persistent_or_limited_recovery": int(
            primary_people["recovery_status"]
            .eq("persistent_or_limited_recovery")
            .sum()
        ),
    }


def _diagnostics(
    frame: pd.DataFrame,
    timing_audit: dict[str, int],
    source_wide_timing_exclusions: int,
) -> dict[str, Any]:
    later_rows = (
        frame["fi_time_role"]
        .eq("later")
        .groupby(frame["person_id"], sort=False)
        .sum()
    )
    post_landmark_distribution = {
        "1": int(later_rows.eq(1).sum()),
        "2": int(later_rows.eq(2).sum()),
        "3": int(later_rows.eq(3).sum()),
        "4_plus": int(later_rows.ge(4).sum()),
    }
    positive_time = frame.loc[frame["fi_time_role"].eq("later"), "time_years"]
    maximum_followup = frame.groupby("person_id", sort=False)["time_years"].max()
    nonmonotonic_people = int(
        frame.groupby("person_id", sort=False)["elapsed_months"]
        .apply(lambda values: bool((np.diff(values.astype(int)) <= 0).any()))
        .sum()
    )
    time_summary = {
        "minimum_positive_time_years": float(positive_time.min()),
        "median_maximum_followup_years": float(maximum_followup.median()),
        "q1_maximum_followup_years": float(maximum_followup.quantile(0.25)),
        "q3_maximum_followup_years": float(maximum_followup.quantile(0.75)),
        "maximum_observed_followup_years": float(maximum_followup.max()),
        "nonpositive_later_time_rows": int(positive_time.le(0).sum()),
        "nonmonotonic_within_person_time": nonmonotonic_people,
    }
    primary = frame["primary_complete_case_eligible"]
    primary_person = frame.loc[
        primary, ["person_id", "landmark_weight_raw"]
    ].drop_duplicates("person_id")
    weight_status = {
        "positive": int(primary_person["landmark_weight_raw"].gt(0).sum()),
        "zero": int(primary_person["landmark_weight_raw"].eq(0).sum()),
        "missing": int(primary_person["landmark_weight_raw"].isna().sum()),
    }
    common = frame["fi_common22"].notna()
    common_person_any = int(frame.loc[common, "person_id"].nunique())
    common_support = (
        frame.assign(_common=common)
        .groupby("person_id", sort=False)
        .apply(
            lambda group: bool(
                (group["_common"] & group["fi_time_role"].eq("landmark")).any()
                and (group["_common"] & group["fi_time_role"].eq("later")).any()
            ),
            include_groups=False,
        )
    )
    primary_people = set(frame.loc[primary, "person_id"].astype(str))
    common_primary_support = int(
        sum(bool(value) and str(person_id) in primary_people for person_id, value in common_support.items())
    )
    exposure_rows: dict[str, dict[str, int]] = {}
    for status in ("substantial_recovery", "persistent_or_limited_recovery"):
        selected = primary & frame["recovery_status"].eq(status)
        exposure_rows[status] = {
            "participants": int(frame.loc[selected, "person_id"].nunique()),
            "fi_rows": int(selected.sum()),
        }
    return {
        "post_landmark_fi_assessment_distribution_base": post_landmark_distribution,
        "time_summary_base": time_summary,
        "primary_landmark_weight_status": weight_status,
        "common22_available_rows_base": int(common.sum()),
        "common22_any_participants_base": common_person_any,
        "common22_landmark_and_later_participants_primary": common_primary_support,
        "primary_recovery_groups": exposure_rows,
        "timed_selector": {
            **timing_audit,
            "source_wide_fi_rows_excluded_for_unavailable_actual_timing": int(
                source_wide_timing_exclusions
            ),
        },
        "final_eligibility_reconciliation": "PASS",
    }


def validate_analysis_panel(
    frame: pd.DataFrame,
    final_frame: pd.DataFrame,
    cohort: str,
) -> dict[str, Any]:
    if tuple(frame.columns) != PANEL_COLUMNS:
        raise RuntimeError(f"{cohort}: Analysis Panel列顺序不符合29列合同")
    if frame.duplicated(["cohort", "person_id", "wave"]).any():
        raise RuntimeError(f"{cohort}: Analysis Panel主键重复")
    if not frame["cohort"].eq(cohort).all():
        raise RuntimeError(f"{cohort}: cohort字段错误")
    for column in PANEL_COLUMNS:
        if column not in NULLABLE_COLUMNS and frame[column].isna().any():
            raise RuntimeError(f"{cohort}: 非空列出现NA：{column}")
        if str(frame[column].dtype) != PANEL_DTYPES[column]:
            raise RuntimeError(f"{cohort}: dtype不符合合同：{column}")

    if set(frame["fi_time_role"].astype(str).unique()) != {"landmark", "later"}:
        raise RuntimeError(f"{cohort}: fi_time_role枚举错误")
    if set(frame["recovery_status"].astype(str).unique()) - {
        "substantial_recovery",
        "persistent_or_limited_recovery",
    }:
        raise RuntimeError(f"{cohort}: recovery_status枚举错误")
    if not np.isfinite(frame["fi_primary"].astype(float)).all():
        raise RuntimeError(f"{cohort}: fi_primary出现非有限值")
    if not frame["fi_primary"].between(0, 1, inclusive="both").all():
        raise RuntimeError(f"{cohort}: fi_primary超出0到1")
    common = frame["fi_common22"].dropna().astype(float)
    if not np.isfinite(common).all() or not common.between(0, 1, inclusive="both").all():
        raise RuntimeError(f"{cohort}: fi_common22出现非法值")
    if not np.isfinite(frame["wealth_loss_fraction"].astype(float)).all():
        raise RuntimeError(f"{cohort}: wealth_loss_fraction出现非有限值")
    if not frame["wealth_loss_fraction"].ge(0.75).all():
        raise RuntimeError(f"{cohort}: wealth_loss_fraction不符合major shock")
    if not np.isfinite(frame["recovery_fraction_raw"].astype(float)).all():
        raise RuntimeError(f"{cohort}: recovery_fraction_raw出现非有限值")

    base_final = final_frame.loc[final_frame["analysis_panel_base_eligible"]].copy()
    expected_people = base_final["person_id"].astype(str).tolist()
    observed_people = frame.drop_duplicates("person_id")["person_id"].astype(str).tolist()
    if observed_people != expected_people:
        raise RuntimeError(f"{cohort}: Analysis Panel人员集合或顺序与Final Eligibility不一致")
    counts = frame.groupby("person_id", sort=False).size().astype("Int16")
    expected_counts = base_final.set_index("person_id")["primary_fi_model_row_count"]
    if not counts.equals(expected_counts.loc[counts.index]):
        raise RuntimeError(f"{cohort}: 每人FI行数与Final Eligibility不一致")

    for _, group in frame.groupby("person_id", sort=False):
        if int(group["fi_time_role"].eq("landmark").sum()) != 1:
            raise RuntimeError(f"{cohort}: 每人必须恰有一个landmark row")
        landmark = group.loc[group["fi_time_role"].eq("landmark")].iloc[0]
        if int(landmark["wave"]) != int(landmark["landmark_wave"]):
            raise RuntimeError(f"{cohort}: landmark row与landmark_wave不一致")
        if int(landmark["elapsed_months"]) != 0 or float(landmark["time_years"]) != 0:
            raise RuntimeError(f"{cohort}: landmark model time不是0")
        later = group.loc[group["fi_time_role"].eq("later"), "elapsed_months"]
        if later.empty or not later.gt(0).all():
            raise RuntimeError(f"{cohort}: 缺少positive-time later FI")
        if (np.diff(group["elapsed_months"].astype(int).to_numpy()) <= 0).any():
            raise RuntimeError(f"{cohort}: model time未递增")

    landmark_timing = frame.loc[
        frame["fi_time_role"].eq("landmark"),
        ["person_id", "interview_year", "interview_month"],
    ].rename(
        columns={
            "interview_year": "landmark_year",
            "interview_month": "landmark_month",
        }
    )
    timing = frame.merge(landmark_timing, on="person_id", validate="many_to_one")
    recomputed = 12 * (
        timing["interview_year"].astype(int) - timing["landmark_year"].astype(int)
    ) + (
        timing["interview_month"].astype(int) - timing["landmark_month"].astype(int)
    )
    if not frame["elapsed_months"].astype(int).reset_index(drop=True).equals(
        recomputed.astype(int).reset_index(drop=True)
    ):
        raise RuntimeError(f"{cohort}: elapsed_months公式不一致")
    if not np.allclose(
        frame["time_years"].astype(float),
        frame["elapsed_months"].astype(float) / 12,
        rtol=0,
        atol=1e-12,
    ):
        raise RuntimeError(f"{cohort}: time_years公式不一致")

    for column in FIXED_PERSON_COLUMNS:
        if frame.groupby("person_id", sort=False)[column].nunique(dropna=False).gt(1).any():
            raise RuntimeError(f"{cohort}: participant-level字段未固定：{column}")
    primary = frame["primary_complete_case_eligible"]
    if frame.loc[primary, list(ADJUSTMENT_COLUMNS)].isna().any(axis=None):
        raise RuntimeError(f"{cohort}: primary rows出现缺失调整变量")
    if frame["landmark_weight_raw"].dropna().lt(0).any():
        raise RuntimeError(f"{cohort}: landmark weight出现负值")
    expected_weighted = (
        primary
        & frame["landmark_weight_raw"].notna()
        & frame["landmark_weight_raw"].gt(0)
    )
    if not expected_weighted.equals(frame["weighted_sensitivity_eligible"]):
        raise RuntimeError(f"{cohort}: weighted sensitivity flag不一致")

    unique_flags = frame[
        ["person_id", "primary_complete_case_eligible", "weighted_sensitivity_eligible"]
    ].drop_duplicates("person_id")
    expected_flags = base_final[
        ["person_id", "primary_complete_case_eligible", "weighted_sensitivity_eligible"]
    ]
    pd.testing.assert_frame_equal(
        unique_flags.reset_index(drop=True),
        expected_flags.reset_index(drop=True),
        check_dtype=False,
    )
    metrics = _participant_metrics(frame, cohort, len(final_frame))
    if metrics != EXPECTED[cohort]:
        raise RuntimeError(f"{cohort}: Analysis Panel汇总不符合预期：{metrics}")
    return {"metrics": metrics}


def build_analysis_panel(
    cohort: str,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    final_frame = _load_final_eligibility(cohort)
    final_frame = final_frame.copy()
    final_frame["_person_order"] = np.arange(len(final_frame), dtype=np.int64)
    base = final_frame.loc[final_frame["analysis_panel_base_eligible"]].copy()

    fi = _load_panel(
        "fi_panel_v2",
        cohort,
        ["cohort", "person_id", "wave", "fi_primary", "fi_common22"],
    )
    spine = _load_panel(
        "spine_v2",
        cohort,
        [
            "cohort",
            "person_id",
            "wave",
            "wave_order",
            "observed_in_wave",
            "interview_year",
            "interview_month",
        ],
    )
    timed, timing, timing_audit = final_source.select_timed_fi_rows(
        base[["cohort", "person_id", "landmark_wave"]], fi, spine
    )
    timing_check = base.merge(
        timing[
            [
                "cohort",
                "person_id",
                "primary_fi_model_row_count",
                "primary_later_fi_model_row_count",
            ]
        ],
        on=["cohort", "person_id"],
        suffixes=("_final", "_selector"),
        validate="one_to_one",
    )
    for name in ("primary_fi_model_row_count", "primary_later_fi_model_row_count"):
        if not timing_check[f"{name}_final"].equals(timing_check[f"{name}_selector"]):
            raise RuntimeError(f"{cohort}: shared timed-FI selector未复现Final Eligibility：{name}")

    row_values = _exact_merge(
        timed,
        fi,
        on=["cohort", "person_id", "wave"],
        validate="one_to_one",
        label=f"{cohort}: timed FI values",
    )
    row_values["fi_time_role"] = pd.Series(
        np.where(row_values["is_landmark"], "landmark", "later"),
        index=row_values.index,
        dtype="string",
    )
    row_values["time_years"] = row_values["elapsed_months"].astype("Float64") / 12

    exposure = _load_panel(
        "exposure_panel_v1",
        cohort,
        [
            "cohort",
            "person_id",
            "candidate_shock_wave",
            "wealth_loss_fraction",
            "recovery_fraction_raw",
            "recovery_status",
            "is_index_shock",
        ],
    )
    exposure = exposure.loc[exposure["is_index_shock"].eq(True)].rename(
        columns={"candidate_shock_wave": "index_shock_wave"}
    )
    exposure = exposure.drop(columns="is_index_shock")

    covariates = _load_panel(
        "covariate_panel_v1",
        cohort,
        [
            "cohort",
            "person_id",
            "wave",
            "age",
            "sex",
            "education",
            "marital_status",
            "household_size",
            "labour_force_status",
            "household_income_rank",
            "current_smoking",
            "wealth_quintile",
        ],
    )
    cov_t = covariates.drop(columns="wealth_quintile").rename(
        columns={"wave": "index_shock_wave"}
    )
    cov_tminus1 = covariates[
        ["cohort", "person_id", "wave", "wealth_quintile"]
    ].rename(columns={"wave": "tminus1_wave"})
    fi_t = fi[["cohort", "person_id", "wave", "fi_primary"]].rename(
        columns={"wave": "index_shock_wave", "fi_primary": "fi_t"}
    )
    weight = _load_panel(
        "weight_design_panel_v1",
        cohort,
        ["cohort", "person_id", "wave", "person_weight_raw"],
    ).rename(
        columns={"wave": "landmark_wave", "person_weight_raw": "landmark_weight_raw"}
    )

    constants = base[
        [
            "cohort",
            "person_id",
            "tminus1_wave",
            "index_shock_wave",
            "landmark_wave",
            "primary_complete_case_eligible",
            "weighted_sensitivity_eligible",
            "_person_order",
        ]
    ]
    constants = _exact_merge(
        constants,
        exposure,
        on=["cohort", "person_id", "index_shock_wave"],
        validate="one_to_one",
        label=f"{cohort}: index exposure",
    )
    constants = _exact_merge(
        constants,
        fi_t,
        on=["cohort", "person_id", "index_shock_wave"],
        validate="one_to_one",
        label=f"{cohort}: FI_t",
    )
    constants = _exact_merge(
        constants,
        cov_t,
        on=["cohort", "person_id", "index_shock_wave"],
        validate="one_to_one",
        label=f"{cohort}: t covariates",
    )
    constants = _exact_merge(
        constants,
        cov_tminus1,
        on=["cohort", "person_id", "tminus1_wave"],
        validate="one_to_one",
        label=f"{cohort}: tminus1 wealth quintile",
    )
    constants = _exact_merge(
        constants,
        weight,
        on=["cohort", "person_id", "landmark_wave"],
        validate="one_to_one",
        label=f"{cohort}: landmark weight",
    )
    if not constants["recovery_status"].equals(
        base["recovery_status"].reset_index(drop=True)
    ):
        raise RuntimeError(f"{cohort}: Exposure与Final Eligibility recovery_status不一致")

    working = row_values.merge(
        constants,
        on=["cohort", "person_id", "landmark_wave"],
        how="left",
        validate="many_to_one",
        indicator=True,
    )
    if not working["_merge"].eq("both").all():
        raise RuntimeError(f"{cohort}: timed rows无法连接participant constants")
    working = working.drop(columns="_merge").sort_values(
        ["_person_order", "wave_order"], kind="mergesort"
    )
    output = _cast(working)
    validation = validate_analysis_panel(output, final_frame, cohort)
    final_freeze = final_source.verify_final_eligibility_freeze()
    source_wide_timing_exclusions = final_freeze["cohorts"][cohort][
        "non_gating_diagnostics"
    ]["fi_rows_excluded_for_unavailable_actual_timing"]
    if source_wide_timing_exclusions != EXPECTED_SOURCE_WIDE_TIMING_EXCLUSIONS[cohort]:
        raise RuntimeError(f"{cohort}: Final Eligibility source-wide时间审计人数漂移")
    diagnostic = _diagnostics(
        output,
        timing_audit,
        source_wide_timing_exclusions,
    )
    if diagnostic["common22_landmark_and_later_participants_primary"] != final_source.EXPECTED[
        cohort
    ]["common22_support"]:
        raise RuntimeError(f"{cohort}: Common-22 primary支持未复现Final Eligibility")
    return output, validation, diagnostic


def _review_rows(
    validations: dict[str, dict[str, Any]],
    diagnostics: dict[str, dict[str, Any]],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    def add(
        section: str,
        cohort: str,
        sample_scope: str,
        metric: str,
        category: str,
        n: int | float,
        expected_n: int | float | None = None,
        status: str | None = None,
    ) -> None:
        if status is None:
            status = "PASS" if expected_n is not None else "INFO"
        rows.append(
            {
                "section": section,
                "cohort": cohort,
                "sample_scope": sample_scope,
                "metric": metric,
                "category": category,
                "n": n,
                "expected_n": expected_n,
                "reconciliation_status": status,
            }
        )

    for cohort in COHORTS:
        metrics = validations[cohort]["metrics"]
        diagnostic = diagnostics[cohort]
        for scope, participant_key, row_key in (
            ("base", "base_participants", "base_fi_rows"),
            ("primary", "primary_participants", "primary_fi_rows"),
            ("weighted", "weighted_participants", "weighted_fi_rows"),
        ):
            add(
                "sample_and_rows",
                cohort,
                scope,
                "participants",
                "all",
                metrics[participant_key],
                EXPECTED[cohort][participant_key],
            )
            add(
                "sample_and_rows",
                cohort,
                scope,
                "fi_rows",
                "all",
                metrics[row_key],
                EXPECTED[cohort][row_key],
            )
        for scope, landmark_key, later_key in (
            ("base", "base_landmark_rows", "base_later_rows"),
            ("primary", "primary_landmark_rows", "primary_later_rows"),
        ):
            add(
                "time_structure",
                cohort,
                scope,
                "fi_rows",
                "landmark",
                metrics[landmark_key],
                EXPECTED[cohort][landmark_key],
            )
            add(
                "time_structure",
                cohort,
                scope,
                "fi_rows",
                "later",
                metrics[later_key],
                EXPECTED[cohort][later_key],
            )
        for category, n in diagnostic[
            "post_landmark_fi_assessment_distribution_base"
        ].items():
            add(
                "followup_structure",
                cohort,
                "base",
                "post_landmark_fi_assessments_per_participant",
                category,
                n,
            )
        for status, values in diagnostic["primary_recovery_groups"].items():
            add(
                "primary_recovery_groups",
                cohort,
                "primary",
                "participants",
                status,
                values["participants"],
                EXPECTED[cohort][status],
            )
            add(
                "primary_recovery_groups",
                cohort,
                "primary",
                "fi_rows",
                status,
                values["fi_rows"],
            )
        add(
            "timing",
            cohort,
            "pre_panel_timing_audit",
            "source_fi_rows_excluded_for_unavailable_actual_timing",
            "all",
            diagnostic["timed_selector"][
                "source_wide_fi_rows_excluded_for_unavailable_actual_timing"
            ],
            EXPECTED_SOURCE_WIDE_TIMING_EXCLUSIONS[cohort],
        )
        add(
            "timing",
            cohort,
            "analysis_panel_base",
            "additional_fi_rows_excluded_during_panel_materialization",
            "all",
            diagnostic["timed_selector"][
                "fi_rows_excluded_for_unavailable_actual_timing"
            ],
            0,
        )
        for metric, value in diagnostic["time_summary_base"].items():
            if metric in {
                "nonpositive_later_time_rows",
                "nonmonotonic_within_person_time",
            }:
                add("time_summary", cohort, "base", metric, "count", value, 0)
            else:
                add("time_summary", cohort, "base", metric, "years", value)
        for metric in (
            "common22_available_rows_base",
            "common22_any_participants_base",
            "common22_landmark_and_later_participants_primary",
        ):
            add("common22", cohort, "base_or_primary", metric, "all", diagnostic[metric])
        for category, n in diagnostic["primary_landmark_weight_status"].items():
            add("landmark_weight", cohort, "primary", "participants", category, n)
        add(
            "final_eligibility_reconciliation",
            cohort,
            "all",
            "reconciliation_pass",
            "all",
            1,
            1,
        )
    return pd.DataFrame(
        rows,
        columns=[
            "section",
            "cohort",
            "sample_scope",
            "metric",
            "category",
            "n",
            "expected_n",
            "reconciliation_status",
        ],
    )


def _dictionary_text() -> str:
    return """# Analysis Panel v1 data dictionary

每行是一名`analysis_panel_base_eligible`参与者在landmark或其后某次具有有效primary FI及实际访谈时间的模型记录。物理Panel本身即base pool，因此不保存恒为TRUE的base eligibility字段。

| Column | Type | Nullable | Definition |
| --- | --- | --- | --- |
| `cohort` | string | no | `HRS`、`ELSA`或`KLoSA`。 |
| `person_id` | string | no | 队列内参与者标识。 |
| `tminus1_wave` | Int16 | no | Index shock前一波。 |
| `index_shock_wave` | Int16 | no | 窗口内首次qualifying major wealth shock波次。 |
| `landmark_wave` | Int16 | no | Exact `t+1` landmark波次。 |
| `wave` | Int16 | no | 当前FI模型记录波次。 |
| `fi_time_role` | string | no | `landmark`或`later`。 |
| `interview_year` | Int16 | no | 当前记录实际访谈年份。 |
| `interview_month` | Int8 | no | 当前记录实际访谈月份。 |
| `elapsed_months` | Int16 | no | 当前访谈年月相对landmark的精确月份差。 |
| `time_years` | Float64 | no | `elapsed_months / 12`，模型时间。 |
| `fi_primary` | Float64 | no | 当前波队列特异primary 30-item FI。 |
| `fi_common22` | Float64 | yes | 当前波common-22 measurement-sensitivity FI；不控制primary资格。 |
| `wealth_loss_fraction` | Float64 | no | Index shock的未截断财富损失比例。 |
| `recovery_fraction_raw` | Float64 | no | Landmark未截断、未winsorise的恢复比例。 |
| `recovery_status` | string | no | `substantial_recovery`或`persistent_or_limited_recovery`。 |
| `fi_t` | Float64 | yes | Index-shock interview的primary FI调整变量。 |
| `age` | Float64 | yes | Index-shock interview年龄。 |
| `sex` | string | yes | 来源支持的固定个人属性。 |
| `education` | string | yes | 来源支持的固定个人属性。 |
| `marital_status` | string | yes | Index-shock interview四分类婚姻状态。 |
| `household_size` | Int16 | yes | Index-shock interview家庭人数。 |
| `labour_force_status` | string | yes | Index-shock interview四分类就业状态。 |
| `household_income_rank` | Float64 | yes | Index-shock interview队列内同波加权收入midrank。 |
| `current_smoking` | boolean | yes | Index-shock interview当前吸烟。 |
| `wealth_quintile` | Int8 | yes | `t-1`未加权同波财富位次五分组。 |
| `landmark_weight_raw` | Float64 | yes | Exact `t+1`来源横断面个人权重；zero保留且不缩放。 |
| `primary_complete_case_eligible` | boolean | no | Participant-level primary complete-case sample flag，重复到该人的全部longitudinal rows。 |
| `weighted_sensitivity_eligible` | boolean | no | Participant-level positive-landmark-weight sensitivity flag，重复到该人的全部longitudinal rows。 |

`elapsed_months = 12*(interview_year_current-interview_year_landmark) + (interview_month_current-interview_month_landmark)`。不使用nominal spacing、插值、假定两年间隔或跨波回填。

Primary、weighted、MI、Common-22及25%/100% recovery analyses必须从本Panel动态筛选或派生；不得另存独立Analysis Panel。Scaled weights、imputed values、model matrices和预测值不属于本资产。
"""


def _schema_contract(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return [
        {
            "name": column,
            "dtype": str(frame[column].dtype),
            "nullable": column in NULLABLE_COLUMNS,
        }
        for column in PANEL_COLUMNS
    ]


def _git_provenance() -> dict[str, Any]:
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, check=True
    ).stdout.decode("ascii").strip()
    diff = subprocess.run(
        ["git", "diff", "--binary", "HEAD"],
        cwd=ROOT,
        capture_output=True,
        check=True,
    ).stdout
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=ROOT,
        capture_output=True,
        check=True,
    ).stdout.decode("utf-8", errors="replace").splitlines()
    return {
        "git_commit": head,
        "git_dirty": bool(status),
        "git_diff_sha256": hashlib.sha256(diff).hexdigest(),
        "git_status_entry_count": len(status),
    }


def _manifest(
    cohort: str,
    panel: pd.DataFrame,
    staged_output: Path,
    validation: dict[str, Any],
    diagnostics: dict[str, Any],
    protected: dict[str, Any],
    design_sha: str,
    provenance: dict[str, Any],
) -> dict[str, Any]:
    return {
        "manifest_schema_version": "analysis_panel_manifest_v1",
        "release_status": "READY_FOR_REVIEW",
        "release_version": RELEASE_VERSION,
        "generated_utc": datetime.now(UTC).isoformat(),
        "cohort": cohort,
        "builder_version": BUILDER_VERSION,
        "builder_sha256": sha256_file(Path(__file__)),
        "runner_version": RUNNER_VERSION,
        **provenance,
        "design_register": {
            "path": _relative(DESIGN_REGISTER_PATH),
            "sha256": design_sha,
            "required_locked_decisions": REQUIRED_DESIGN,
        },
        "row_unit": "one retained landmark or later timed primary-FI observation",
        "common_key": ["cohort", "person_id", "wave"],
        "physical_columns": list(PANEL_COLUMNS),
        "schema": _schema_contract(panel),
        "row_count": len(panel),
        "participant_count": int(panel["person_id"].nunique()),
        "unique_key": not panel.duplicated(["cohort", "person_id", "wave"]).any(),
        **validation,
        "diagnostics": diagnostics,
        "output": {
            "path": _relative(_output_path(cohort)),
            "sha256": sha256_file(staged_output),
            "compression": "zstd",
        },
        "protected_input_snapshot": protected,
        "invariants": {
            "physical_rows_are_final_eligibility_base_pool_only": True,
            "shared_timed_fi_selector_reused": True,
            "one_landmark_and_at_least_one_later_row_per_participant": True,
            "participant_level_fields_fixed_within_person": True,
            "primary_and_weighted_flags_repeated_from_final_eligibility": True,
            "no_scaled_or_imputed_values": True,
            "no_variant_analysis_panels_created": True,
            "models_not_run": True,
        },
    }


def _publish_transaction(staged_to_final: list[tuple[Path, Path]]) -> None:
    published: list[Path] = []
    try:
        for staged, final in staged_to_final:
            if final.exists():
                raise RuntimeError(f"拒绝覆盖既有资产：{final}")
            final.parent.mkdir(parents=True, exist_ok=True)
            os.replace(staged, final)
            published.append(final)
    except Exception:
        for path in reversed(published):
            path.unlink(missing_ok=True)
        raise


def _replace_unfrozen_metadata_transaction(
    staged_to_final: list[tuple[Path, Path]], backup_root: Path
) -> None:
    """只替换未冻结候选的汇总元数据；失败时恢复原文件。"""
    if FORMAL_FREEZE_PATH.exists():
        raise RuntimeError("Analysis Panel v1已冻结，拒绝刷新候选元数据")
    backup_root.mkdir(parents=True, exist_ok=False)
    replaced: list[tuple[Path, Path]] = []
    try:
        for staged, final in staged_to_final:
            if not final.is_file():
                raise RuntimeError(f"缺少待替换的候选元数据：{final}")
            backup = backup_root / final.name
            os.replace(final, backup)
            os.replace(staged, final)
            replaced.append((final, backup))
    except Exception:
        for final, backup in reversed(replaced):
            final.unlink(missing_ok=True)
            os.replace(backup, final)
        raise


def run_analysis_panel_release(cohorts: list[str]) -> dict[str, Any]:
    """一次性生成三队列Analysis Panel v1复核候选。"""
    if tuple(cohorts) != COHORTS:
        raise RuntimeError("analysis-panel v1必须使用--cohort all共同构建")
    _verify_release_absent()
    _read_design_register()
    design_sha = sha256_file(DESIGN_REGISTER_PATH)
    protected_before = protected_snapshot()
    provenance = _git_provenance()

    tmp_root = ROOT / "tmp"
    tmp_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="analysis_panel_", dir=tmp_root) as name:
        staging = Path(name)
        panels: dict[str, pd.DataFrame] = {}
        validations: dict[str, dict[str, Any]] = {}
        diagnostics: dict[str, dict[str, Any]] = {}
        staged_outputs: dict[str, Path] = {}
        staged_manifests: dict[str, Path] = {}
        for cohort in COHORTS:
            panel, validation, diagnostic = build_analysis_panel(cohort)
            panels[cohort] = panel
            validations[cohort] = validation
            diagnostics[cohort] = diagnostic
            output = staging / f"{cohort.lower()}_analysis_panel_v1.parquet"
            panel.to_parquet(output, index=False, compression="zstd")
            staged_outputs[cohort] = output
            manifest_path = staging / f"{cohort.lower()}_analysis_panel_v1.json"
            _json_write(
                manifest_path,
                _manifest(
                    cohort,
                    panel,
                    output,
                    validation,
                    diagnostic,
                    protected_before,
                    design_sha,
                    provenance,
                ),
            )
            staged_manifests[cohort] = manifest_path

        review = _review_rows(validations, diagnostics)
        staged_review = staging / REVIEW_PATH.name
        review.to_csv(staged_review, index=False, encoding="utf-8-sig")
        staged_dictionary = staging / DICTIONARY_PATH.name
        staged_dictionary.write_text(_dictionary_text(), encoding="utf-8")
        release = {
            "release_candidate_schema_version": "analysis_panel_release_candidate_v1",
            "release_status": "READY_FOR_REVIEW",
            "candidate_label": "ANALYSIS PANEL V1 READY FOR REVIEW",
            "formal_freeze_created": False,
            "release_version": RELEASE_VERSION,
            "schema_version": SCHEMA_VERSION,
            "generated_utc": datetime.now(UTC).isoformat(),
            "builder_version": BUILDER_VERSION,
            "builder_sha256": sha256_file(Path(__file__)),
            "runner_version": RUNNER_VERSION,
            **provenance,
            "design_register": {
                "path": _relative(DESIGN_REGISTER_PATH),
                "sha256": design_sha,
                "required_locked_decisions": REQUIRED_DESIGN,
            },
            "row_unit": "one retained landmark or later timed primary-FI observation",
            "common_key": ["cohort", "person_id", "wave"],
            "physical_columns": list(PANEL_COLUMNS),
            "cohorts": {
                cohort: {
                    "parquet_path": _relative(_output_path(cohort)),
                    "parquet_sha256": sha256_file(staged_outputs[cohort]),
                    "manifest_path": _relative(_manifest_path(cohort)),
                    "manifest_sha256": sha256_file(staged_manifests[cohort]),
                    **validations[cohort],
                    "diagnostics": diagnostics[cohort],
                }
                for cohort in COHORTS
            },
            "review_summary": {
                "path": _relative(REVIEW_PATH),
                "sha256": sha256_file(staged_review),
                "contains_participant_ids": False,
            },
            "data_dictionary": {
                "path": _relative(DICTIONARY_PATH),
                "sha256": sha256_file(staged_dictionary),
            },
            "protected_input_snapshot": protected_before,
            "scope_boundaries": {
                "final_eligibility_reexecuted": False,
                "analysis_panel_base_pool_materialized": True,
                "separate_analysis_panel_variants_created": False,
                "scaled_weights_or_imputations_created": False,
                "model_run": False,
                "figure_or_manuscript_modified": False,
            },
            "next_step": (
                "Author review; an unchanged candidate may be frozen without rebuilding. "
                "Model contract and model execution remain separately approval-gated."
            ),
        }
        staged_release = staging / RELEASE_CANDIDATE_PATH.name
        _json_write(staged_release, release)

        if protected_snapshot() != protected_before:
            raise RuntimeError("Analysis Panel构建期间冻结输入发生变化")
        if sha256_file(DESIGN_REGISTER_PATH) != design_sha:
            raise RuntimeError("Analysis Panel构建期间design register发生变化")
        publish = [
            *((staged_outputs[cohort], _output_path(cohort)) for cohort in COHORTS),
            *((staged_manifests[cohort], _manifest_path(cohort)) for cohort in COHORTS),
            (staged_review, REVIEW_PATH),
            (staged_dictionary, DICTIONARY_PATH),
            (staged_release, RELEASE_CANDIDATE_PATH),
        ]
        _publish_transaction(publish)

    return verify_analysis_panel_candidate()


def refresh_analysis_panel_review_candidate() -> dict[str, Any]:
    """不重写Parquet，只刷新作者要求的review、manifest和共同候选。"""
    if FORMAL_FREEZE_PATH.exists():
        raise RuntimeError("Analysis Panel v1已冻结，拒绝刷新候选")
    if not RELEASE_CANDIDATE_PATH.is_file():
        raise RuntimeError("缺少可刷新的Analysis Panel候选")
    old_sha = sha256_file(RELEASE_CANDIDATE_PATH)
    old = json.loads(RELEASE_CANDIDATE_PATH.read_text(encoding="utf-8"))
    if old.get("release_status") != "READY_FOR_REVIEW":
        raise RuntimeError("Analysis Panel候选状态错误")
    if old.get("physical_columns") != list(PANEL_COLUMNS):
        raise RuntimeError("Analysis Panel候选不是29列合同")
    _read_design_register()
    design_sha = sha256_file(DESIGN_REGISTER_PATH)
    if old["design_register"]["sha256"] != design_sha:
        raise RuntimeError("Analysis Panel候选design register哈希漂移")
    protected = protected_snapshot()
    if old["protected_input_snapshot"] != protected:
        raise RuntimeError("Analysis Panel候选冻结输入漂移")

    provenance = _git_provenance()
    validations: dict[str, dict[str, Any]] = {}
    diagnostics: dict[str, dict[str, Any]] = {}
    panels: dict[str, pd.DataFrame] = {}
    for cohort in COHORTS:
        record = old["cohorts"][cohort]
        output = ROOT / record["parquet_path"]
        if sha256_file(output) != record["parquet_sha256"]:
            raise RuntimeError(f"{cohort}: review刷新前Parquet哈希漂移")
        panel = pd.read_parquet(output)
        validation = validate_analysis_panel(
            panel,
            _load_final_eligibility(cohort),
            cohort,
        )
        old_timing = record["diagnostics"]["timed_selector"]
        timing_audit = {
            "valid_fi_rows_from_landmark": int(
                old_timing["valid_fi_rows_from_landmark"]
            ),
            "timed_fi_rows_retained": int(old_timing["timed_fi_rows_retained"]),
            "fi_rows_excluded_for_unavailable_actual_timing": int(
                old_timing["fi_rows_excluded_for_unavailable_actual_timing"]
            ),
        }
        if timing_audit != {
            "valid_fi_rows_from_landmark": len(panel),
            "timed_fi_rows_retained": len(panel),
            "fi_rows_excluded_for_unavailable_actual_timing": 0,
        }:
            raise RuntimeError(f"{cohort}: base-pool timed selector审计不允许刷新")
        source_exclusions = protected["final_eligibility_v1"]["cohorts"][cohort][
            "non_gating_diagnostics"
        ]["fi_rows_excluded_for_unavailable_actual_timing"]
        diagnostic = _diagnostics(panel, timing_audit, source_exclusions)
        panels[cohort] = panel
        validations[cohort] = validation
        diagnostics[cohort] = diagnostic

    tmp_root = ROOT / "tmp"
    tmp_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="analysis_panel_review_refresh_", dir=tmp_root) as name:
        staging = Path(name)
        staged_manifests: dict[str, Path] = {}
        for cohort in COHORTS:
            path = staging / _manifest_path(cohort).name
            _json_write(
                path,
                _manifest(
                    cohort,
                    panels[cohort],
                    _output_path(cohort),
                    validations[cohort],
                    diagnostics[cohort],
                    protected,
                    design_sha,
                    provenance,
                ),
            )
            staged_manifests[cohort] = path

        staged_review = staging / REVIEW_PATH.name
        _review_rows(validations, diagnostics).to_csv(
            staged_review,
            index=False,
            encoding="utf-8-sig",
        )
        staged_dictionary = staging / DICTIONARY_PATH.name
        staged_dictionary.write_text(_dictionary_text(), encoding="utf-8")
        release = {
            **old,
            "generated_utc": datetime.now(UTC).isoformat(),
            "builder_version": BUILDER_VERSION,
            "builder_sha256": sha256_file(Path(__file__)),
            "runner_version": RUNNER_VERSION,
            **provenance,
            "cohorts": {
                cohort: {
                    "parquet_path": _relative(_output_path(cohort)),
                    "parquet_sha256": sha256_file(_output_path(cohort)),
                    "manifest_path": _relative(_manifest_path(cohort)),
                    "manifest_sha256": sha256_file(staged_manifests[cohort]),
                    **validations[cohort],
                    "diagnostics": diagnostics[cohort],
                }
                for cohort in COHORTS
            },
            "review_summary": {
                "path": _relative(REVIEW_PATH),
                "sha256": sha256_file(staged_review),
                "contains_participant_ids": False,
            },
            "data_dictionary": {
                "path": _relative(DICTIONARY_PATH),
                "sha256": sha256_file(staged_dictionary),
            },
            "candidate_refresh": {
                "kind": "review_and_manifest_only",
                "prior_candidate_sha256": old_sha,
                "parquet_rebuilt": False,
                "scientific_definition_changed": False,
                "schema_changed": False,
            },
        }
        staged_release = staging / RELEASE_CANDIDATE_PATH.name
        _json_write(staged_release, release)
        if protected_snapshot() != protected:
            raise RuntimeError("Analysis Panel review刷新期间冻结输入发生变化")
        if sha256_file(DESIGN_REGISTER_PATH) != design_sha:
            raise RuntimeError("Analysis Panel review刷新期间design register发生变化")
        replacements = [
            *((staged_manifests[c], _manifest_path(c)) for c in COHORTS),
            (staged_review, REVIEW_PATH),
            (staged_dictionary, DICTIONARY_PATH),
            (staged_release, RELEASE_CANDIDATE_PATH),
        ]
        _replace_unfrozen_metadata_transaction(
            replacements,
            staging / "backup",
        )
    return verify_analysis_panel_candidate()


def verify_analysis_panel_candidate(
    *,
    require_current_design_hash: bool = True,
    require_current_builder_hash: bool = True,
) -> dict[str, Any]:
    if not RELEASE_CANDIDATE_PATH.is_file():
        raise RuntimeError("缺少Analysis Panel v1 release candidate")
    release = json.loads(RELEASE_CANDIDATE_PATH.read_text(encoding="utf-8"))
    if release.get("release_status") != "READY_FOR_REVIEW":
        raise RuntimeError("Analysis Panel候选状态错误")
    if release.get("physical_columns") != list(PANEL_COLUMNS):
        raise RuntimeError("Analysis Panel候选schema漂移")
    if (
        require_current_builder_hash
        and release.get("builder_sha256") != sha256_file(Path(__file__))
    ):
        raise RuntimeError("Analysis Panel候选builder哈希漂移")
    _read_design_register()
    if (
        require_current_design_hash
        and release["design_register"]["sha256"] != sha256_file(DESIGN_REGISTER_PATH)
    ):
        raise RuntimeError("Analysis Panel候选design register哈希漂移")
    if release["protected_input_snapshot"] != protected_snapshot():
        raise RuntimeError("Analysis Panel候选冻结输入漂移")

    validations: dict[str, dict[str, Any]] = {}
    diagnostics: dict[str, dict[str, Any]] = {}
    for cohort in COHORTS:
        record = release["cohorts"][cohort]
        if sha256_file(ROOT / record["parquet_path"]) != record["parquet_sha256"]:
            raise RuntimeError(f"{cohort}: Analysis Panel Parquet哈希漂移")
        if sha256_file(ROOT / record["manifest_path"]) != record["manifest_sha256"]:
            raise RuntimeError(f"{cohort}: Analysis Panel manifest哈希漂移")
        frame = pd.read_parquet(ROOT / record["parquet_path"])
        validation = validate_analysis_panel(frame, _load_final_eligibility(cohort), cohort)
        if validation != {"metrics": record["metrics"]}:
            raise RuntimeError(f"{cohort}: Analysis Panel内容与候选不一致")
        validations[cohort] = validation
        diagnostics[cohort] = record["diagnostics"]
    observed = pd.read_csv(ROOT / release["review_summary"]["path"], encoding="utf-8-sig")
    if sha256_file(ROOT / release["review_summary"]["path"]) != release[
        "review_summary"
    ]["sha256"]:
        raise RuntimeError("Analysis Panel review CSV哈希漂移")
    expected = _review_rows(validations, diagnostics)
    pd.testing.assert_frame_equal(observed, expected, check_dtype=False)
    if "person_id" in observed.columns:
        raise RuntimeError("Analysis Panel review CSV不得包含person_id")
    if sha256_file(ROOT / release["data_dictionary"]["path"]) != release[
        "data_dictionary"
    ]["sha256"]:
        raise RuntimeError("Analysis Panel数据字典哈希漂移")
    return release


def freeze_analysis_panel_v1(approved_candidate_sha256: str) -> dict[str, Any]:
    """按作者批准候选哈希冻结；不重建任何Panel。"""
    if FORMAL_FREEZE_PATH.exists():
        raise RuntimeError("Analysis Panel v1已冻结，拒绝覆盖")
    release = verify_analysis_panel_candidate()
    current_sha = sha256_file(RELEASE_CANDIDATE_PATH)
    if current_sha != approved_candidate_sha256:
        raise RuntimeError("Analysis Panel候选哈希与作者批准值不一致")
    freeze = {
        "freeze_record_schema_version": "analysis_panel_freeze_record_v1",
        "freeze_status": "FROZEN",
        "analysis_panel_status": "MODEL_READY_LONGITUDINAL_PANEL",
        "release_version": RELEASE_VERSION,
        "schema_version": SCHEMA_VERSION,
        "generated_utc": datetime.now(UTC).isoformat(),
        "source_release_candidate": {
            "path": _relative(RELEASE_CANDIDATE_PATH),
            "sha256": current_sha,
        },
        "cohorts": release["cohorts"],
        "review_summary": release["review_summary"],
        "data_dictionary": release["data_dictionary"],
        "design_register": release["design_register"],
        "protected_input_snapshot": release["protected_input_snapshot"],
        "release_integrity": {
            "parquet_rebuilt_for_freeze": False,
            "cohort_manifest_rebuilt_for_freeze": False,
            "review_csv_rebuilt_for_freeze": False,
        },
        "analysis_boundaries": {
            "single_base_pool_panel_per_cohort": True,
            "primary_and_weighted_branches_are_dynamic_filters": True,
            "mi_values_not_created": True,
            "scaled_weights_not_created": True,
            "models_not_run": True,
        },
        "overwrite_policy": "Analysis Panel v1 is immutable; changes require a new version.",
        "next_stage": "model-contract",
    }
    ANALYSIS_MANIFEST_ROOT.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="analysis_panel_freeze_", dir=ROOT / "tmp") as name:
        staged = Path(name) / FORMAL_FREEZE_PATH.name
        _json_write(staged, freeze)
        os.replace(staged, FORMAL_FREEZE_PATH)
    return verify_analysis_panel_freeze()


def verify_analysis_panel_freeze() -> dict[str, Any]:
    if not FORMAL_FREEZE_PATH.is_file():
        raise RuntimeError("Analysis Panel v1 freeze不存在")
    freeze = json.loads(FORMAL_FREEZE_PATH.read_text(encoding="utf-8"))
    if freeze.get("freeze_status") != "FROZEN":
        raise RuntimeError("Analysis Panel freeze_status错误")
    if freeze.get("analysis_panel_status") != "MODEL_READY_LONGITUDINAL_PANEL":
        raise RuntimeError("Analysis Panel freeze语义错误")
    if sha256_file(RELEASE_CANDIDATE_PATH) != freeze["source_release_candidate"]["sha256"]:
        raise RuntimeError("Frozen Analysis Panel候选哈希漂移")
    release = verify_analysis_panel_candidate(
        require_current_design_hash=False,
        require_current_builder_hash=False,
    )
    if freeze["cohorts"] != release["cohorts"]:
        raise RuntimeError("Analysis Panel freeze与候选不一致")
    return freeze


__all__ = [
    "ADJUSTMENT_COLUMNS",
    "COHORTS",
    "DICTIONARY_PATH",
    "EXPECTED",
    "FORMAL_FREEZE_PATH",
    "PANEL_COLUMNS",
    "PANEL_DTYPES",
    "RELEASE_CANDIDATE_PATH",
    "REVIEW_PATH",
    "build_analysis_panel",
    "freeze_analysis_panel_v1",
    "protected_snapshot",
    "refresh_analysis_panel_review_candidate",
    "run_analysis_panel_release",
    "validate_analysis_panel",
    "verify_analysis_panel_candidate",
    "verify_analysis_panel_freeze",
]
