"""构建 Final Eligibility v1 人工复核候选。"""

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
from panel_core import RUNNER_VERSION, sha256_file


ROOT = Path(__file__).resolve().parents[2]
MANIFEST_ROOT = ROOT / "00_admin" / "panel_manifests"
FINAL_MANIFEST_ROOT = MANIFEST_ROOT / "final_eligibility"
DESIGN_REGISTER_PATH = ROOT / "00_admin" / "panel_design_register.csv"
REVIEW_PATH = ROOT / "04_collaboration" / "final_eligibility_review_v1.csv"
DICTIONARY_PATH = FINAL_MANIFEST_ROOT / "final_eligibility_data_dictionary.md"
RELEASE_CANDIDATE_PATH = (
    FINAL_MANIFEST_ROOT / "final_eligibility_v1_release_candidate.json"
)
FORMAL_FREEZE_PATH = FINAL_MANIFEST_ROOT / "final_eligibility_v1_freeze.json"

COHORTS = ("HRS", "ELSA", "KLoSA")
RELEASE_VERSION = "v1"
SCHEMA_VERSION = "final_eligibility_v1"
BUILDER_VERSION = "final_eligibility_builder_v1.0.0"

FINAL_COLUMNS = (
    "cohort",
    "person_id",
    "tminus1_wave",
    "index_shock_wave",
    "landmark_wave",
    "recovery_status",
    "landmark_actual_timing_valid",
    "primary_fi_model_row_count",
    "primary_later_fi_model_row_count",
    "analysis_panel_base_eligible",
    "primary_complete_case_eligible",
    "weighted_sensitivity_eligible",
    "final_failed_gate",
)

FINAL_DTYPES = {
    "cohort": "string",
    "person_id": "string",
    "tminus1_wave": "Int16",
    "index_shock_wave": "Int16",
    "landmark_wave": "Int16",
    "recovery_status": "string",
    "landmark_actual_timing_valid": "boolean",
    "primary_fi_model_row_count": "Int16",
    "primary_later_fi_model_row_count": "Int16",
    "analysis_panel_base_eligible": "boolean",
    "primary_complete_case_eligible": "boolean",
    "weighted_sensitivity_eligible": "boolean",
    "final_failed_gate": "string",
}

NULLABLE_COLUMNS = {"recovery_status"}
RECOVERY_VALUES = {
    "substantial_recovery",
    "persistent_or_limited_recovery",
}
FINAL_FAILED_GATE_VALUES = {
    "recovery_unavailable",
    "landmark_fi_missing",
    "landmark_timing_unavailable",
    "no_later_fi",
    "no_timed_later_fi",
    "fi_t_missing",
    "prespecified_covariate_missing",
    "none",
}

REQUIRED_DESIGN = {
    "recovery_landmark_wave": "t+1",
    "baseline_outcome_wave": "t+1",
    "first_followup_outcome_wave": "t+2_onward",
    "exposure_outcome_separation_rule": (
        "landmark_baseline_then_subsequent_progression"
    ),
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
        "rows": 8204,
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
        "common22_support": 5082,
        "substantial_recovery": 1703,
        "persistent_or_limited_recovery": 3411,
    },
    "ELSA": {
        "rows": 1284,
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
        "common22_support": 699,
        "substantial_recovery": 271,
        "persistent_or_limited_recovery": 428,
    },
    "KLoSA": {
        "rows": 3125,
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
        "common22_support": 2502,
        "substantial_recovery": 943,
        "persistent_or_limited_recovery": 1559,
    },
}

EXPECTED_GATES = {
    "HRS": {
        "recovery_unavailable": 1657,
        "landmark_fi_missing": 212,
        "landmark_timing_unavailable": 0,
        "no_later_fi": 1113,
        "no_timed_later_fi": 0,
        "fi_t_missing": 72,
        "prespecified_covariate_missing": 36,
        "none": 5114,
    },
    "ELSA": {
        "recovery_unavailable": 316,
        "landmark_fi_missing": 0,
        "landmark_timing_unavailable": 0,
        "no_later_fi": 196,
        "no_timed_later_fi": 0,
        "fi_t_missing": 0,
        "prespecified_covariate_missing": 73,
        "none": 699,
    },
    "KLoSA": {
        "recovery_unavailable": 370,
        "landmark_fi_missing": 0,
        "landmark_timing_unavailable": 0,
        "no_later_fi": 248,
        "no_timed_later_fi": 0,
        "fi_t_missing": 0,
        "prespecified_covariate_missing": 5,
        "none": 2502,
    },
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
        / "11_final_eligibility"
        / f"{cohort.lower()}_final_eligibility_v1.parquet"
    )


def _manifest_path(cohort: str) -> Path:
    return FINAL_MANIFEST_ROOT / f"{cohort.lower()}_final_eligibility_v1.json"


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
        raise RuntimeError("Final Eligibility v1拒绝覆盖既有资产：" + "; ".join(existing))


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
    """验证冻结Ledger及其全部冻结上游。"""
    freeze = ledger_source.verify_analysis_cohort_ledger_freeze()
    return {
        "analysis_cohort_ledger_v1": {
            "freeze_path": _relative(ledger_source.FORMAL_FREEZE_PATH),
            "freeze_sha256": sha256_file(ledger_source.FORMAL_FREEZE_PATH),
            "cohorts": freeze["cohorts"],
            "review_summary": freeze["review_summary"],
            "data_dictionary": freeze["data_dictionary"],
        },
        "upstream_frozen_snapshot": freeze["upstream_frozen_snapshot"],
    }


def _load_ledger(cohort: str) -> pd.DataFrame:
    freeze = ledger_source.verify_analysis_cohort_ledger_freeze()
    record = freeze["cohorts"][cohort]
    frame = pd.read_parquet(ROOT / record["parquet_path"])
    ledger_source.validate_ledger(frame, cohort)
    return frame


def _valid_year(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    return (
        numeric.notna()
        & numeric.between(1900, 2100, inclusive="both")
        & numeric.eq(numeric.round())
    )


def _valid_month(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    return (
        numeric.notna()
        & numeric.between(1, 12, inclusive="both")
        & numeric.eq(numeric.round())
    )


def select_timed_fi_rows(
    people: pd.DataFrame,
    fi: pd.DataFrame,
    spine: pd.DataFrame,
    *,
    fi_column: str = "fi_primary",
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    """选择landmark起具有有效实际年月的FI行；不保存为独立面板。"""
    person_columns = ["cohort", "person_id", "landmark_wave"]
    if people.duplicated(["cohort", "person_id"]).any():
        raise RuntimeError("timed-FI selector收到重复参与者")
    if fi.duplicated(["cohort", "person_id", "wave"]).any():
        raise RuntimeError("FI Panel主键重复")
    if spine.duplicated(["cohort", "person_id", "wave"]).any():
        raise RuntimeError("Full Spine主键重复")

    landmark = spine.rename(
        columns={
            "wave": "landmark_wave",
            "wave_order": "landmark_wave_order",
            "interview_year": "landmark_interview_year",
            "interview_month": "landmark_interview_month",
        }
    )[
        [
            "cohort",
            "person_id",
            "landmark_wave",
            "landmark_wave_order",
            "landmark_interview_year",
            "landmark_interview_month",
        ]
    ]
    person_timing = people[person_columns].merge(
        landmark,
        on=person_columns,
        how="left",
        validate="one_to_one",
        indicator=True,
    )
    if not person_timing["_merge"].eq("both").all():
        raise RuntimeError("landmark无法精确连接Full Spine")
    person_timing = person_timing.drop(columns="_merge")
    person_timing["landmark_actual_timing_valid"] = _valid_year(
        person_timing["landmark_interview_year"]
    ) & _valid_month(person_timing["landmark_interview_month"])

    valid_fi = fi.loc[fi[fi_column].notna(), ["cohort", "person_id", "wave"]]
    rows = valid_fi.merge(
        person_timing,
        on=["cohort", "person_id"],
        how="inner",
        validate="many_to_one",
    ).merge(
        spine[
            [
                "cohort",
                "person_id",
                "wave",
                "wave_order",
                "observed_in_wave",
                "interview_year",
                "interview_month",
            ]
        ],
        on=["cohort", "person_id", "wave"],
        how="left",
        validate="one_to_one",
        indicator=True,
    )
    if not rows["_merge"].eq("both").all():
        raise RuntimeError("有效FI行无法精确连接Full Spine")
    rows = rows.drop(columns="_merge")
    if not rows["observed_in_wave"].eq(True).all():
        raise RuntimeError("有效FI出现在未确认访谈行")
    rows = rows.loc[rows["wave_order"] >= rows["landmark_wave_order"]].copy()
    rows["is_landmark"] = rows["wave"].astype("Int64").eq(
        rows["landmark_wave"].astype("Int64")
    ) & rows["wave_order"].eq(rows["landmark_wave_order"])
    rows["is_later"] = rows["wave_order"] > rows["landmark_wave_order"]
    rows["current_actual_timing_valid"] = _valid_year(
        rows["interview_year"]
    ) & _valid_month(rows["interview_month"])
    rows["time_calculable"] = rows["landmark_actual_timing_valid"] & rows[
        "current_actual_timing_valid"
    ]
    rows["elapsed_months"] = pd.Series(pd.NA, index=rows.index, dtype="Int64")
    calculable = rows["time_calculable"]
    rows.loc[calculable, "elapsed_months"] = (
        12
        * (
            rows.loc[calculable, "interview_year"].astype("Int64")
            - rows.loc[calculable, "landmark_interview_year"].astype("Int64")
        )
        + (
            rows.loc[calculable, "interview_month"].astype("Int64")
            - rows.loc[calculable, "landmark_interview_month"].astype("Int64")
        )
    ).astype("Int64")

    if rows.loc[rows["is_landmark"] & calculable, "elapsed_months"].ne(0).any():
        raise RuntimeError("landmark model time不是0")
    if rows.loc[rows["is_later"] & calculable, "elapsed_months"].le(0).any():
        raise RuntimeError("later FI出现非正model time")

    retained = rows.loc[
        calculable
        & (
            (rows["is_landmark"] & rows["elapsed_months"].eq(0))
            | (rows["is_later"] & rows["elapsed_months"].gt(0))
        )
    ].copy()
    retained = retained.sort_values(
        ["cohort", "person_id", "wave_order"], kind="mergesort"
    )
    earlier = retained.groupby(["cohort", "person_id"], observed=True)[
        "elapsed_months"
    ].shift(1)
    order_check = retained.loc[earlier.notna(), "elapsed_months"].to_numpy()
    prior_check = earlier.loc[earlier.notna()].to_numpy()
    if (order_check <= prior_check).any():
        raise RuntimeError("FI model time未按参与者严格递增")
    if retained.duplicated(["cohort", "person_id", "wave"]).any():
        raise RuntimeError("timed-FI selector输出键重复")

    counts = (
        retained.groupby(["cohort", "person_id"], observed=True)
        .agg(
            primary_fi_model_row_count=("wave", "size"),
            primary_later_fi_model_row_count=("is_later", "sum"),
            timed_landmark_count=("is_landmark", "sum"),
        )
        .reset_index()
    )
    person_timing = person_timing.merge(
        counts,
        on=["cohort", "person_id"],
        how="left",
        validate="one_to_one",
    )
    for column in (
        "primary_fi_model_row_count",
        "primary_later_fi_model_row_count",
        "timed_landmark_count",
    ):
        person_timing[column] = person_timing[column].fillna(0).astype("Int16")
    audit = {
        "valid_fi_rows_from_landmark": int(len(rows)),
        "timed_fi_rows_retained": int(len(retained)),
        "fi_rows_excluded_for_unavailable_actual_timing": int(
            (~rows["time_calculable"]).sum()
        ),
    }
    return retained, person_timing, audit


def _first_failed_gate(frame: pd.DataFrame) -> pd.Series:
    conditions = [
        ~frame["recovery_classifiable"],
        ~frame["landmark_fi_available"],
        ~frame["landmark_actual_timing_valid"],
        frame["later_fi_count"].lt(1),
        frame["primary_later_fi_model_row_count"].lt(1),
        ~frame["fi_t_available"],
        ~frame["nine_covariates_complete"],
    ]
    values = [
        "recovery_unavailable",
        "landmark_fi_missing",
        "landmark_timing_unavailable",
        "no_later_fi",
        "no_timed_later_fi",
        "fi_t_missing",
        "prespecified_covariate_missing",
    ]
    return pd.Series(
        np.select(conditions, values, default="none"),
        index=frame.index,
        dtype="string",
    )


def _validate_against_ledger(
    frame: pd.DataFrame, source: pd.DataFrame, cohort: str
) -> None:
    """确认Final Eligibility没有改变冻结Ledger的人员、顺序和锚点。"""
    if len(frame) != len(source):
        raise RuntimeError(f"{cohort}: Final Eligibility与Ledger行数不一致")
    comparison = source[
        [
            "cohort",
            "person_id",
            "tminus1_wave",
            "index_shock_wave",
            "landmark_wave",
            "recovery_status",
        ]
    ].copy()
    pd.testing.assert_frame_equal(
        frame[comparison.columns].reset_index(drop=True),
        comparison.reset_index(drop=True),
        check_dtype=False,
        check_like=False,
    )


def _cast(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.loc[:, FINAL_COLUMNS].copy()
    for column, dtype in FINAL_DTYPES.items():
        output[column] = output[column].astype(dtype)
    return output


def validate_final_eligibility(frame: pd.DataFrame, cohort: str) -> dict[str, Any]:
    if tuple(frame.columns) != FINAL_COLUMNS:
        raise RuntimeError(f"{cohort}: Final Eligibility列顺序不符合13列合同")
    if len(frame) != EXPECTED[cohort]["rows"]:
        raise RuntimeError(f"{cohort}: Final Eligibility行数不符合预期")
    if frame.duplicated(["cohort", "person_id"]).any():
        raise RuntimeError(f"{cohort}: Final Eligibility主键重复")
    if not frame["cohort"].eq(cohort).all():
        raise RuntimeError(f"{cohort}: cohort字段错误")
    for column in FINAL_COLUMNS:
        if column not in NULLABLE_COLUMNS and frame[column].isna().any():
            raise RuntimeError(f"{cohort}: 非空列出现NA：{column}")
        if str(frame[column].dtype) != FINAL_DTYPES[column]:
            raise RuntimeError(f"{cohort}: dtype不符合合同：{column}")
    observed_recovery = set(frame["recovery_status"].dropna().astype(str).unique())
    if not observed_recovery.issubset(RECOVERY_VALUES):
        raise RuntimeError(f"{cohort}: recovery_status出现非法枚举")
    if not set(frame["final_failed_gate"].astype(str).unique()).issubset(
        FINAL_FAILED_GATE_VALUES
    ):
        raise RuntimeError(f"{cohort}: final_failed_gate出现非法枚举")

    base = frame["analysis_panel_base_eligible"]
    primary = frame["primary_complete_case_eligible"]
    weighted = frame["weighted_sensitivity_eligible"]
    if not (weighted <= primary).all() or not (primary <= base).all():
        raise RuntimeError(f"{cohort}: eligibility层级不嵌套")
    if not frame["final_failed_gate"].eq("none").eq(primary).all():
        raise RuntimeError(f"{cohort}: none与primary eligibility不一致")
    if not frame.loc[base, "landmark_actual_timing_valid"].eq(True).all():
        raise RuntimeError(f"{cohort}: base eligible存在无效landmark时间")
    if not frame.loc[base, "primary_fi_model_row_count"].ge(2).all():
        raise RuntimeError(f"{cohort}: base eligible的FI行数少于2")
    if not frame.loc[base, "primary_later_fi_model_row_count"].ge(1).all():
        raise RuntimeError(f"{cohort}: base eligible缺少later FI行")
    if not frame.loc[base, "primary_fi_model_row_count"].eq(
        1 + frame.loc[base, "primary_later_fi_model_row_count"]
    ).all():
        raise RuntimeError(f"{cohort}: base eligible行数不等于1+later")

    gates = {
        name: int(frame["final_failed_gate"].eq(name).sum())
        for name in (
            "recovery_unavailable",
            "landmark_fi_missing",
            "landmark_timing_unavailable",
            "no_later_fi",
            "no_timed_later_fi",
            "fi_t_missing",
            "prespecified_covariate_missing",
            "none",
        )
    }
    metrics = {
        "rows": len(frame),
        "base_participants": int(base.sum()),
        "base_fi_rows": int(frame.loc[base, "primary_fi_model_row_count"].sum()),
        "base_landmark_rows": int(base.sum()),
        "base_later_rows": int(
            frame.loc[base, "primary_later_fi_model_row_count"].sum()
        ),
        "primary_participants": int(primary.sum()),
        "primary_fi_rows": int(
            frame.loc[primary, "primary_fi_model_row_count"].sum()
        ),
        "primary_landmark_rows": int(primary.sum()),
        "primary_later_rows": int(
            frame.loc[primary, "primary_later_fi_model_row_count"].sum()
        ),
        "weighted_participants": int(weighted.sum()),
        "weighted_fi_rows": int(
            frame.loc[weighted, "primary_fi_model_row_count"].sum()
        ),
        "substantial_recovery": int(
            frame.loc[primary, "recovery_status"].eq("substantial_recovery").sum()
        ),
        "persistent_or_limited_recovery": int(
            frame.loc[primary, "recovery_status"]
            .eq("persistent_or_limited_recovery")
            .sum()
        ),
    }
    expected_metrics = {
        key: value for key, value in EXPECTED[cohort].items() if key != "common22_support"
    }
    if metrics != expected_metrics:
        raise RuntimeError(f"{cohort}: Final Eligibility汇总不符合预期：{metrics}")
    if sum(gates.values()) != len(frame) or gates != EXPECTED_GATES[cohort]:
        raise RuntimeError(f"{cohort}: final failed gate不守恒")
    if (
        metrics["substantial_recovery"]
        + metrics["persistent_or_limited_recovery"]
        != metrics["primary_participants"]
    ):
        raise RuntimeError(f"{cohort}: primary recovery组不守恒")
    return {"metrics": metrics, "final_failed_gate": gates}


def build_final_eligibility(
    cohort: str,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    source = _load_ledger(cohort)
    fi = ledger_source._load_frozen_panel(
        "fi_panel_v2",
        cohort,
        ["cohort", "person_id", "wave", "fi_primary", "fi_common22"],
    )
    spine = ledger_source._load_frozen_panel(
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
    _, timing, timing_audit = select_timed_fi_rows(source, fi, spine)
    working = source.merge(
        timing[
            [
                "cohort",
                "person_id",
                "landmark_actual_timing_valid",
                "primary_fi_model_row_count",
                "primary_later_fi_model_row_count",
                "timed_landmark_count",
            ]
        ],
        on=["cohort", "person_id"],
        how="left",
        validate="one_to_one",
    )
    working["analysis_panel_base_eligible"] = (
        working["recovery_classifiable"]
        & working["landmark_fi_available"]
        & working["landmark_actual_timing_valid"]
        & working["later_fi_count"].ge(1)
        & working["primary_later_fi_model_row_count"].ge(1)
        & working["timed_landmark_count"].eq(1)
    )
    working["primary_complete_case_eligible"] = (
        working["analysis_panel_base_eligible"]
        & working["fi_t_available"]
        & working["nine_covariates_complete"]
    )
    working["weighted_sensitivity_eligible"] = (
        working["primary_complete_case_eligible"]
        & working["landmark_weight_status"].eq("positive")
    )
    working["final_failed_gate"] = _first_failed_gate(working)
    output = _cast(working)
    _validate_against_ledger(output, source, cohort)

    # Common-22只在primary pool内做非门控支持诊断。
    primary_people = output.loc[
        output["primary_complete_case_eligible"],
        ["cohort", "person_id", "landmark_wave"],
    ]
    common_rows, common_timing, _ = select_timed_fi_rows(
        primary_people, fi, spine, fi_column="fi_common22"
    )
    common_support = common_timing.loc[
        common_timing["timed_landmark_count"].eq(1)
        & common_timing["primary_later_fi_model_row_count"].ge(1),
        "person_id",
    ].nunique()
    if int(common_support) != EXPECTED[cohort]["common22_support"]:
        raise RuntimeError(f"{cohort}: Common-22非门控支持人数不符合预期")
    diagnostics = {
        **timing_audit,
        "common22_landmark_and_later_support_within_primary": int(common_support),
        "common22_timed_rows_within_primary_pool": int(len(common_rows)),
    }
    validation = validate_final_eligibility(output, cohort)
    return output, validation, diagnostics


def _review_rows(
    panels: dict[str, pd.DataFrame],
    validations: dict[str, dict[str, Any]],
    diagnostics: dict[str, dict[str, Any]],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for cohort in COHORTS:
        frame = panels[cohort]
        metrics = validations[cohort]["metrics"]
        for metric in (
            "base_participants",
            "base_fi_rows",
            "base_landmark_rows",
            "base_later_rows",
            "primary_participants",
            "primary_fi_rows",
            "primary_landmark_rows",
            "primary_later_rows",
            "weighted_participants",
            "weighted_fi_rows",
        ):
            rows.append(
                {
                    "section": "eligibility_and_model_rows",
                    "cohort": cohort,
                    "metric": metric,
                    "recovery_status": "all",
                    "n": metrics[metric],
                }
            )
        for gate, n in validations[cohort]["final_failed_gate"].items():
            rows.append(
                {
                    "section": "mutually_exclusive_final_failed_gate",
                    "cohort": cohort,
                    "metric": gate,
                    "recovery_status": "all",
                    "n": n,
                }
            )
        primary = frame.loc[frame["primary_complete_case_eligible"]]
        for status in sorted(RECOVERY_VALUES):
            rows.append(
                {
                    "section": "primary_recovery_groups",
                    "cohort": cohort,
                    "metric": "primary_complete_case_participants",
                    "recovery_status": status,
                    "n": int(primary["recovery_status"].eq(status).sum()),
                }
            )
        for metric in (
            "fi_rows_excluded_for_unavailable_actual_timing",
            "common22_landmark_and_later_support_within_primary",
        ):
            rows.append(
                {
                    "section": "non_gating_diagnostics",
                    "cohort": cohort,
                    "metric": metric,
                    "recovery_status": "all",
                    "n": diagnostics[cohort][metric],
                }
            )
    return pd.DataFrame(rows, columns=["section", "cohort", "metric", "recovery_status", "n"])


def _dictionary_text() -> str:
    return """# Final Eligibility v1 data dictionary

每行是一名已由冻结 Analysis Cohort Ledger v1 识别的 analytic index-shock participant。该资产是 Analysis Panel 前最后一个 participant-level eligibility gate；它不是 longitudinal panel，也不保存模型结局、协变量或权重数值。

| Column | Type | Nullable | Definition |
| --- | --- | --- | --- |
| `cohort` | string | no | `HRS`、`ELSA`或`KLoSA`。 |
| `person_id` | string | no | 队列内参与者标识。 |
| `tminus1_wave` | Int16 | no | index shock前一波。 |
| `index_shock_wave` | Int16 | no | analytic window内首次qualifying major shock波次。 |
| `landmark_wave` | Int16 | no | exact `t+1` landmark波次。 |
| `recovery_status` | string | yes | `substantial_recovery`、`persistent_or_limited_recovery`或NA。 |
| `landmark_actual_timing_valid` | boolean | no | landmark实际访谈年份为1900--2100且月份为1--12。 |
| `primary_fi_model_row_count` | Int16 | no | 共享timed-FI selector从landmark起保留的、具有有效model timing的primary-FI记录数。 |
| `primary_later_fi_model_row_count` | Int16 | no | landmark后具有有效正model time的primary-FI记录数。 |
| `analysis_panel_base_eligible` | boolean | no | Eligibility for inclusion in the longitudinal Analysis Panel base pool; not equivalent to primary complete-case eligibility. |
| `primary_complete_case_eligible` | boolean | no | Base eligible且`FI_t`及九项预设协变量完整；正式participant-level primary eligibility。 |
| `weighted_sensitivity_eligible` | boolean | no | Primary eligible且exact-`t+1` landmark person weight为正。 |
| `final_failed_gate` | string | no | 按冻结顺序得到的首个primary eligibility失败原因；全部通过为`none`。 |

Model time只使用Full Spine实际`interview_year`和`interview_month`：`elapsed_months = 12*(Y_current-Y_landmark) + (M_current-M_landmark)`，`time_years = elapsed_months/12`。不使用nominal year、插值、假定两年间隔或跨波回填。单条later FI缺少时间只排除该记录；仍有至少一条有效later FI时不排除参与者。

`final_failed_gate`依次为：`recovery_unavailable`、`landmark_fi_missing`、`landmark_timing_unavailable`、`no_later_fi`、`no_timed_later_fi`、`fi_t_missing`、`prespecified_covariate_missing`、`none`。

Common-22只作为非门控支持诊断。未来MI可从`analysis_panel_base_eligible`最大池开始，但本资产不决定是否插补`FI_t`、插补模型或最终MI样本。
"""


def _schema_contract(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return [
        {
            "name": column,
            "dtype": str(frame[column].dtype),
            "nullable": column in NULLABLE_COLUMNS,
        }
        for column in FINAL_COLUMNS
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
        "manifest_schema_version": "final_eligibility_manifest_v1",
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
        "row_unit": "one participant with one analytic index shock",
        "common_key": ["cohort", "person_id"],
        "physical_columns": list(FINAL_COLUMNS),
        "schema": _schema_contract(panel),
        "row_count": len(panel),
        "unique_key": not panel.duplicated(["cohort", "person_id"]).any(),
        **validation,
        "non_gating_diagnostics": diagnostics,
        "output": {
            "path": _relative(_output_path(cohort)),
            "sha256": sha256_file(staged_output),
            "compression": "zstd",
        },
        "protected_input_snapshot": protected,
        "invariants": {
            "ledger_row_set_and_order_preserved": True,
            "base_rows_equal_one_plus_later_rows": True,
            "base_requires_valid_landmark_time_and_at_least_one_later_row": True,
            "eligibility_layers_nested": True,
            "exact_tplus1_weight_without_fallback": True,
            "common22_does_not_gate_primary": True,
            "analysis_panel_not_built": True,
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


def run_final_eligibility_release(cohorts: list[str]) -> dict[str, Any]:
    """一次性生成三队列Final Eligibility v1复核候选。"""
    if tuple(cohorts) != COHORTS:
        raise RuntimeError("eligibility-final v1必须使用--cohort all共同构建")
    _verify_release_absent()
    _read_design_register()
    design_sha = sha256_file(DESIGN_REGISTER_PATH)
    protected_before = protected_snapshot()
    provenance = _git_provenance()

    tmp_root = ROOT / "tmp"
    tmp_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="final_eligibility_", dir=tmp_root) as name:
        staging = Path(name)
        panels: dict[str, pd.DataFrame] = {}
        validations: dict[str, dict[str, Any]] = {}
        diagnostics: dict[str, dict[str, Any]] = {}
        staged_outputs: dict[str, Path] = {}
        staged_manifests: dict[str, Path] = {}
        for cohort in COHORTS:
            panel, validation, diagnostic = build_final_eligibility(cohort)
            panels[cohort] = panel
            validations[cohort] = validation
            diagnostics[cohort] = diagnostic
            output = staging / f"{cohort.lower()}_final_eligibility_v1.parquet"
            panel.to_parquet(output, index=False, compression="zstd")
            staged_outputs[cohort] = output
            manifest_path = staging / f"{cohort.lower()}_final_eligibility_v1.json"
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

        review = _review_rows(panels, validations, diagnostics)
        staged_review = staging / REVIEW_PATH.name
        review.to_csv(staged_review, index=False, encoding="utf-8-sig")
        staged_dictionary = staging / DICTIONARY_PATH.name
        staged_dictionary.write_text(_dictionary_text(), encoding="utf-8")
        release = {
            "release_candidate_schema_version": "final_eligibility_release_candidate_v1",
            "release_status": "READY_FOR_REVIEW",
            "candidate_label": "FINAL ELIGIBILITY V1 READY FOR REVIEW",
            "formal_freeze_created": False,
            "release_version": RELEASE_VERSION,
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
            "row_unit": "one participant with one analytic index shock",
            "common_key": ["cohort", "person_id"],
            "physical_columns": list(FINAL_COLUMNS),
            "cohorts": {
                cohort: {
                    "parquet_path": _relative(_output_path(cohort)),
                    "parquet_sha256": sha256_file(staged_outputs[cohort]),
                    "manifest_path": _relative(_manifest_path(cohort)),
                    "manifest_sha256": sha256_file(staged_manifests[cohort]),
                    **validations[cohort],
                    "non_gating_diagnostics": diagnostics[cohort],
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
                "participant_level_primary_eligibility_defined_in_candidate": True,
                "formal_eligibility_freeze_created": False,
                "analysis_panel_built": False,
                "model_run": False,
                "figure_s1_modified": False,
                "manuscript_or_supplement_modified": False,
            },
            "next_step": (
                "Author review; an unchanged candidate may be frozen without rebuilding. "
                "After freeze the next participant-data stage is Analysis Panel."
            ),
        }
        staged_release = staging / RELEASE_CANDIDATE_PATH.name
        _json_write(staged_release, release)
        protected_after = protected_snapshot()
        if protected_after != protected_before:
            raise RuntimeError("Final Eligibility构建期间冻结上游发生变化")
        if sha256_file(DESIGN_REGISTER_PATH) != design_sha:
            raise RuntimeError("Final Eligibility构建期间design register发生变化")

        publish = [
            *(
                (staged_outputs[cohort], _output_path(cohort))
                for cohort in COHORTS
            ),
            *(
                (staged_manifests[cohort], _manifest_path(cohort))
                for cohort in COHORTS
            ),
            (staged_review, REVIEW_PATH),
            (staged_dictionary, DICTIONARY_PATH),
            (staged_release, RELEASE_CANDIDATE_PATH),
        ]
        _publish_transaction(publish)

    return verify_final_eligibility_candidate()


def verify_final_eligibility_candidate(
    *,
    require_current_design_hash: bool = True,
    require_current_builder_hash: bool = True,
) -> dict[str, Any]:
    """重载候选及全部引用资产。"""
    if not RELEASE_CANDIDATE_PATH.is_file():
        raise RuntimeError("缺少Final Eligibility v1 release candidate")
    release = json.loads(RELEASE_CANDIDATE_PATH.read_text(encoding="utf-8"))
    if release.get("release_status") != "READY_FOR_REVIEW":
        raise RuntimeError("Final Eligibility候选状态错误")
    if release.get("physical_columns") != list(FINAL_COLUMNS):
        raise RuntimeError("Final Eligibility候选schema漂移")
    if (
        require_current_builder_hash
        and release.get("builder_sha256") != sha256_file(Path(__file__))
    ):
        raise RuntimeError("Final Eligibility候选builder哈希漂移")
    _read_design_register()
    if (
        require_current_design_hash
        and release["design_register"]["sha256"] != sha256_file(DESIGN_REGISTER_PATH)
    ):
        raise RuntimeError("Final Eligibility候选design register哈希漂移")
    if release["protected_input_snapshot"] != protected_snapshot():
        raise RuntimeError("Final Eligibility候选冻结上游漂移")
    panels: dict[str, pd.DataFrame] = {}
    validations: dict[str, dict[str, Any]] = {}
    diagnostics: dict[str, dict[str, Any]] = {}
    for cohort in COHORTS:
        record = release["cohorts"][cohort]
        if sha256_file(ROOT / record["parquet_path"]) != record["parquet_sha256"]:
            raise RuntimeError(f"{cohort}: Final Eligibility Parquet哈希漂移")
        if sha256_file(ROOT / record["manifest_path"]) != record["manifest_sha256"]:
            raise RuntimeError(f"{cohort}: Final Eligibility manifest哈希漂移")
        frame = pd.read_parquet(ROOT / record["parquet_path"])
        validation = validate_final_eligibility(frame, cohort)
        _validate_against_ledger(frame, _load_ledger(cohort), cohort)
        if validation != {
            "metrics": record["metrics"],
            "final_failed_gate": record["final_failed_gate"],
        }:
            raise RuntimeError(f"{cohort}: Final Eligibility内容与候选不一致")
        panels[cohort] = frame
        validations[cohort] = validation
        diagnostics[cohort] = record["non_gating_diagnostics"]
    review_record = release["review_summary"]
    if sha256_file(ROOT / review_record["path"]) != review_record["sha256"]:
        raise RuntimeError("Final Eligibility review CSV哈希漂移")
    observed = pd.read_csv(ROOT / review_record["path"], encoding="utf-8-sig")
    expected = _review_rows(panels, validations, diagnostics)
    pd.testing.assert_frame_equal(observed, expected, check_dtype=False)
    if "person_id" in observed.columns:
        raise RuntimeError("Final Eligibility review CSV不得包含person_id")
    if sha256_file(ROOT / release["data_dictionary"]["path"]) != release[
        "data_dictionary"
    ]["sha256"]:
        raise RuntimeError("Final Eligibility数据字典哈希漂移")
    return release


def freeze_final_eligibility_v1(approved_candidate_sha256: str) -> dict[str, Any]:
    """按作者批准的候选哈希一次性冻结；不重建Parquet。"""
    if FORMAL_FREEZE_PATH.exists():
        raise RuntimeError("Final Eligibility v1已冻结，拒绝覆盖")
    release = verify_final_eligibility_candidate()
    current_sha = sha256_file(RELEASE_CANDIDATE_PATH)
    if current_sha != approved_candidate_sha256:
        raise RuntimeError("Final Eligibility候选哈希与作者批准值不一致")
    freeze = {
        "freeze_record_schema_version": "final_eligibility_freeze_record_v1",
        "freeze_status": "FROZEN",
        "eligibility_status": "PARTICIPANT_LEVEL_ELIGIBILITY_LOCKED",
        "release_version": RELEASE_VERSION,
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
        "interpretation": {
            "primary_complete_case_eligible_is_final_participant_eligibility": True,
            "analysis_panel_base_is_not_primary_sample": True,
            "weighted_branch_does_not_restrict_primary": True,
            "no_further_participant_level_eligibility_stage": True,
            "analysis_panel_still_pending": True,
        },
        "release_integrity": {
            "parquet_rebuilt_for_freeze": False,
            "cohort_manifest_rebuilt_for_freeze": False,
            "review_csv_rebuilt_for_freeze": False,
        },
        "overwrite_policy": (
            "Final Eligibility v1 is immutable; changes require a new version."
        ),
        "next_participant_data_stage": "analysis-panel",
    }
    FINAL_MANIFEST_ROOT.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="final_eligibility_freeze_", dir=ROOT / "tmp") as name:
        staged = Path(name) / FORMAL_FREEZE_PATH.name
        _json_write(staged, freeze)
        os.replace(staged, FORMAL_FREEZE_PATH)
    return verify_final_eligibility_freeze()


def verify_final_eligibility_freeze() -> dict[str, Any]:
    if not FORMAL_FREEZE_PATH.is_file():
        raise RuntimeError("Final Eligibility v1 freeze不存在")
    freeze = json.loads(FORMAL_FREEZE_PATH.read_text(encoding="utf-8"))
    if freeze.get("freeze_status") != "FROZEN":
        raise RuntimeError("Final Eligibility freeze_status错误")
    source = freeze["source_release_candidate"]
    if sha256_file(RELEASE_CANDIDATE_PATH) != source["sha256"]:
        raise RuntimeError("Frozen Final Eligibility候选哈希漂移")
    release = verify_final_eligibility_candidate(
        require_current_design_hash=False,
        require_current_builder_hash=False,
    )
    if freeze["cohorts"] != release["cohorts"]:
        raise RuntimeError("Final Eligibility freeze与候选不一致")
    return freeze


__all__ = [
    "COHORTS",
    "EXPECTED",
    "EXPECTED_GATES",
    "FINAL_COLUMNS",
    "FINAL_DTYPES",
    "FORMAL_FREEZE_PATH",
    "RELEASE_CANDIDATE_PATH",
    "REVIEW_PATH",
    "build_final_eligibility",
    "freeze_final_eligibility_v1",
    "run_final_eligibility_release",
    "select_timed_fi_rows",
    "validate_final_eligibility",
    "verify_final_eligibility_candidate",
    "verify_final_eligibility_freeze",
]
