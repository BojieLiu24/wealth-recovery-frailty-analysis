from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import subprocess
import sys
import zipfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

import numpy as np
import pandas as pd
import pyarrow


ROOT = Path(__file__).resolve().parents[3]
PANEL_CODE_DIR = ROOT / "02_code" / "01_panel_build"
TABLE_1_CODE_DIR = ROOT / "02_code" / "03_results" / "table_1"
for code_dir in (PANEL_CODE_DIR, TABLE_1_CODE_DIR):
    if str(code_dir) not in sys.path:
        sys.path.insert(0, str(code_dir))

import analysis_cohort_ledger
import analysis_panel
import build_table_1 as table_1_builder
import final_eligibility


OUTPUT_DIR = (
    ROOT
    / "03_output"
    / "3.1 Study Population, Wealth Recovery, and Follow-up"
    / "Table S5"
)
PLAN_PATH = ROOT / "07_manuscript" / "Table" / "table_s5_generation_plan_20260915.md"
MANUSCRIPT_PATH = ROOT / "07_manuscript" / "manuscript_v1.docx"
SUPPLEMENT_DIR = ROOT / "07_manuscript" / "supplementary_materials"
SUPPLEMENT_MAIN_PATH = SUPPLEMENT_DIR / "main.tex"
SUPPLEMENT_TABLE_PATH = SUPPLEMENT_DIR / "tables" / "table_s5.tex"
SUPPLEMENT_PDF_PATH = SUPPLEMENT_DIR / "output" / "Supplementary_Materials.pdf"
REGISTRY_PATH = ROOT / "00_admin" / "cohort_registry.csv"

LEDGER_FREEZE_PATH = (
    ROOT
    / "00_admin"
    / "panel_manifests"
    / "analysis_cohort_ledger"
    / "analysis_cohort_ledger_v1_freeze.json"
)
FINAL_FREEZE_PATH = (
    ROOT
    / "00_admin"
    / "panel_manifests"
    / "final_eligibility"
    / "final_eligibility_v1_freeze.json"
)
ANALYSIS_FREEZE_PATH = (
    ROOT
    / "00_admin"
    / "panel_manifests"
    / "analysis_panel"
    / "analysis_panel_v1_freeze.json"
)
TABLE_1_OUTPUT_DIR = (
    ROOT
    / "03_output"
    / "3.1 Study Population, Wealth Recovery, and Follow-up"
    / "Table 1"
)
TABLE_1_FREEZE_PATH = TABLE_1_OUTPUT_DIR / "table_1_freeze.json"
TABLE_1_SUMMARY_PATH = TABLE_1_OUTPUT_DIR / "table_1_summary_long.csv"

SUMMARY_NAME = "table_s5_summary_long.csv"
SPEC_NAME = "table_s5_display_spec.json"
DOCX_NAME = "table_s5_editable.docx"
TEX_NAME = "table_s5.tex"
MANIFEST_NAME = "table_s5_manifest.json"
VALIDATION_NAME = "table_s5_validation.json"
TABLE_FREEZE_NAME = "table_s5_freeze.json"
CONTEXT_NAME = "table_s5_build_context.json"
KNOWN_OUTPUTS = (
    SUMMARY_NAME,
    SPEC_NAME,
    DOCX_NAME,
    TEX_NAME,
    MANIFEST_NAME,
    VALIDATION_NAME,
)

COHORTS = ("HRS", "ELSA", "KLoSA")
GROUPS = ("substantial_recovery", "persistent_or_limited_recovery")
GROUP_LABELS = {
    "substantial_recovery": "Substantial recovery",
    "persistent_or_limited_recovery": "Persistent/limited recovery",
}
THRESHOLDS = (2, 4, 6, 8, 10)
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
DISPLAYED_MISSINGNESS = (
    ("fi_t", "Missing FI at shock interview"),
    ("education", "Missing education"),
    ("marital_status", "Missing marital status"),
    ("current_smoking", "Missing current smoking"),
    ("labour_force_status", "Missing labour-force status"),
)
COMPLETE_ADJUSTMENT_COLUMNS = (
    "age",
    "sex",
    "household_size",
    "household_income_rank",
    "wealth_quintile",
)

EXPECTED = {
    "HRS": {
        "shock_cohort": 8204,
        "recovery_unavailable": 1657,
        "landmark_fi_missing": 212,
        "landmark_timing_unavailable": 0,
        "no_later_fi": 1113,
        "no_timed_later_fi": 0,
        "mi_pool": 5222,
        "any_missing": 108,
        "complete_case": 5114,
        "later_rows": 18297,
        "groups": {
            "substantial_recovery": 1703,
            "persistent_or_limited_recovery": 3411,
        },
        "missing": {
            "fi_t": 72,
            "education": 1,
            "marital_status": 6,
            "current_smoking": 30,
            "labour_force_status": 0,
        },
        "assessment_distribution": {
            "substantial_recovery": {"1": 360, "2": 316, "3": 278, "4_plus": 749},
            "persistent_or_limited_recovery": {"1": 836, "2": 691, "3": 480, "4_plus": 1404},
        },
        "support_10_years": {
            "substantial_recovery": 516,
            "persistent_or_limited_recovery": 937,
        },
    },
    "ELSA": {
        "shock_cohort": 1284,
        "recovery_unavailable": 316,
        "landmark_fi_missing": 0,
        "landmark_timing_unavailable": 0,
        "no_later_fi": 196,
        "no_timed_later_fi": 0,
        "mi_pool": 772,
        "any_missing": 73,
        "complete_case": 699,
        "later_rows": 1885,
        "groups": {
            "substantial_recovery": 271,
            "persistent_or_limited_recovery": 428,
        },
        "missing": {
            "fi_t": 0,
            "education": 69,
            "marital_status": 1,
            "current_smoking": 3,
            "labour_force_status": 0,
        },
        "assessment_distribution": {
            "substantial_recovery": {"1": 83, "2": 62, "3": 46, "4_plus": 80},
            "persistent_or_limited_recovery": {"1": 124, "2": 86, "3": 95, "4_plus": 123},
        },
        "support_10_years": {
            "substantial_recovery": 40,
            "persistent_or_limited_recovery": 69,
        },
    },
    "KLoSA": {
        "shock_cohort": 3125,
        "recovery_unavailable": 370,
        "landmark_fi_missing": 0,
        "landmark_timing_unavailable": 0,
        "no_later_fi": 248,
        "no_timed_later_fi": 0,
        "mi_pool": 2507,
        "any_missing": 5,
        "complete_case": 2502,
        "later_rows": 8260,
        "groups": {
            "substantial_recovery": 943,
            "persistent_or_limited_recovery": 1559,
        },
        "missing": {
            "fi_t": 0,
            "education": 0,
            "marital_status": 0,
            "current_smoking": 0,
            "labour_force_status": 5,
        },
        "assessment_distribution": {
            "substantial_recovery": {"1": 203, "2": 158, "3": 172, "4_plus": 410},
            "persistent_or_limited_recovery": {"1": 300, "2": 271, "3": 294, "4_plus": 694},
        },
        "support_10_years": {
            "substantial_recovery": 223,
            "persistent_or_limited_recovery": 413,
        },
    },
}

CAPTION = (
    "Supplementary Table S5. Follow-up availability and missing data by cohort "
    "and post-shock wealth-recovery status."
)
PANEL_A_HEADING = "Panel A. Follow-up support in the primary complete-case analytic sample"
NOTES = (
    "Values are n (%) unless otherwise indicated. Follow-up support is shown for "
    "the cohort-specific primary complete-case analytic sample. Percentages in "
    "Panel A use the corresponding cohort × recovery-status group as the "
    "denominator. Participants reaching a follow-up threshold had at least one "
    "valid post-landmark FI assessment at or beyond that time, based on actual "
    "interview dates. Sample selection from the shock-experienced cohort to the "
    "MI support pool is shown in Supplementary Figure S1. Adjustment-variable "
    "missingness is reported within the participant-level MI support pool; "
    "missingness categories may overlap and therefore need not sum to the number "
    "with any missing adjustment variable. The MI support pool represents the "
    "maximum longitudinal participant pool available for the planned "
    "multiple-imputation sensitivity analysis; no imputation was performed for "
    "this table. Age, sex, household size, household income "
    "rank, and pre-shock wealth quintile were complete within the MI support pool. "
    "Substantial recovery was defined as regaining at least 50% of the wealth lost "
    "by the landmark interview; persistent/limited recovery was defined as regaining "
    "less than 50%. Descriptive statistics were unweighted. Post-landmark "
    "observations exclude the landmark assessment. FI, frailty index; MI, multiple "
    "imputation; HRS, Health and Retirement Study; ELSA, English Longitudinal Study "
    "of Ageing; KLoSA, Korean Longitudinal Study of Aging."
)


def _relative(path: Path) -> str:
    """返回仓库相对 POSIX 路径。"""
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def _sha256_file(path: Path) -> str:
    """计算文件 SHA-256。"""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _protected_hash(path: Path) -> dict[str, Any]:
    """计算保护文件哈希；若被 Word 独占则使用 Git blob 哈希。"""
    stat = path.stat()
    try:
        return {
            "algorithm": "sha256",
            "hash": _sha256_file(path),
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
        }
    except PermissionError:
        result = subprocess.run(
            ["git", "hash-object", "--", str(path)],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
        return {
            "algorithm": "git_blob_sha1",
            "hash": result.stdout.strip(),
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
        }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    """以 UTF-8 和稳定键序写入 JSON。"""
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _read_json(path: Path) -> dict[str, Any]:
    """读取 UTF-8 JSON。"""
    return json.loads(path.read_text(encoding="utf-8"))


def _git_state() -> dict[str, Any]:
    """记录 Git 状态，不改变索引或工作树。"""
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True
    ).stdout.decode("ascii").strip()
    status = subprocess.run(
        ["git", "status", "--porcelain=v1"], cwd=ROOT, check=True, capture_output=True
    ).stdout.decode("utf-8", errors="replace").splitlines()
    diff = subprocess.run(
        ["git", "diff", "--binary", "--no-ext-diff"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    ).stdout
    return {
        "git_commit": commit,
        "git_dirty": bool(status),
        "git_status_entries": status,
        "git_diff_sha256": hashlib.sha256(diff).hexdigest(),
    }


def _protected_snapshot() -> dict[str, Any]:
    """记录正文及未经授权改动的 Supplementary Materials 文件。"""
    excluded = {
        SUPPLEMENT_MAIN_PATH.resolve(),
        SUPPLEMENT_TABLE_PATH.resolve(),
        SUPPLEMENT_PDF_PATH.resolve(),
    }
    supplement: dict[str, Any] = {}
    for path in sorted(item for item in SUPPLEMENT_DIR.rglob("*") if item.is_file()):
        if path.resolve() in excluded:
            continue
        supplement[_relative(path)] = _protected_hash(path)
    return {
        "manuscript": {"path": _relative(MANUSCRIPT_PATH), **_protected_hash(MANUSCRIPT_PATH)},
        "supplementary_materials_unchanged_scope": supplement,
    }


def _raw_snapshot() -> dict[str, Any]:
    """只为保护性核验读取原始文件哈希，不把它们作为分析输入。"""
    registry = pd.read_csv(REGISTRY_PATH, dtype="string")
    output: dict[str, Any] = {}
    for row in registry.itertuples(index=False):
        path = Path(str(row.dataset_path))
        stat = path.stat()
        output[str(row.cohort)] = {
            "filename": path.name,
            "sha256": _sha256_file(path),
            "bytes": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
        }
    return output


def _verify_table_1_assets() -> dict[str, Any]:
    """验证 Table 1 冻结资产与依赖，不要求旧补充材料快照保持不变。"""
    freeze = _read_json(TABLE_1_FREEZE_PATH)
    if freeze.get("freeze_status") != "FROZEN":
        raise RuntimeError("Table 1 freeze_status 不是 FROZEN")
    for name, record in freeze["release_assets"].items():
        path = TABLE_1_OUTPUT_DIR / name
        if not path.is_file() or _sha256_file(path) != record["sha256"]:
            raise RuntimeError(f"Table 1 冻结资产漂移：{name}")
    for record in freeze["dependencies"].values():
        path = ROOT / record["path"]
        if _sha256_file(path) != record["sha256"]:
            raise RuntimeError(f"Table 1 冻结依赖漂移：{record['path']}")
    return {
        "freeze_status": "FROZEN",
        "release_version": freeze["release_version"],
        "freeze_sha256": _sha256_file(TABLE_1_FREEZE_PATH),
        "summary_sha256": _sha256_file(TABLE_1_SUMMARY_PATH),
        "release_asset_count": len(freeze["release_assets"]),
    }


def _source_snapshot(
    ledger_freeze: dict[str, Any],
    final_freeze: dict[str, Any],
    panel_freeze: dict[str, Any],
) -> dict[str, Any]:
    """记录三层冻结输入及 Table 1 anchor。"""
    output: dict[str, Any] = {}
    for label, path, freeze in (
        ("analysis_cohort_ledger", LEDGER_FREEZE_PATH, ledger_freeze),
        ("final_eligibility", FINAL_FREEZE_PATH, final_freeze),
        ("analysis_panel", ANALYSIS_FREEZE_PATH, panel_freeze),
    ):
        output[label] = {
            "freeze_path": _relative(path),
            "freeze_sha256": _sha256_file(path),
            "release_version": freeze["release_version"],
            "cohorts": {
                cohort: {
                    "parquet_path": freeze["cohorts"][cohort]["parquet_path"],
                    "parquet_sha256": freeze["cohorts"][cohort]["parquet_sha256"],
                    "manifest_path": freeze["cohorts"][cohort]["manifest_path"],
                    "manifest_sha256": freeze["cohorts"][cohort]["manifest_sha256"],
                }
                for cohort in COHORTS
            },
        }
    output["table_1"] = _verify_table_1_assets()
    return output


def _format_count(value: int) -> str:
    """使用英文千位分隔符格式化整数。"""
    return f"{int(value):,}"


def _format_count_percent(numerator: int, denominator: int) -> tuple[str, float]:
    """格式化 n (%)。"""
    percentage = 100.0 * numerator / denominator
    return f"{_format_count(numerator)} ({percentage:.1f})", percentage


def _format_median_iqr(series: pd.Series) -> tuple[str, dict[str, float]]:
    """格式化一位小数的 median (IQR)。"""
    values = series.astype(float)
    median = float(values.median())
    q1 = float(values.quantile(0.25))
    q3 = float(values.quantile(0.75))
    return f"{median:.1f} ({q1:.1f}–{q3:.1f})", {"median": median, "q1": q1, "q3": q3}


def _assert_finite(series: pd.Series, label: str) -> None:
    """拒绝非有限数值。"""
    if not np.isfinite(series.astype(float).to_numpy()).all():
        raise RuntimeError(f"{label}: 出现非有限数值")


def _load_and_validate_state() -> tuple[dict[str, Any], dict[str, Any]]:
    """验证三层冻结输入并建立生成所需的队列状态。"""
    ledger_freeze = analysis_cohort_ledger.verify_analysis_cohort_ledger_freeze()
    final_freeze = final_eligibility.verify_final_eligibility_freeze()
    panel_freeze = analysis_panel.verify_analysis_panel_freeze()
    source_snapshot = _source_snapshot(ledger_freeze, final_freeze, panel_freeze)

    state: dict[str, Any] = {}
    for cohort in COHORTS:
        ledger = pd.read_parquet(ROOT / ledger_freeze["cohorts"][cohort]["parquet_path"])
        final = pd.read_parquet(ROOT / final_freeze["cohorts"][cohort]["parquet_path"])
        panel = pd.read_parquet(ROOT / panel_freeze["cohorts"][cohort]["parquet_path"])

        if ledger.duplicated(["cohort", "person_id"]).any():
            raise RuntimeError(f"{cohort}: Ledger participant 主键重复")
        if final.duplicated(["cohort", "person_id"]).any():
            raise RuntimeError(f"{cohort}: Final Eligibility participant 主键重复")
        if panel.duplicated(["cohort", "person_id", "wave"]).any():
            raise RuntimeError(f"{cohort}: Analysis Panel person-wave 主键重复")
        if not ledger["cohort"].eq(cohort).all() or not final["cohort"].eq(cohort).all():
            raise RuntimeError(f"{cohort}: participant-level cohort 字段不一致")
        if not panel["cohort"].eq(cohort).all():
            raise RuntimeError(f"{cohort}: Analysis Panel cohort 字段不一致")

        ledger_ids = set(ledger["person_id"].astype(str))
        final_ids = set(final["person_id"].astype(str))
        if ledger_ids != final_ids:
            raise RuntimeError(f"{cohort}: Ledger 与 Final Eligibility participant universe 不一致")
        shared = ledger[["person_id", "recovery_status"]].merge(
            final[["person_id", "recovery_status"]],
            on="person_id",
            how="inner",
            validate="one_to_one",
            suffixes=("_ledger", "_final"),
        )
        same_recovery = (
            shared["recovery_status_ledger"].fillna("<NA>")
            == shared["recovery_status_final"].fillna("<NA>")
        )
        if not same_recovery.all():
            raise RuntimeError(f"{cohort}: Ledger 与 Final Eligibility recovery_status 漂移")

        landmarks = panel.loc[panel["fi_time_role"].eq("landmark")].copy()
        later = panel.loc[panel["fi_time_role"].eq("later")].copy()
        if not landmarks.groupby("person_id", observed=True).size().eq(1).all():
            raise RuntimeError(f"{cohort}: base pool 每人必须恰有一个 landmark 行")
        later_counts_base = later.groupby("person_id", observed=True).size()
        if set(landmarks["person_id"].astype(str)) != set(later_counts_base.index.astype(str)):
            raise RuntimeError(f"{cohort}: base pool 每人必须至少有一个 later FI")
        _assert_finite(later["time_years"], f"{cohort}/later time_years")
        if not later["time_years"].gt(0).all():
            raise RuntimeError(f"{cohort}: later time_years 必须大于零")

        final_base_ids = set(
            final.loc[final["analysis_panel_base_eligible"], "person_id"].astype(str)
        )
        panel_base_ids = set(landmarks["person_id"].astype(str))
        if final_base_ids != panel_base_ids:
            raise RuntimeError(f"{cohort}: Final Eligibility base pool 与 Analysis Panel 不一致")
        ledger_base_ids = set(
            ledger.loc[ledger["longitudinal_support_complete"], "person_id"].astype(str)
        )
        if ledger_base_ids != final_base_ids:
            raise RuntimeError(f"{cohort}: Ledger longitudinal support 与 final base pool 不一致")

        primary = panel.loc[panel["primary_complete_case_eligible"]].copy()
        primary_landmarks = primary.loc[primary["fi_time_role"].eq("landmark")].copy()
        primary_later = primary.loc[primary["fi_time_role"].eq("later")].copy()
        final_primary_ids = set(
            final.loc[final["primary_complete_case_eligible"], "person_id"].astype(str)
        )
        if set(primary_landmarks["person_id"].astype(str)) != final_primary_ids:
            raise RuntimeError(f"{cohort}: primary participant set 跨层不一致")
        if not primary_landmarks.groupby("person_id", observed=True).size().eq(1).all():
            raise RuntimeError(f"{cohort}: primary 每人必须恰有一个 landmark 行")
        primary_later_counts = primary_later.groupby("person_id", observed=True).size()
        if set(primary_landmarks["person_id"].astype(str)) != set(
            primary_later_counts.index.astype(str)
        ):
            raise RuntimeError(f"{cohort}: primary 每人必须至少有一个 later FI")

        missing_matrix = landmarks.loc[:, ADJUSTMENT_COLUMNS].isna()
        any_missing = missing_matrix.any(axis=1)
        no_missing_ids = set(landmarks.loc[~any_missing, "person_id"].astype(str))
        if no_missing_ids != final_primary_ids:
            raise RuntimeError(f"{cohort}: base pool adjustment completeness 与 primary flag 不一致")
        for column in COMPLETE_ADJUSTMENT_COLUMNS:
            if bool(missing_matrix[column].any()):
                raise RuntimeError(f"{cohort}: 合同声明完整的变量出现缺失：{column}")

        gates = {str(key): int(value) for key, value in final["final_failed_gate"].value_counts().items()}
        expected = EXPECTED[cohort]
        for gate in (
            "recovery_unavailable",
            "landmark_fi_missing",
            "landmark_timing_unavailable",
            "no_later_fi",
            "no_timed_later_fi",
        ):
            if gates.get(gate, 0) != expected[gate]:
                raise RuntimeError(f"{cohort}/{gate}: gate count 与合同不一致")
        if len(final) != expected["shock_cohort"]:
            raise RuntimeError(f"{cohort}: shock cohort count 与合同不一致")
        if len(landmarks) != expected["mi_pool"]:
            raise RuntimeError(f"{cohort}: MI support pool count 与合同不一致")
        if len(primary_landmarks) != expected["complete_case"]:
            raise RuntimeError(f"{cohort}: complete-case count 与合同不一致")
        if len(primary_later) != expected["later_rows"]:
            raise RuntimeError(f"{cohort}: primary later FI count 与合同不一致")
        if int(any_missing.sum()) != expected["any_missing"]:
            raise RuntimeError(f"{cohort}: any adjustment missing count 与合同不一致")
        for column, expected_n in expected["missing"].items():
            if int(missing_matrix[column].sum()) != expected_n:
                raise RuntimeError(f"{cohort}/{column}: missing count 与合同不一致")
        for group in GROUPS:
            observed = int(primary_landmarks["recovery_status"].eq(group).sum())
            if observed != expected["groups"][group]:
                raise RuntimeError(f"{cohort}/{group}: recovery group count 与合同不一致")

        follow = (
            primary_later.groupby(["person_id", "recovery_status"], observed=True)
            .agg(
                post_landmark_assessments=("wave", "size"),
                follow_up_duration_years=("time_years", "max"),
            )
            .reset_index()
        )
        for group in GROUPS:
            group_follow = follow.loc[follow["recovery_status"].eq(group)]
            observed_distribution = {
                "1": int(group_follow["post_landmark_assessments"].eq(1).sum()),
                "2": int(group_follow["post_landmark_assessments"].eq(2).sum()),
                "3": int(group_follow["post_landmark_assessments"].eq(3).sum()),
                "4_plus": int(group_follow["post_landmark_assessments"].ge(4).sum()),
            }
            if observed_distribution != expected["assessment_distribution"][group]:
                raise RuntimeError(f"{cohort}/{group}: assessment distribution 与合同不一致")
            support_10 = int(group_follow["follow_up_duration_years"].ge(10).sum())
            if support_10 != expected["support_10_years"][group]:
                raise RuntimeError(f"{cohort}/{group}: 10-year support 与合同不一致")

        state[cohort] = {
            "ledger": ledger,
            "final": final,
            "panel": panel,
            "landmarks": landmarks,
            "later": later,
            "primary_landmarks": primary_landmarks,
            "primary_later": primary_later,
            "follow": follow,
            "missing_matrix": missing_matrix,
            "any_missing": any_missing,
            "gates": gates,
        }
    return state, source_snapshot


def build_table_data() -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any], dict[str, Any]]:
    """构建全精度长表、显示规格、诊断与输入快照。"""
    state, source_snapshot = _load_and_validate_state()
    columns = [
        {
            "key": f"{cohort}__{group}",
            "cohort": cohort,
            "recovery_status": group,
            "label": GROUP_LABELS[group],
        }
        for cohort in COHORTS
        for group in GROUPS
    ]
    display_rows: list[dict[str, Any]] = []
    summary_records: list[dict[str, Any]] = []
    display_order = 0
    summary_order = 0

    def add_heading(label: str, row_type: str) -> None:
        nonlocal display_order
        display_order += 1
        display_rows.append(
            {
                "display_order": display_order,
                "row_id": f"{row_type}_{display_order}",
                "row_type": row_type,
                "label": label,
                "indent": 0,
                "values": {},
            }
        )

    def add_group_row(
        *,
        row_id: str,
        section: str,
        block: str,
        label: str,
        statistic_type: str,
        source_column: str,
        source_time: str,
        indent: int = 0,
        target: int | str | None = None,
        threshold: int | None = None,
        display_in_docx: bool = True,
    ) -> None:
        nonlocal display_order, summary_order
        values: dict[str, str] = {}
        for cohort in COHORTS:
            cohort_state = state[cohort]
            for group in GROUPS:
                summary_order += 1
                key = f"{cohort}__{group}"
                landmarks = cohort_state["primary_landmarks"].loc[
                    cohort_state["primary_landmarks"]["recovery_status"].eq(group)
                ]
                later = cohort_state["primary_later"].loc[
                    cohort_state["primary_later"]["recovery_status"].eq(group)
                ]
                follow = cohort_state["follow"].loc[
                    cohort_state["follow"]["recovery_status"].eq(group)
                ]
                denominator = int(len(landmarks))
                record: dict[str, Any] = {
                    "summary_order": summary_order,
                    "row_id": row_id,
                    "section": section,
                    "block": block,
                    "characteristic": label,
                    "cohort": cohort,
                    "recovery_status": group,
                    "display_group_label": GROUP_LABELS[group],
                    "sample_universe": "primary_complete_case_analytic_sample",
                    "statistic_type": statistic_type,
                    "source_asset": "analysis_panel_v1",
                    "source_column": source_column,
                    "source_time": source_time,
                    "threshold_years": threshold,
                    "numerator": None,
                    "denominator": None,
                    "value": None,
                    "percentage": None,
                    "median": None,
                    "q1": None,
                    "q3": None,
                    "display_value": None,
                    "display_in_docx": display_in_docx,
                }
                if statistic_type == "participant_count":
                    value = denominator
                    display = _format_count(value)
                    record.update({"value": value})
                elif statistic_type == "total_later_count":
                    value = int(len(later))
                    display = _format_count(value)
                    record.update({"value": value})
                elif statistic_type == "assessment_category":
                    counts = follow["post_landmark_assessments"]
                    if target == "4_plus":
                        numerator = int(counts.ge(4).sum())
                    else:
                        numerator = int(counts.eq(int(target)).sum())
                    display, percentage = _format_count_percent(numerator, denominator)
                    record.update(
                        {"numerator": numerator, "denominator": denominator, "percentage": percentage}
                    )
                elif statistic_type == "follow_up_median_iqr":
                    display, stats = _format_median_iqr(follow["follow_up_duration_years"])
                    record.update(stats)
                    record["denominator"] = denominator
                elif statistic_type == "threshold_participants":
                    assert threshold is not None
                    numerator = int(follow["follow_up_duration_years"].ge(threshold).sum())
                    display, percentage = _format_count_percent(numerator, denominator)
                    record.update(
                        {"numerator": numerator, "denominator": denominator, "percentage": percentage}
                    )
                elif statistic_type == "threshold_observations":
                    assert threshold is not None
                    value = int(later["time_years"].ge(threshold).sum())
                    display = _format_count(value)
                    record.update({"value": value})
                else:
                    raise RuntimeError(f"未知 group statistic_type：{statistic_type}")
                record["display_value"] = display
                summary_records.append(record)
                values[key] = display
        if display_in_docx:
            display_order += 1
            display_rows.append(
                {
                    "display_order": display_order,
                    "row_id": row_id,
                    "row_type": "data_group",
                    "label": label,
                    "indent": indent,
                    "values": values,
                }
            )

    def add_cohort_row(
        *,
        row_id: str,
        block: str,
        label: str,
        statistic_type: str,
        source_column: str,
        display_in_docx: bool = True,
    ) -> None:
        nonlocal display_order, summary_order
        values: dict[str, str] = {}
        for cohort in COHORTS:
            summary_order += 1
            cohort_state = state[cohort]
            expected = EXPECTED[cohort]
            shock_n = int(len(cohort_state["final"]))
            mi_n = int(len(cohort_state["landmarks"]))
            record: dict[str, Any] = {
                "summary_order": summary_order,
                "row_id": row_id,
                "section": "Missingness and analytic exclusions",
                "block": block,
                "characteristic": label,
                "cohort": cohort,
                "recovery_status": None,
                "display_group_label": None,
                "sample_universe": (
                    "shock_experienced_cohort"
                    if block == "Sequential analytic exclusions"
                    else "mi_support_pool"
                ),
                "statistic_type": statistic_type,
                "source_asset": (
                    "final_eligibility_v1"
                    if block == "Sequential analytic exclusions"
                    else "analysis_panel_v1"
                ),
                "source_column": source_column,
                "source_time": "participant",
                "threshold_years": None,
                "numerator": None,
                "denominator": None,
                "value": None,
                "percentage": None,
                "median": None,
                "q1": None,
                "q3": None,
                "display_value": None,
                "display_in_docx": display_in_docx,
            }
            if statistic_type == "shock_cohort_count":
                display = _format_count(shock_n)
                record["value"] = shock_n
            elif statistic_type == "gate_count_percent":
                numerator = int(cohort_state["gates"].get(source_column, 0))
                display, percentage = _format_count_percent(numerator, shock_n)
                record.update(
                    {"numerator": numerator, "denominator": shock_n, "percentage": percentage}
                )
            elif statistic_type == "mi_pool_count_percent":
                display, percentage = _format_count_percent(mi_n, shock_n)
                record.update({"numerator": mi_n, "denominator": shock_n, "percentage": percentage})
            elif statistic_type == "mi_pool_count":
                display = _format_count(mi_n)
                record.update({"value": mi_n, "numerator": mi_n})
            elif statistic_type == "missing_count_percent":
                numerator = int(cohort_state["missing_matrix"][source_column].sum())
                display, percentage = _format_count_percent(numerator, mi_n)
                record.update({"numerator": numerator, "denominator": mi_n, "percentage": percentage})
            elif statistic_type == "any_missing_count_percent":
                numerator = int(cohort_state["any_missing"].sum())
                display, percentage = _format_count_percent(numerator, mi_n)
                record.update({"numerator": numerator, "denominator": mi_n, "percentage": percentage})
            elif statistic_type == "complete_case_count_percent":
                numerator = expected["complete_case"]
                display, percentage = _format_count_percent(numerator, mi_n)
                record.update({"numerator": numerator, "denominator": mi_n, "percentage": percentage})
            else:
                raise RuntimeError(f"未知 cohort statistic_type：{statistic_type}")
            record["display_value"] = display
            summary_records.append(record)
            values[cohort] = display
        if display_in_docx:
            display_order += 1
            display_rows.append(
                {
                    "display_order": display_order,
                    "row_id": row_id,
                    "row_type": "data_cohort_merged",
                    "label": label,
                    "indent": 0,
                    "values": values,
                }
            )

    add_group_row(
        row_id="participants",
        section="Follow-up support",
        block="Core support",
        label="Participants, n",
        statistic_type="participant_count",
        source_column="person_id",
        source_time="participant",
    )
    add_group_row(
        row_id="total_post_landmark_fi_observations",
        section="Follow-up support",
        block="Core support",
        label="Total post-landmark FI observations, n",
        statistic_type="total_later_count",
        source_column="fi_primary",
        source_time="t+2 onward",
    )
    add_heading("Post-landmark FI assessments per participant, n (%)", "category_header")
    for target, label, row_id in (
        (1, "1 assessment", "assessments_1"),
        (2, "2 assessments", "assessments_2"),
        (3, "3 assessments", "assessments_3"),
        ("4_plus", "≥4 assessments", "assessments_4_plus"),
    ):
        add_group_row(
            row_id=row_id,
            section="Follow-up support",
            block="Assessment frequency",
            label=label,
            statistic_type="assessment_category",
            source_column="post_landmark_assessments",
            source_time="t+2 onward",
            indent=1,
            target=target,
        )
    add_group_row(
        row_id="follow_up_duration",
        section="Follow-up support",
        block="Core support",
        label="Follow-up duration, years, median (IQR)",
        statistic_type="follow_up_median_iqr",
        source_column="time_years",
        source_time="landmark to last valid FI",
    )
    add_heading("Observed follow-up support, n (%)", "category_header")
    for threshold in THRESHOLDS:
        add_group_row(
            row_id=f"participants_reaching_{threshold}_years",
            section="Follow-up support",
            block="Observed time support",
            label=f"Participants reaching ≥{threshold} years",
            statistic_type="threshold_participants",
            source_column="time_years",
            source_time="actual elapsed years from landmark",
            indent=1,
            threshold=threshold,
        )
        add_group_row(
            row_id=f"observations_at_or_beyond_{threshold}_years",
            section="Follow-up support",
            block="Observed time support",
            label=f"FI observations at or beyond {threshold} years, n",
            statistic_type="threshold_observations",
            source_column="time_years",
            source_time="actual elapsed years from landmark",
            threshold=threshold,
            display_in_docx=False,
        )

    add_heading("Panel B. Adjustment-variable missingness and MI support", "section")
    add_cohort_row(
        row_id="shock_experienced_cohort",
        block="Sequential analytic exclusions",
        label="Shock-experienced cohort, n",
        statistic_type="shock_cohort_count",
        source_column="person_id",
        display_in_docx=False,
    )
    for row_id, label, gate in (
        ("recovery_status_unavailable", "Recovery status unavailable", "recovery_unavailable"),
        ("landmark_fi_unavailable", "Landmark FI unavailable", "landmark_fi_missing"),
        ("no_valid_post_landmark_fi", "No valid post-landmark FI", "no_later_fi"),
    ):
        add_cohort_row(
            row_id=row_id,
            block="Sequential analytic exclusions",
            label=label,
            statistic_type="gate_count_percent",
            source_column=gate,
            display_in_docx=False,
        )
    add_cohort_row(
        row_id="mi_support_pool",
        block="MI support",
        label="MI support pool, n",
        statistic_type="mi_pool_count",
        source_column="analysis_panel_base_eligible",
    )

    add_heading(
        "Adjustment-variable missingness, n (% of MI support pool)",
        "category_header",
    )
    for column, label in DISPLAYED_MISSINGNESS:
        add_cohort_row(
            row_id=f"missing_{column}",
            block="Adjustment missingness",
            label=label,
            statistic_type="missing_count_percent",
            source_column=column,
        )
    add_cohort_row(
        row_id="any_missing_adjustment",
        block="Adjustment missingness",
        label="Any missing prespecified adjustment variable",
        statistic_type="any_missing_count_percent",
        source_column="any_adjustment_missing",
    )
    add_cohort_row(
        row_id="primary_complete_case_sample",
        block="Adjustment missingness",
        label="Primary complete-case analytic sample, n (% of MI support pool)",
        statistic_type="complete_case_count_percent",
        source_column="primary_complete_case_eligible",
    )

    summary = pd.DataFrame.from_records(summary_records)
    if "person_id" in summary.columns:
        raise RuntimeError("聚合 CSV 不得包含 person_id 列")

    diagnostics = {
        cohort: {
            "shock_experienced_cohort": EXPECTED[cohort]["shock_cohort"],
            "mi_support_pool": EXPECTED[cohort]["mi_pool"],
            "primary_complete_case_sample": EXPECTED[cohort]["complete_case"],
            "primary_post_landmark_fi_observations": EXPECTED[cohort]["later_rows"],
            "timing_gate_counts": {
                "landmark_timing_unavailable": state[cohort]["gates"].get(
                    "landmark_timing_unavailable", 0
                ),
                "no_timed_later_fi": state[cohort]["gates"].get("no_timed_later_fi", 0),
            },
            "hidden_threshold_observations": {
                str(threshold): {
                    group: int(
                        state[cohort]["primary_later"].loc[
                            state[cohort]["primary_later"]["recovery_status"].eq(group),
                            "time_years",
                        ].ge(threshold).sum()
                    )
                    for group in GROUPS
                }
                for threshold in THRESHOLDS
            },
        }
        for cohort in COHORTS
    }

    spec = {
        "schema_version": "table_s5_display_spec_v6",
        "caption": CAPTION,
        "notes": NOTES,
        "table": {
            "external_heading": PANEL_A_HEADING,
            "cohort_order": list(COHORTS),
            "group_order": list(GROUPS),
            "columns": columns,
            "rows": display_rows,
        },
        "figure_3_support": {
            "candidate_display_range_years": [0, 10],
            "final_range_locked": False,
            "decision_boundary": "confirm after frozen Table S5 support audit before Figure 3 release",
            "thresholds_reported_years": list(THRESHOLDS),
            "threshold_observation_counts_in_docx": False,
            "threshold_observation_counts_in_summary_and_validation": True,
        },
        "layout": {
            "paper": "A4",
            "orientation": "landscape",
            "font": "Times New Roman",
            "body_font_pt": 8.5,
            "note_font_pt": 8.0,
            "minimum_font_pt": 8.0,
            "maximum_pages": 2,
            "vertical_gridlines": False,
            "background_shading": False,
            "native_table_count": 2,
            "panel_b_columns": ["Characteristic", "HRS", "ELSA", "KLoSA"],
        },
        "reporting": {
            "weighted": False,
            "follow_up_sample": "primary_complete_case_eligible",
            "follow_up_percentage_denominator": "cohort by recovery-status group",
            "sequential_exclusion_denominator": "shock-experienced cohort",
            "selection_flow_displayed_in_docx": False,
            "selection_flow_reference": "Supplementary Figure S1",
            "adjustment_missingness_denominator": "MI support pool",
            "post_landmark_excludes_landmark": True,
            "actual_interview_timing_only": True,
            "mi_performed_for_table": False,
        },
    }
    return summary, spec, diagnostics, source_snapshot


def _compare_table_1_anchors(summary: pd.DataFrame) -> list[dict[str, Any]]:
    """将重叠 anchor 与冻结 Table 1 作全精度及显示核对。"""
    _verify_table_1_assets()
    table_1 = pd.read_csv(TABLE_1_SUMMARY_PATH, encoding="utf-8-sig")
    checks: list[dict[str, Any]] = []
    for cohort in COHORTS:
        for group in GROUPS:
            filters = summary["cohort"].eq(cohort) & summary["recovery_status"].eq(group)
            table_1_filters = table_1["cohort"].eq(cohort) & table_1["recovery_status"].eq(group)

            s5_participant = summary.loc[filters & summary["row_id"].eq("participants")].iloc[0]
            t1_participant = table_1.loc[
                table_1_filters & table_1["row_id"].eq("participants")
            ].iloc[0]
            s5_n = int(s5_participant["value"])
            t1_n = int(t1_participant["numerator"])
            if s5_n != t1_n or str(s5_participant["display_value"]) != _format_count(t1_n):
                raise RuntimeError(f"{cohort}/{group}: participant anchor 与 Table 1 不一致")

            s5_obs = summary.loc[
                filters & summary["row_id"].eq("total_post_landmark_fi_observations")
            ].iloc[0]
            t1_obs = table_1.loc[
                table_1_filters
                & table_1["row_id"].eq("total_post_landmark_fi_observations")
            ].iloc[0]
            if int(s5_obs["value"]) != int(t1_obs["value"]) or str(
                s5_obs["display_value"]
            ) != str(t1_obs["display_value"]):
                raise RuntimeError(f"{cohort}/{group}: later FI anchor 与 Table 1 不一致")

            s5_follow = summary.loc[
                filters & summary["row_id"].eq("follow_up_duration")
            ].iloc[0]
            t1_follow = table_1.loc[
                table_1_filters & table_1["row_id"].eq("follow_up_duration")
            ].iloc[0]
            for statistic in ("q1", "median", "q3"):
                if not np.isclose(
                    float(s5_follow[statistic]),
                    float(t1_follow[statistic]),
                    rtol=0.0,
                    atol=1e-12,
                ):
                    raise RuntimeError(
                        f"{cohort}/{group}: follow-up {statistic} anchor 与 Table 1 不一致"
                    )
            if str(s5_follow["display_value"]) != str(t1_follow["display_value"]):
                raise RuntimeError(f"{cohort}/{group}: follow-up display 与 Table 1 不一致")
            checks.append(
                {
                    "cohort": cohort,
                    "recovery_status": group,
                    "participant_count": "PASS",
                    "post_landmark_fi_observations": "PASS",
                    "follow_up_duration_q1_median_q3": "PASS",
                    "display_values": "PASS",
                }
            )
    return checks


def _validate_summary(summary: pd.DataFrame) -> dict[str, Any]:
    """验证样本闭合、assessment 分布与时间支持单调性。"""
    if "person_id" in summary.columns:
        raise RuntimeError("summary_long 不得包含 participant identifier")
    monotonic_checks: list[dict[str, Any]] = []
    for cohort in COHORTS:
        expected = EXPECTED[cohort]
        shock = summary.loc[
            summary["cohort"].eq(cohort) & summary["row_id"].eq("shock_experienced_cohort")
        ].iloc[0]
        if int(shock["value"]) != expected["shock_cohort"]:
            raise RuntimeError(f"{cohort}: shock cohort summary 漂移")
        for row_id, expected_value in (
            ("recovery_status_unavailable", expected["recovery_unavailable"]),
            ("landmark_fi_unavailable", expected["landmark_fi_missing"]),
            ("no_valid_post_landmark_fi", expected["no_later_fi"]),
        ):
            row = summary.loc[
                summary["cohort"].eq(cohort) & summary["row_id"].eq(row_id)
            ].iloc[0]
            if int(row["numerator"]) != expected_value:
                raise RuntimeError(f"{cohort}/{row_id}: sequential exclusion 漂移")
        mi = summary.loc[
            summary["cohort"].eq(cohort) & summary["row_id"].eq("mi_support_pool")
        ].iloc[0]
        any_missing = summary.loc[
            summary["cohort"].eq(cohort) & summary["row_id"].eq("any_missing_adjustment")
        ].iloc[0]
        complete = summary.loc[
            summary["cohort"].eq(cohort) & summary["row_id"].eq("primary_complete_case_sample")
        ].iloc[0]
        if int(mi["numerator"]) - int(any_missing["numerator"]) != int(complete["numerator"]):
            raise RuntimeError(f"{cohort}: MI pool - any missing 未闭合到 complete case")
        displayed_missing_union_upper = int(
            summary.loc[
                summary["cohort"].eq(cohort)
                & summary["statistic_type"].eq("missing_count_percent"),
                "numerator",
            ].sum()
        )
        if displayed_missing_union_upper < int(any_missing["numerator"]):
            raise RuntimeError(f"{cohort}: missingness union 大于各变量计数之和")

        for group in GROUPS:
            group_filter = summary["cohort"].eq(cohort) & summary["recovery_status"].eq(group)
            participant_n = int(
                summary.loc[group_filter & summary["row_id"].eq("participants"), "value"].iloc[0]
            )
            assessment_rows = summary.loc[
                group_filter & summary["statistic_type"].eq("assessment_category")
            ]
            if int(assessment_rows["numerator"].sum()) != participant_n:
                raise RuntimeError(f"{cohort}/{group}: assessment categories 未归一")
            participant_support = summary.loc[
                group_filter & summary["statistic_type"].eq("threshold_participants")
            ].sort_values("threshold_years")
            observation_support = summary.loc[
                group_filter & summary["statistic_type"].eq("threshold_observations")
            ].sort_values("threshold_years")
            if not participant_support["numerator"].astype(int).is_monotonic_decreasing:
                raise RuntimeError(f"{cohort}/{group}: participant support 未单调不增")
            if not observation_support["value"].astype(int).is_monotonic_decreasing:
                raise RuntimeError(f"{cohort}/{group}: observation support 未单调不增")
            ten_year_n = int(
                participant_support.loc[participant_support["threshold_years"].eq(10), "numerator"].iloc[0]
            )
            if ten_year_n != expected["support_10_years"][group]:
                raise RuntimeError(f"{cohort}/{group}: 10-year support 漂移")
            monotonic_checks.append(
                {
                    "cohort": cohort,
                    "recovery_status": group,
                    "participant_support": "PASS",
                    "observation_support": "PASS",
                }
            )
    anchor_checks = _compare_table_1_anchors(summary)
    return {
        "sample_flow_reconciliation": "PASS",
        "missingness_union_reconciliation": "PASS",
        "assessment_distribution_reconciliation": "PASS",
        "threshold_monotonicity": "PASS",
        "threshold_monotonicity_checks": monotonic_checks,
        "table_1_anchor_consistency": "PASS",
        "table_1_anchor_checks": anchor_checks,
    }


def _latex_escape(text: str) -> str:
    """转义 S5 表格所需的 LaTeX 特殊字符。"""
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }
    escaped = "".join(replacements.get(character, character) for character in text)
    return escaped.replace(">=", r"$\geq$")


def _build_tex(spec: dict[str, Any]) -> str:
    """从同一 display spec 生成 Supplementary Materials 的 LaTeX 表格。"""
    caption = spec["caption"].removeprefix("Supplementary Table S5. ")
    panel_a_heading = _latex_escape(spec["table"]["external_heading"])
    rows = spec["table"]["rows"]
    panel_b_index = next(
        (index for index, row in enumerate(rows) if row["row_type"] == "section"),
        None,
    )
    if panel_b_index is None:
        raise RuntimeError("display spec 缺少 Panel B 分隔行")
    panel_a_rows = rows[:panel_b_index]
    panel_b_heading = _latex_escape(rows[panel_b_index]["label"])
    panel_b_rows = rows[panel_b_index + 1 :]
    lines = [
        r"\begin{landscape}",
        r"\fontsize{8}{9.4}\selectfont",
        r"\setlength{\tabcolsep}{2.2pt}",
        r"\renewcommand{\arraystretch}{1.02}",
        "",
        rf"\captionof{{table}}{{{_latex_escape(caption)}}}\label{{tab:s5_followup_missingness}}",
        r"\vspace{0.25em}",
        rf"\noindent\textbf{{{panel_a_heading}}}\par",
        r"\vspace{0.45em}",
        r"\begin{tabular}{@{}L{5.10cm}*{6}{>{\raggedleft\arraybackslash}p{2.85cm}}@{}}",
        r"\toprule",
        r"Characteristic & \multicolumn{2}{c}{HRS} & \multicolumn{2}{c}{ELSA} & \multicolumn{2}{c}{KLoSA} \\",
        r" & \multicolumn{1}{c}{Substantial recovery} & \multicolumn{1}{c}{Persistent/limited recovery} & \multicolumn{1}{c}{Substantial recovery} & \multicolumn{1}{c}{Persistent/limited recovery} & \multicolumn{1}{c}{Substantial recovery} & \multicolumn{1}{c}{Persistent/limited recovery} \\",
        r"\midrule",
    ]
    for row in panel_a_rows:
        label = _latex_escape(row["label"])
        if row["row_type"] == "block":
            lines.extend([r"\midrule", rf"\multicolumn{{7}}{{@{{}}l}}{{\textbf{{{label}}}}}\\"])
        elif row["row_type"] == "category_header":
            lines.append(rf"\textit{{{label}}} & & & & & & \\")
        elif row["row_type"] == "data_cohort_merged":
            values = row["values"]
            lines.append(
                f"{label} & "
                + " & ".join(
                    rf"\multicolumn{{2}}{{c}}{{{_latex_escape(str(values[cohort]))}}}"
                    for cohort in COHORTS
                )
                + r" \\"
            )
        else:
            if row.get("indent"):
                label = rf"\hspace{{0.9em}}{label}"
            values = [
                _latex_escape(str(row["values"][f"{cohort}__{group}"]))
                for cohort in COHORTS
                for group in GROUPS
            ]
            lines.append(f"{label} & " + " & ".join(values) + r" \\")
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            "",
            r"\vspace{0.7em}",
            rf"\noindent\textbf{{{panel_b_heading}}}\par",
            r"\vspace{0.45em}",
            r"\begin{tabular}{@{}L{10.80cm}*{3}{>{\raggedleft\arraybackslash}p{3.80cm}}@{}}",
            r"\toprule",
            r"Characteristic & \multicolumn{1}{c}{HRS} & \multicolumn{1}{c}{ELSA} & \multicolumn{1}{c}{KLoSA} \\",
            r"\midrule",
        ]
    )
    for row in panel_b_rows:
        label = _latex_escape(row["label"])
        if row["row_type"] == "category_header":
            lines.append(rf"\textit{{{label}}} & & & \\")
        else:
            values = [_latex_escape(str(row["values"][cohort])) for cohort in COHORTS]
            lines.append(f"{label} & " + " & ".join(values) + r" \\")
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            "",
            rf"\parbox{{\linewidth}}{{\raggedright\fontsize{{8}}{{9.2}}\selectfont\textit{{Notes:}} {_latex_escape(spec['notes'])}}}",
            r"\end{landscape}",
            "",
        ]
    )
    return "\n".join(lines)


def _validate_docx(path: Path, expected_panel_a_rows: int, expected_panel_b_rows: int) -> dict[str, Any]:
    """验证 Word 两个原生子表、表头结构与禁用样式。"""
    if not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError("缺少非空 table_s5_editable.docx")
    with zipfile.ZipFile(path) as archive:
        if "word/document.xml" not in archive.namelist():
            raise RuntimeError("DOCX 缺少 word/document.xml")
        document_xml = archive.read("word/document.xml")
    root = ET.fromstring(document_xml)
    ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    word_ns = f"{{{ns['w']}}}"
    tables = root.findall(".//w:tbl", ns)
    if len(tables) != 2:
        raise RuntimeError(f"DOCX 必须恰有两个原生子表，实际 {len(tables)}")
    panel_a_table, panel_b_table = tables
    panel_a_columns = panel_a_table.findall("./w:tblGrid/w:gridCol", ns)
    panel_b_columns = panel_b_table.findall("./w:tblGrid/w:gridCol", ns)
    panel_a_table_rows = panel_a_table.findall("./w:tr", ns)
    panel_b_table_rows = panel_b_table.findall("./w:tr", ns)
    if len(panel_a_columns) != 7 or len(panel_a_table_rows) != expected_panel_a_rows:
        raise RuntimeError(
            "DOCX Panel A 结构不一致："
            f"columns={len(panel_a_columns)}, rows={len(panel_a_table_rows)}"
        )
    if len(panel_b_columns) != 4 or len(panel_b_table_rows) != expected_panel_b_rows:
        raise RuntimeError(
            "DOCX Panel B 结构不一致："
            f"columns={len(panel_b_columns)}, rows={len(panel_b_table_rows)}"
        )
    if len(panel_a_table.findall(".//w:gridSpan", ns)) < 3:
        raise RuntimeError("DOCX Panel A 缺少 grouped header")
    if len(panel_a_table.findall(".//w:vMerge", ns)) < 2:
        raise RuntimeError("DOCX 缺少 Characteristic 纵向合并表头")
    for index in (0, 1):
        if len(panel_a_table_rows[index].findall("./w:trPr/w:tblHeader", ns)) != 1:
            raise RuntimeError(f"DOCX Panel A 第 {index + 1} 行未设置重复表头")
    if len(panel_b_table_rows[0].findall("./w:trPr/w:tblHeader", ns)) != 1:
        raise RuntimeError("DOCX Panel B 表头未设置为重复表头")
    if root.findall(".//w:drawing", ns) or root.findall(".//w:pict", ns):
        raise RuntimeError("DOCX 不得包含图片化表格")

    full_text = "".join(node.text or "" for node in root.findall(".//w:t", ns))
    tables_text = "".join(node.text or "" for table in tables for node in table.findall(".//w:t", ns))
    if PANEL_A_HEADING in tables_text or "Panel B. Adjustment-variable missingness and MI support" in tables_text:
        raise RuntimeError("DOCX 的 Panel 标题必须位于对应原生表格之外")
    for required in (
        CAPTION,
        PANEL_A_HEADING,
        "Panel B. Adjustment-variable missingness and MI support",
        "MI support pool, n",
        "Adjustment-variable missingness, n (% of MI support pool)",
        "Primary complete-case analytic sample, n (% of MI support pool)",
        "Participants reaching ≥10 years",
        "MI, multiple imputation",
        "Sample selection from the shock-experienced cohort to the MI support pool is shown in Supplementary Figure S1",
    ):
        if required not in full_text:
            raise RuntimeError(f"DOCX 缺少必需文本：{required}")
    for prohibited in (
        "FI observations at or beyond 2 years",
        "Sequential analytic exclusions, n (% of shock-experienced cohort)",
        "Shock-experienced cohort, n",
        "Recovery status unavailable",
        "Landmark FI unavailable",
        "No valid post-landmark FI",
        "Landmark timing unavailable",
        "No timed post-landmark FI",
        "P value",
        "SMD",
        "Pooled",
        "pooled",
    ):
        if prohibited in full_text:
            raise RuntimeError(f"DOCX 出现禁用文本：{prohibited}")

    vertical_violations = []
    for tag in ("left", "right", "insideV"):
        for node in root.findall(f".//w:{tag}", ns):
            value = node.attrib.get(f"{word_ns}val", "")
            if value not in ("nil", "none", ""):
                vertical_violations.append(f"{tag}:{value}")
    if vertical_violations:
        raise RuntimeError("DOCX 出现竖线：" + ", ".join(vertical_violations[:10]))
    shading = []
    for node in root.findall(".//w:shd", ns):
        fill = node.attrib.get(f"{word_ns}fill", "")
        if fill.lower() not in ("", "auto", "ffffff"):
            shading.append(fill)
    if shading:
        raise RuntimeError("DOCX 出现背景色：" + ", ".join(shading[:10]))
    sizes = [
        float(node.attrib[f"{word_ns}val"]) / 2.0
        for node in root.findall(".//w:sz", ns)
        if f"{word_ns}val" in node.attrib
    ]
    if sizes and min(sizes) < 8.0:
        raise RuntimeError(f"DOCX 最小字号低于 8 pt：{min(sizes)}")
    section = root.find(".//w:sectPr", ns)
    page_size = None if section is None else section.find("./w:pgSz", ns)
    if page_size is None or page_size.attrib.get(f"{word_ns}orient") != "landscape":
        raise RuntimeError("DOCX 必须为 landscape")
    page_width = int(page_size.attrib.get(f"{word_ns}w", "0"))
    page_height = int(page_size.attrib.get(f"{word_ns}h", "0"))
    if page_width <= page_height:
        raise RuntimeError("DOCX landscape 页面尺寸无效")
    return {
        "native_table_count": 2,
        "panel_a_columns": len(panel_a_columns),
        "panel_a_rows": len(panel_a_table_rows),
        "panel_b_columns": len(panel_b_columns),
        "panel_b_rows": len(panel_b_table_rows),
        "grid_span_count": sum(len(table.findall(".//w:gridSpan", ns)) for table in tables),
        "vertical_merge_count": sum(len(table.findall(".//w:vMerge", ns)) for table in tables),
        "image_count": 0,
        "minimum_explicit_font_pt": min(sizes) if sizes else None,
        "orientation": "landscape",
        "vertical_gridlines": False,
        "background_shading": False,
    }


def _validate_supplement_integration(output_dir: Path) -> dict[str, Any]:
    """验证 LaTeX source 已接入主补充材料并成功编译。"""
    formal_tex = output_dir / TEX_NAME
    if not formal_tex.is_file() or not SUPPLEMENT_TABLE_PATH.is_file():
        raise RuntimeError("Supplementary Materials 缺少正式 Table S5 LaTeX source")
    if _sha256_file(formal_tex) != _sha256_file(SUPPLEMENT_TABLE_PATH):
        raise RuntimeError("Supplementary Materials 的 table_s5.tex 与正式输出不一致")
    main_text = SUPPLEMENT_MAIN_PATH.read_text(encoding="utf-8")
    for required in (r"\setcounter{table}{4}", r"\input{tables/table_s5}"):
        if required not in main_text:
            raise RuntimeError(f"supplementary main.tex 缺少 Table S5 接入：{required}")
    if not SUPPLEMENT_PDF_PATH.is_file() or SUPPLEMENT_PDF_PATH.stat().st_size == 0:
        raise RuntimeError("缺少重新编译的 Supplementary_Materials.pdf")
    pdftotext = shutil.which("pdftotext")
    if pdftotext is None:
        raise RuntimeError("找不到 bundled/system pdftotext，无法验证 Supplementary PDF")
    extracted = subprocess.run(
        [pdftotext, "-layout", str(SUPPLEMENT_PDF_PATH), "-"],
        check=True,
        capture_output=True,
    ).stdout.decode("utf-8", errors="replace")
    for required in (
        "Supplementary Table S5",
        "Follow-up availability and missing data by cohort",
        "Participants reaching",
        "MI support pool, n",
    ):
        if required not in extracted:
            raise RuntimeError(f"Supplementary PDF 缺少 Table S5 文本：{required}")
    forbidden_suffixes = (".aux", ".log", ".out", ".fls", ".fdb_latexmk", ".xdv")
    retained_aux = [
        _relative(path)
        for path in SUPPLEMENT_DIR.iterdir()
        if path.is_file() and path.suffix.lower() in forbidden_suffixes
    ]
    if retained_aux:
        raise RuntimeError("Supplementary Materials 根目录残留 LaTeX 临时文件")
    return {
        "status": "PASS",
        "source_path": _relative(SUPPLEMENT_TABLE_PATH),
        "source_sha256": _sha256_file(SUPPLEMENT_TABLE_PATH),
        "main_path": _relative(SUPPLEMENT_MAIN_PATH),
        "main_sha256_at_validation": _sha256_file(SUPPLEMENT_MAIN_PATH),
        "pdf_path": _relative(SUPPLEMENT_PDF_PATH),
        "pdf_sha256_at_validation": _sha256_file(SUPPLEMENT_PDF_PATH),
        "pdf_bytes": SUPPLEMENT_PDF_PATH.stat().st_size,
        "pdf_text_checks": "PASS",
        "retained_auxiliary_files": [],
    }


def prepare(staging_dir: Path) -> None:
    """生成聚合数据、显示规格、LaTeX source 与输入快照。"""
    staging_dir.mkdir(parents=True, exist_ok=False)
    summary, spec, diagnostics, source_snapshot = build_table_data()
    summary_checks = _validate_summary(summary)
    summary.to_csv(
        staging_dir / SUMMARY_NAME,
        index=False,
        encoding="utf-8-sig",
        lineterminator="\n",
        float_format="%.17g",
    )
    _write_json(staging_dir / SPEC_NAME, spec)
    (staging_dir / TEX_NAME).write_text(_build_tex(spec), encoding="utf-8")
    context = {
        "schema_version": "table_s5_build_context_v1",
        "generated_utc": datetime.now(UTC).isoformat(),
        "source_snapshot": source_snapshot,
        "protected_snapshot_before": _protected_snapshot(),
        "raw_snapshot_before": _raw_snapshot(),
        "diagnostics": diagnostics,
        "summary_checks": summary_checks,
        "git": _git_state(),
        "software": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "pyarrow": pyarrow.__version__,
        },
        "builders": {
            "python": {"path": _relative(Path(__file__)), "sha256": _sha256_file(Path(__file__))},
            "node": {
                "path": _relative(Path(__file__).with_name("build_table_s5_docx.mjs")),
                "sha256": _sha256_file(Path(__file__).with_name("build_table_s5_docx.mjs")),
            },
            "runner": {
                "path": _relative(Path(__file__).with_name("run_table_s5.ps1")),
                "sha256": _sha256_file(Path(__file__).with_name("run_table_s5.ps1")),
            },
        },
        "plan": {"path": _relative(PLAN_PATH), "sha256": _sha256_file(PLAN_PATH)},
    }
    _write_json(staging_dir / CONTEXT_NAME, context)
    print(json.dumps({"status": "PREPARED", "diagnostics": diagnostics}, ensure_ascii=False))


def _output_hashes(directory: Path) -> dict[str, Any]:
    """计算除 manifest 自身以外的正式输出哈希。"""
    names = (SUMMARY_NAME, SPEC_NAME, DOCX_NAME, TEX_NAME, VALIDATION_NAME)
    return {
        name: {"sha256": _sha256_file(directory / name), "bytes": (directory / name).stat().st_size}
        for name in names
    }


def _build_manifest(
    directory: Path, context: dict[str, Any], validation: dict[str, Any]
) -> dict[str, Any]:
    """构建 Table S5 manifest。"""
    return {
        "manifest_schema_version": "table_s5_manifest_v1",
        "release_status": validation["release_status"],
        "generated_utc": context["generated_utc"],
        "source_snapshot": context["source_snapshot"],
        "protected_snapshot": context["protected_snapshot_before"],
        "raw_snapshot": context["raw_snapshot_before"],
        "sample_counts": context["diagnostics"],
        "reporting_contract": {
            "follow_up_sample": "primary_complete_case_eligible",
            "mi_support_pool": "analysis_panel_base_eligible",
            "descriptive_statistics": "unweighted",
            "later_fi_excludes_landmark": True,
            "threshold_observations_hidden_from_docx": True,
            "figure_3_candidate_range_years": [0, 10],
            "figure_3_final_range_locked": False,
            "mi_performed": False,
            "models_or_meta_run": False,
        },
        "builders": context["builders"],
        "plan": context["plan"],
        "git": context["git"],
        "software": context["software"],
        "outputs": _output_hashes(directory),
        "render_qa": validation["render_qa"],
        "supplement_integration": validation["supplement_integration"],
        "contains_participant_identifiers": False,
    }


def finalize(staging_dir: Path) -> None:
    """结构验证后写入等待 Supplement 与 render QA 的正式记录。"""
    context = _read_json(staging_dir / CONTEXT_NAME)
    if _protected_snapshot() != context["protected_snapshot_before"]:
        raise RuntimeError("生成期间正文或非目标 Supplementary Materials 文件发生变化")
    if _raw_snapshot() != context["raw_snapshot_before"]:
        raise RuntimeError("生成期间原始数据哈希或文件状态发生变化")
    summary = pd.read_csv(staging_dir / SUMMARY_NAME, encoding="utf-8-sig")
    spec = _read_json(staging_dir / SPEC_NAME)
    summary_checks = _validate_summary(summary)
    expected_tex = _build_tex(spec)
    if (staging_dir / TEX_NAME).read_text(encoding="utf-8") != expected_tex:
        raise RuntimeError("staging LaTeX source 与 display spec 不一致")
    panel_b_index = next(
        index for index, row in enumerate(spec["table"]["rows"]) if row["row_type"] == "section"
    )
    docx_structure = _validate_docx(
        staging_dir / DOCX_NAME,
        expected_panel_a_rows=2 + panel_b_index,
        expected_panel_b_rows=1 + len(spec["table"]["rows"]) - panel_b_index - 1,
    )
    timing_gate_counts = {
        cohort: context["diagnostics"][cohort]["timing_gate_counts"] for cohort in COHORTS
    }
    validation = {
        "validation_schema_version": "table_s5_validation_v1",
        "release_status": "PENDING_SUPPLEMENT_AND_RENDER_QA",
        "validated_utc": datetime.now(UTC).isoformat(),
        "data_checks": {
            "ledger_freeze": "PASS",
            "final_eligibility_freeze": "PASS",
            "analysis_panel_freeze": "PASS",
            "table_1_frozen_assets": "PASS",
            "sample_flow_reconciliation": summary_checks["sample_flow_reconciliation"],
            "missingness_union_reconciliation": summary_checks[
                "missingness_union_reconciliation"
            ],
            "assessment_distribution_reconciliation": summary_checks[
                "assessment_distribution_reconciliation"
            ],
            "threshold_monotonicity": summary_checks["threshold_monotonicity"],
            "table_1_anchor_consistency": summary_checks["table_1_anchor_consistency"],
            "table_1_anchor_check_count": len(summary_checks["table_1_anchor_checks"]),
            "one_landmark_per_participant": "PASS",
            "at_least_one_later_fi_per_participant": "PASS",
            "later_time_positive_and_finite": "PASS",
            "person_wave_unique": "PASS",
            "timing_gate_counts": timing_gate_counts,
            "participant_identifiers_in_outputs": False,
        },
        "docx_structure": {"status": "PASS", **docx_structure},
        "latex_source": {"status": "PASS", "generated_from_display_spec": True},
        "protected_assets": {
            "status": "PASS",
            "manuscript_and_non_target_supplement_files_unchanged": True,
            "raw_files_unchanged": True,
        },
        "supplement_integration": {"status": "PENDING"},
        "render_qa": {
            "status": "PENDING",
            "page_count": None,
            "method": None,
            "inspected_utc": None,
            "checks": [],
        },
    }
    _write_json(staging_dir / VALIDATION_NAME, validation)
    _write_json(staging_dir / MANIFEST_NAME, _build_manifest(staging_dir, context, validation))
    print(json.dumps({"status": "FINALIZED_PENDING_SUPPLEMENT_AND_RENDER_QA"}, ensure_ascii=False))


def _compare_summary(expected: pd.DataFrame, observed: pd.DataFrame) -> None:
    """比较正式 CSV 与冻结输入内存重建结果。"""
    expected = expected.copy()
    observed = observed.copy()
    if list(expected.columns) != list(observed.columns):
        raise RuntimeError("summary_long CSV 列顺序漂移")
    text_columns = [
        "row_id",
        "section",
        "block",
        "characteristic",
        "cohort",
        "recovery_status",
        "display_group_label",
        "sample_universe",
        "statistic_type",
        "source_asset",
        "source_column",
        "source_time",
        "display_value",
    ]
    for column in text_columns:
        if expected[column].fillna("").astype(str).tolist() != observed[column].fillna("").astype(str).tolist():
            raise RuntimeError(f"summary_long CSV 文本列漂移：{column}")
    bool_columns = ["display_in_docx"]
    for column in bool_columns:
        if expected[column].astype(bool).tolist() != observed[column].astype(bool).tolist():
            raise RuntimeError(f"summary_long CSV 布尔列漂移：{column}")
    numeric_columns = [
        column for column in expected.columns if column not in text_columns + bool_columns
    ]
    for column in numeric_columns:
        left = pd.to_numeric(expected[column], errors="coerce").to_numpy(float)
        right = pd.to_numeric(observed[column], errors="coerce").to_numpy(float)
        if not np.allclose(left, right, rtol=1e-12, atol=1e-12, equal_nan=True):
            raise RuntimeError(f"summary_long CSV 数值列漂移：{column}")


def check_only(
    output_dir: Path,
    allow_pending_render: bool = False,
    verify_frozen: bool = True,
) -> dict[str, Any]:
    """只读重建并验证全部正式输出和 Supplement 集成。"""
    missing = [name for name in KNOWN_OUTPUTS if not (output_dir / name).is_file()]
    if missing:
        raise RuntimeError("缺少正式输出：" + ", ".join(missing))
    expected_summary, expected_spec, expected_diagnostics, source_snapshot = build_table_data()
    observed_summary = pd.read_csv(output_dir / SUMMARY_NAME, encoding="utf-8-sig")
    _compare_summary(expected_summary, observed_summary)
    observed_spec = _read_json(output_dir / SPEC_NAME)
    if observed_spec != expected_spec:
        raise RuntimeError("display spec 与冻结输入重建结果不一致")
    if (output_dir / TEX_NAME).read_text(encoding="utf-8") != _build_tex(expected_spec):
        raise RuntimeError("正式 LaTeX source 与 display spec 不一致")
    summary_checks = _validate_summary(observed_summary)
    panel_b_index = next(
        index
        for index, row in enumerate(observed_spec["table"]["rows"])
        if row["row_type"] == "section"
    )
    docx_structure = _validate_docx(
        output_dir / DOCX_NAME,
        expected_panel_a_rows=2 + panel_b_index,
        expected_panel_b_rows=1 + len(observed_spec["table"]["rows"]) - panel_b_index - 1,
    )
    supplement_integration = _validate_supplement_integration(output_dir)

    validation = _read_json(output_dir / VALIDATION_NAME)
    manifest = _read_json(output_dir / MANIFEST_NAME)
    if manifest["sample_counts"] != expected_diagnostics:
        raise RuntimeError("manifest sample counts 漂移")
    if manifest["source_snapshot"] != source_snapshot:
        raise RuntimeError("manifest frozen source snapshot 漂移")
    for name, record in manifest["outputs"].items():
        if _sha256_file(output_dir / name) != record["sha256"]:
            raise RuntimeError(f"输出哈希漂移：{name}")
    if _protected_snapshot() != manifest["protected_snapshot"]:
        raise RuntimeError("正文或非目标 Supplementary Materials 文件发生变化")
    if _raw_snapshot() != manifest["raw_snapshot"]:
        raise RuntimeError("原始数据哈希或文件状态发生变化")
    if validation["supplement_integration"].get("status") != "PASS":
        raise RuntimeError("Supplementary Materials 集成尚未登记为 PASS")
    if validation["supplement_integration"]["source_sha256"] != supplement_integration[
        "source_sha256"
    ]:
        raise RuntimeError("Supplementary Materials 中的 Table S5 source 漂移")
    if validation["data_checks"]["table_1_anchor_consistency"] != "PASS" or summary_checks[
        "table_1_anchor_consistency"
    ] != "PASS":
        raise RuntimeError("Table S5 与 Table 1 anchor consistency 未通过")
    if validation["release_status"] != "PASS" and not allow_pending_render:
        raise RuntimeError("DOCX render QA 尚未完成，正式输出不能标记为 PASS")

    result = {
        "status": "PASS" if validation["release_status"] == "PASS" else "PASS_PENDING_RENDER",
        "release_status": validation["release_status"],
        "sample_counts": expected_diagnostics,
        "docx_structure": docx_structure,
        "supplement_integration": "PASS",
        "table_1_anchor_consistency": "PASS",
        "protected_assets_unchanged": True,
        "raw_files_unchanged": True,
    }
    if verify_frozen and (output_dir / TABLE_FREEZE_NAME).is_file():
        result["table_s5_freeze"] = verify_table_s5_freeze(output_dir, emit=False)
    print(json.dumps(result, ensure_ascii=False))
    return result


def record_supplement_pass(output_dir: Path) -> None:
    """在 LaTeX source 接入并成功编译后登记 Supplement 集成 PASS。"""
    if (output_dir / TABLE_FREEZE_NAME).exists():
        raise RuntimeError("Table S5 已冻结，不能改写 Supplement 集成记录")
    validation = _read_json(output_dir / VALIDATION_NAME)
    manifest = _read_json(output_dir / MANIFEST_NAME)
    integration = _validate_supplement_integration(output_dir)
    validation["release_status"] = "PENDING_RENDER_QA"
    validation["validated_utc"] = datetime.now(UTC).isoformat()
    validation["supplement_integration"] = integration
    _write_json(output_dir / VALIDATION_NAME, validation)
    manifest["release_status"] = validation["release_status"]
    manifest["supplement_integration"] = integration
    manifest["outputs"] = _output_hashes(output_dir)
    _write_json(output_dir / MANIFEST_NAME, manifest)
    print(json.dumps({"status": "SUPPLEMENT_INTEGRATION_RECORDED"}, ensure_ascii=False))


def record_render_pass(output_dir: Path, page_count: int, method: str) -> None:
    """在逐页检查 Word render 后登记 QA PASS。"""
    if (output_dir / TABLE_FREEZE_NAME).exists():
        raise RuntimeError("Table S5 已冻结，不能改写 render QA")
    if page_count not in (1, 2):
        raise RuntimeError("Table S5 DOCX 渲染页数必须为 1 或 2")
    validation = _read_json(output_dir / VALIDATION_NAME)
    manifest = _read_json(output_dir / MANIFEST_NAME)
    if validation["docx_structure"]["status"] != "PASS":
        raise RuntimeError("DOCX 结构验证未通过")
    if validation["supplement_integration"].get("status") != "PASS":
        raise RuntimeError("Supplement 集成未通过，不能登记 release PASS")
    checks = [
        "no_clipping_or_overlap",
        "no_missing_glyphs",
        "no_abnormal_wrapping",
        "grouped_header_clear",
        "panel_b_denominator_hierarchy_clear",
        "numeric_alignment_clear",
        "font_size_at_least_8_pt",
    ]
    if page_count == 2:
        checks.append("continuation_headers_repeated")
    validation["release_status"] = "PASS"
    validation["validated_utc"] = datetime.now(UTC).isoformat()
    validation["render_qa"] = {
        "status": "PASS",
        "page_count": page_count,
        "method": method,
        "inspected_utc": datetime.now(UTC).isoformat(),
        "checks": checks,
    }
    _write_json(output_dir / VALIDATION_NAME, validation)
    manifest["release_status"] = "PASS"
    manifest["render_qa"] = validation["render_qa"]
    manifest["outputs"] = _output_hashes(output_dir)
    _write_json(output_dir / MANIFEST_NAME, manifest)
    print(json.dumps({"status": "RENDER_QA_RECORDED", "page_count": page_count}, ensure_ascii=False))


def _release_asset_hashes(output_dir: Path) -> dict[str, dict[str, Any]]:
    """计算 freeze 覆盖的六个正式资产哈希。"""
    return {
        name: {"sha256": _sha256_file(output_dir / name), "bytes": (output_dir / name).stat().st_size}
        for name in KNOWN_OUTPUTS
    }


def verify_table_s5_freeze(output_dir: Path, emit: bool = True) -> dict[str, Any]:
    """验证 Table S5 freeze、正式资产及不可变依赖。"""
    freeze_path = output_dir / TABLE_FREEZE_NAME
    if not freeze_path.is_file():
        raise RuntimeError("Table S5 freeze 记录不存在")
    freeze = _read_json(freeze_path)
    if freeze.get("freeze_status") != "FROZEN":
        raise RuntimeError("Table S5 freeze_status 不是 FROZEN")
    if freeze.get("freeze_record_schema_version") != "table_s5_freeze_record_v1":
        raise RuntimeError("Table S5 freeze schema 不一致")
    for name, record in freeze["release_assets"].items():
        path = output_dir / name
        if not path.is_file() or _sha256_file(path) != record["sha256"]:
            raise RuntimeError(f"Table S5 冻结资产漂移：{name}")
    for record in freeze["dependencies"].values():
        path = ROOT / record["path"]
        if _sha256_file(path) != record["sha256"]:
            raise RuntimeError(f"Table S5 冻结依赖漂移：{record['path']}")
    if _protected_snapshot() != freeze["protected_snapshot"]:
        raise RuntimeError("正文或非目标 Supplementary Materials 文件相对 S5 freeze 发生变化")
    if _raw_snapshot() != freeze["raw_snapshot"]:
        raise RuntimeError("原始数据相对 Table S5 freeze 发生变化")
    _verify_table_1_assets()
    _validate_supplement_integration(output_dir)
    result = {
        "status": "PASS",
        "freeze_status": "FROZEN",
        "release_version": freeze["release_version"],
        "release_asset_count": len(freeze["release_assets"]),
        "figure_3_final_range_locked": False,
    }
    if emit:
        print(json.dumps(result, ensure_ascii=False))
    return result


def freeze_release(output_dir: Path) -> None:
    """在严格检查、Supplement 集成和 render QA 后冻结 Table S5 v1。"""
    freeze_path = output_dir / TABLE_FREEZE_NAME
    if freeze_path.exists():
        verify_table_s5_freeze(output_dir)
        return
    check_only(output_dir, verify_frozen=False)
    validation = _read_json(output_dir / VALIDATION_NAME)
    if validation["release_status"] != "PASS":
        raise RuntimeError("Table S5 尚未通过 render QA，不能冻结")
    manifest = _read_json(output_dir / MANIFEST_NAME)
    dependency_paths = {
        "analysis_cohort_ledger_freeze": LEDGER_FREEZE_PATH,
        "final_eligibility_freeze": FINAL_FREEZE_PATH,
        "analysis_panel_freeze": ANALYSIS_FREEZE_PATH,
        "table_1_freeze": TABLE_1_FREEZE_PATH,
        "table_1_summary": TABLE_1_SUMMARY_PATH,
        "table_s5_builder": Path(__file__),
        "table_s5_docx_builder": Path(__file__).with_name("build_table_s5_docx.mjs"),
        "table_s5_runner": Path(__file__).with_name("run_table_s5.ps1"),
        "generation_plan": PLAN_PATH,
    }
    freeze = {
        "freeze_record_schema_version": "table_s5_freeze_record_v1",
        "freeze_status": "FROZEN",
        "release_version": "table_s5_v6",
        "frozen_utc": datetime.now(UTC).isoformat(),
        "release_assets": _release_asset_hashes(output_dir),
        "dependencies": {
            name: {"path": _relative(path), "sha256": _sha256_file(path)}
            for name, path in dependency_paths.items()
        },
        "reporting_contract": manifest["reporting_contract"],
        "support_audit": manifest["sample_counts"],
        "table_1_anchor_consistency": "PASS",
        "render_qa": validation["render_qa"],
        "supplement_integration_at_freeze": validation["supplement_integration"],
        "protected_snapshot": _protected_snapshot(),
        "raw_snapshot": _raw_snapshot(),
        "git": _git_state(),
        "overwrite_policy": (
            "fail closed; frozen Table S5 assets must not be overwritten; "
            "a substantive or schema change requires a new release version"
        ),
        "figure_3_decision_boundary": {
            "candidate_display_range_years": [0, 10],
            "final_range_locked": False,
            "author_confirmation_required_before_figure_3_release": True,
        },
    }
    temporary = freeze_path.with_suffix(".json.tmp")
    _write_json(temporary, freeze)
    temporary.replace(freeze_path)
    verify_table_s5_freeze(output_dir)


def parse_args() -> argparse.Namespace:
    """解析生成、集成、验证和冻结子命令。"""
    parser = argparse.ArgumentParser(description="生成并验证正式 Supplementary Table S5")
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--staging-dir", type=Path, required=True)
    finalize_parser = subparsers.add_parser("finalize")
    finalize_parser.add_argument("--staging-dir", type=Path, required=True)
    check_parser = subparsers.add_parser("check-only")
    check_parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    check_parser.add_argument("--allow-pending-render", action="store_true")
    supplement_parser = subparsers.add_parser("record-supplement-pass")
    supplement_parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    render_parser = subparsers.add_parser("record-render-pass")
    render_parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    render_parser.add_argument("--page-count", type=int, required=True)
    render_parser.add_argument("--method", required=True)
    freeze_parser = subparsers.add_parser("freeze")
    freeze_parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    verify_parser = subparsers.add_parser("verify-freeze")
    verify_parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    """执行所选子命令。"""
    args = parse_args()
    if args.command == "prepare":
        prepare(args.staging_dir.resolve())
    elif args.command == "finalize":
        finalize(args.staging_dir.resolve())
    elif args.command == "check-only":
        check_only(args.output_dir.resolve(), args.allow_pending_render)
    elif args.command == "record-supplement-pass":
        record_supplement_pass(args.output_dir.resolve())
    elif args.command == "record-render-pass":
        record_render_pass(args.output_dir.resolve(), args.page_count, args.method)
    elif args.command == "freeze":
        freeze_release(args.output_dir.resolve())
    elif args.command == "verify-freeze":
        verify_table_s5_freeze(args.output_dir.resolve())
    else:
        raise RuntimeError(f"未知命令：{args.command}")


if __name__ == "__main__":
    main()
