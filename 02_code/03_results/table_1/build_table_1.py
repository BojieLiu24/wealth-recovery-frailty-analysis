from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import zipfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable
from xml.etree import ElementTree as ET

import numpy as np
import pandas as pd
import pyarrow


ROOT = Path(__file__).resolve().parents[3]
PANEL_CODE_DIR = ROOT / "02_code" / "01_panel_build"
if str(PANEL_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(PANEL_CODE_DIR))

import analysis_panel


OUTPUT_DIR = (
    ROOT
    / "03_output"
    / "3.1 Study Population, Wealth Recovery, and Follow-up"
    / "Table 1"
)
PLAN_PATH = ROOT / "07_manuscript" / "Table" / "table_1_generation_plan_20260915.md"
MANUSCRIPT_PATH = ROOT / "07_manuscript" / "manuscript_v1.docx"
SUPPLEMENT_DIR = ROOT / "07_manuscript" / "supplementary_materials"
FREEZE_PATH = (
    ROOT
    / "00_admin"
    / "panel_manifests"
    / "analysis_panel"
    / "analysis_panel_v1_freeze.json"
)

SUMMARY_NAME = "table_1_summary_long.csv"
SPEC_NAME = "table_1_display_spec.json"
DOCX_NAME = "table_1_editable.docx"
MANIFEST_NAME = "table_1_manifest.json"
VALIDATION_NAME = "table_1_validation.json"
TABLE_FREEZE_NAME = "table_1_freeze.json"
CONTEXT_NAME = "table_1_build_context.json"
KNOWN_OUTPUTS = (SUMMARY_NAME, SPEC_NAME, DOCX_NAME, MANIFEST_NAME, VALIDATION_NAME)
TABLE_FREEZE_PATH = OUTPUT_DIR / TABLE_FREEZE_NAME
COVARIATE_CODE_PATH = PANEL_CODE_DIR / "covariate_panel.py"
COVARIATE_FREEZE_PATH = (
    ROOT / "00_admin" / "panel_manifests" / "covariate_panel" / "covariate_panel_v1_freeze.json"
)
KLOSA_COVARIATE_MANIFEST_PATH = (
    ROOT / "00_admin" / "panel_manifests" / "covariate_panel" / "klosa_covariate_panel_v1.json"
)
EXPOSURE_FREEZE_PATH = (
    ROOT / "00_admin" / "panel_manifests" / "exposure_panel" / "exposure_panel_v1_freeze.json"
)

COHORTS = ("HRS", "ELSA", "KLoSA")
GROUPS = ("substantial_recovery", "persistent_or_limited_recovery")
GROUP_LABELS = {
    "substantial_recovery": "Substantial recovery",
    "persistent_or_limited_recovery": "Persistent/limited recovery",
}

EXPECTED = {
    "HRS": {
        "participants": 5114,
        "groups": {
            "substantial_recovery": 1703,
            "persistent_or_limited_recovery": 3411,
        },
        "later_rows": 18297,
    },
    "ELSA": {
        "participants": 699,
        "groups": {
            "substantial_recovery": 271,
            "persistent_or_limited_recovery": 428,
        },
        "later_rows": 1885,
    },
    "KLoSA": {
        "participants": 2502,
        "groups": {
            "substantial_recovery": 943,
            "persistent_or_limited_recovery": 1559,
        },
        "later_rows": 8260,
    },
}

REQUIRED_LANDMARK_COLUMNS = (
    "age",
    "sex",
    "education",
    "marital_status",
    "household_size",
    "labour_force_status",
    "household_income_rank",
    "current_smoking",
    "wealth_quintile",
    "wealth_loss_fraction",
    "fi_t",
    "fi_primary",
    "recovery_status",
)

ALLOWED_LEVELS = {
    "sex": ("Male", "Female"),
    "education": ("Low", "Medium", "High"),
    "marital_status": (
        "Married/partnered",
        "Separated/divorced",
        "Widowed",
        "Never married",
    ),
    "labour_force_status": (
        "Working",
        "Unemployed",
        "Retired or partly retired",
        "Other economically inactive",
    ),
    "wealth_quintile": (1, 2, 3, 4, 5),
}

CAPTION = (
    "Table 1. Participant characteristics by cohort and post-shock "
    "wealth-recovery status."
)
NOTES = (
    "Values are mean (SD), median (IQR), or n (%), unless otherwise indicated. "
    "Participants are shown as n (% of the cohort-specific primary complete-case "
    "sample). Percentages for other categorical characteristics use the "
    "corresponding cohort × recovery-status group as the denominator. Substantial "
    "recovery was defined as regaining at least 50% of the wealth lost by the "
    "landmark interview (t+1); persistent/limited recovery was defined as regaining "
    "less than 50%. Descriptive statistics were unweighted. Participant counts "
    "refer to unique individuals. Measurement timing follows the definitions in "
    "Section 2.5 and Supplementary Table S4. Follow-up duration was measured from "
    "the landmark interview to the participant’s last valid FI assessment. "
    "Post-landmark FI observations and assessments exclude the landmark assessment. "
    "Wealth loss may exceed 100% when wealth at the shock interview is negative; "
    "zero and negative shock-wave wealth values were retained. FI, frailty index; "
    "HRS, Health and Retirement Study; ELSA, English Longitudinal Study of Ageing; "
    "KLoSA, Korean Longitudinal Study of Aging."
)


def _relative(path: Path) -> str:
    """返回仓库相对 POSIX 路径。"""
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def _sha256_file(path: Path) -> str:
    """计算普通文件 SHA-256。"""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _protected_hash(path: Path) -> dict[str, Any]:
    """优先使用 SHA-256；独占锁存在时退回 Git blob 哈希。"""
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
    """记录生成时 Git 状态，不改变索引或工作树。"""
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    ).stdout.decode("ascii").strip()
    status = subprocess.run(
        ["git", "status", "--porcelain=v1"],
        cwd=ROOT,
        check=True,
        capture_output=True,
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


def _supplement_snapshot() -> dict[str, dict[str, Any]]:
    """记录 Supplementary Materials 全部现有文件。"""
    output: dict[str, dict[str, Any]] = {}
    for path in sorted(item for item in SUPPLEMENT_DIR.rglob("*") if item.is_file()):
        output[_relative(path)] = _protected_hash(path)
    return output


def _protected_snapshot() -> dict[str, Any]:
    """记录不得被本流程修改的正文与补充材料。"""
    return {
        "manuscript": {
            "path": _relative(MANUSCRIPT_PATH),
            **_protected_hash(MANUSCRIPT_PATH),
        },
        "supplementary_materials": _supplement_snapshot(),
    }


def _source_snapshot(freeze: dict[str, Any]) -> dict[str, Any]:
    """压缩记录冻结 Analysis Panel 的正式输入引用。"""
    cohorts: dict[str, Any] = {}
    for cohort in COHORTS:
        record = freeze["cohorts"][cohort]
        cohorts[cohort] = {
            "parquet_path": record["parquet_path"],
            "parquet_sha256": record["parquet_sha256"],
            "manifest_path": record["manifest_path"],
            "manifest_sha256": record["manifest_sha256"],
        }
    return {
        "analysis_panel_freeze": {
            "path": _relative(FREEZE_PATH),
            "sha256": _sha256_file(FREEZE_PATH),
            "release_version": freeze["release_version"],
            "schema_version": freeze["schema_version"],
        },
        "release_candidate": freeze["source_release_candidate"],
        "design_register": freeze["design_register"],
        "cohorts": cohorts,
    }


def _format_count(value: int) -> str:
    """以英文千位分隔符格式化整数。"""
    return f"{int(value):,}"


def _format_mean_sd(series: pd.Series, precision: int) -> tuple[str, dict[str, float]]:
    """生成 mean (SD) 显示值与全精度统计量。"""
    values = series.astype(float)
    mean = float(values.mean())
    sd = float(values.std(ddof=1))
    return f"{mean:.{precision}f} ({sd:.{precision}f})", {"mean": mean, "sd": sd}


def _format_median_iqr(
    series: pd.Series,
    precision: int,
    transform: Callable[[pd.Series], pd.Series] | None = None,
) -> tuple[str, dict[str, float]]:
    """生成 median (IQR) 显示值与全精度统计量。"""
    values = series.astype(float)
    if transform is not None:
        values = transform(values)
    median = float(values.median())
    q1 = float(values.quantile(0.25))
    q3 = float(values.quantile(0.75))
    return (
        f"{median:.{precision}f} ({q1:.{precision}f}–{q3:.{precision}f})",
        {"median": median, "q1": q1, "q3": q3},
    )


def _format_count_percent(numerator: int, denominator: int) -> tuple[str, float]:
    """生成 n (%) 显示值。"""
    percentage = 100.0 * numerator / denominator
    return f"{_format_count(numerator)} ({percentage:.1f})", percentage


def _assert_finite(series: pd.Series, label: str) -> None:
    """拒绝任何非有限数值。"""
    values = series.astype(float).to_numpy()
    if not np.isfinite(values).all():
        raise RuntimeError(f"{label}: 出现非有限数值")


def _load_and_validate_primary(
    freeze: dict[str, Any], cohort: str
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """读取一个队列并验证 Table 1 所需的主样本结构。"""
    record = freeze["cohorts"][cohort]
    frame = pd.read_parquet(ROOT / record["parquet_path"])
    primary = frame.loc[frame["primary_complete_case_eligible"]].copy()
    if primary.duplicated(["cohort", "person_id", "wave"]).any():
        raise RuntimeError(f"{cohort}: primary person-wave 主键重复")
    if not primary["cohort"].eq(cohort).all():
        raise RuntimeError(f"{cohort}: cohort 字段不一致")
    if primary["person_id"].isna().any():
        raise RuntimeError(f"{cohort}: person_id 缺失")

    landmarks = primary.loc[primary["fi_time_role"].eq("landmark")].copy()
    later = primary.loc[primary["fi_time_role"].eq("later")].copy()
    landmark_counts = landmarks.groupby("person_id", observed=True).size()
    if not landmark_counts.eq(1).all():
        raise RuntimeError(f"{cohort}: 每人必须恰有一个 landmark 行")
    later_counts = later.groupby("person_id", observed=True).size()
    if set(landmarks["person_id"]) != set(later_counts.index):
        raise RuntimeError(f"{cohort}: 每名主样本参与者必须至少有一个 later FI")
    if not later["time_years"].gt(0).all():
        raise RuntimeError(f"{cohort}: later FI time_years 必须大于零")
    if landmarks.loc[:, REQUIRED_LANDMARK_COLUMNS].isna().any(axis=None):
        missing = landmarks.loc[:, REQUIRED_LANDMARK_COLUMNS].isna().sum()
        raise RuntimeError(f"{cohort}: Table 1 字段缺失：{missing[missing.gt(0)].to_dict()}")

    for column in (
        "age",
        "household_size",
        "household_income_rank",
        "wealth_loss_fraction",
        "fi_t",
        "fi_primary",
    ):
        _assert_finite(landmarks[column], f"{cohort}/{column}")
    _assert_finite(later["time_years"], f"{cohort}/later time_years")

    for column, allowed in ALLOWED_LEVELS.items():
        observed = set(landmarks[column].tolist())
        if not observed.issubset(set(allowed)):
            raise RuntimeError(f"{cohort}/{column}: 出现未声明 level：{observed}")
    if set(landmarks["recovery_status"].tolist()) != set(GROUPS):
        raise RuntimeError(f"{cohort}: recovery_status levels 不完整")

    expected = EXPECTED[cohort]
    if landmarks["person_id"].nunique() != expected["participants"]:
        raise RuntimeError(f"{cohort}: participant count 与合同不一致")
    if len(later) != expected["later_rows"]:
        raise RuntimeError(f"{cohort}: later row count 与合同不一致")
    for group in GROUPS:
        observed = int(landmarks["recovery_status"].eq(group).sum())
        if observed != expected["groups"][group]:
            raise RuntimeError(f"{cohort}/{group}: group count 与合同不一致")

    return primary, landmarks, later


def build_table_data(
    freeze: dict[str, Any]
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    """在内存中构建全精度长表和显示规格。"""
    state: dict[str, dict[str, Any]] = {}
    diagnostics: dict[str, Any] = {}
    for cohort in COHORTS:
        primary, landmarks, later = _load_and_validate_primary(freeze, cohort)
        follow = (
            later.groupby(["person_id", "recovery_status"], observed=True)
            .agg(
                post_landmark_assessments=("wave", "size"),
                follow_up_duration_years=("time_years", "max"),
            )
            .reset_index()
        )
        state[cohort] = {
            "primary": primary,
            "landmarks": landmarks,
            "later": later,
            "follow": follow,
        }
        diagnostics[cohort] = {
            "participants": int(len(landmarks)),
            "post_landmark_fi_observations": int(len(later)),
            "groups": {
                group: int(landmarks["recovery_status"].eq(group).sum())
                for group in GROUPS
            },
        }

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
    row_order = 0

    def add_section(label: str) -> None:
        nonlocal row_order
        row_order += 1
        display_rows.append(
            {
                "row_order": row_order,
                "row_id": f"section_{row_order}",
                "row_type": "section",
                "label": label,
                "indent": 0,
                "values": {column["key"]: "" for column in columns},
            }
        )

    def add_category_header(label: str) -> None:
        nonlocal row_order
        row_order += 1
        display_rows.append(
            {
                "row_order": row_order,
                "row_id": f"header_{row_order}",
                "row_type": "category_header",
                "label": label,
                "indent": 0,
                "values": {column["key"]: "" for column in columns},
            }
        )

    def add_data_row(
        *,
        row_id: str,
        section: str,
        label: str,
        characteristic: str,
        level: str | None,
        statistic_type: str,
        source_column: str,
        source_time: str,
        precision: int = 1,
        target: Any | None = None,
        indent: int = 0,
        transform: Callable[[pd.Series], pd.Series] | None = None,
    ) -> None:
        nonlocal row_order
        row_order += 1
        display_values: dict[str, str] = {}
        for cohort in COHORTS:
            cohort_total = len(state[cohort]["landmarks"])
            for group in GROUPS:
                key = f"{cohort}__{group}"
                landmark_group = state[cohort]["landmarks"].loc[
                    state[cohort]["landmarks"]["recovery_status"].eq(group)
                ]
                later_group = state[cohort]["later"].loc[
                    state[cohort]["later"]["recovery_status"].eq(group)
                ]
                follow_group = state[cohort]["follow"].loc[
                    state[cohort]["follow"]["recovery_status"].eq(group)
                ]
                denominator = int(len(landmark_group))
                record: dict[str, Any] = {
                    "row_order": row_order,
                    "row_id": row_id,
                    "section": section,
                    "characteristic": characteristic,
                    "level": level,
                    "cohort": cohort,
                    "recovery_status": group,
                    "display_group_label": GROUP_LABELS[group],
                    "statistic_type": statistic_type,
                    "source_column": source_column,
                    "source_time": source_time,
                    "numerator": None,
                    "denominator": denominator,
                    "value": None,
                    "percentage": None,
                    "mean": None,
                    "sd": None,
                    "median": None,
                    "q1": None,
                    "q3": None,
                }

                if statistic_type == "mean_sd":
                    display, stats = _format_mean_sd(landmark_group[source_column], precision)
                    record.update(stats)
                elif statistic_type == "median_iqr":
                    display, stats = _format_median_iqr(
                        landmark_group[source_column], precision, transform
                    )
                    record.update(stats)
                elif statistic_type == "count_percent":
                    numerator = int(landmark_group[source_column].eq(target).sum())
                    display, percentage = _format_count_percent(numerator, denominator)
                    record.update({"numerator": numerator, "percentage": percentage})
                elif statistic_type == "participants_count_percent":
                    numerator = denominator
                    display, percentage = _format_count_percent(numerator, cohort_total)
                    record.update(
                        {
                            "numerator": numerator,
                            "denominator": cohort_total,
                            "percentage": percentage,
                        }
                    )
                elif statistic_type == "total_later_count":
                    value = int(len(later_group))
                    display = _format_count(value)
                    record.update({"value": value})
                elif statistic_type == "follow_up_median_iqr":
                    display, stats = _format_median_iqr(
                        follow_group[source_column], precision
                    )
                    record.update(stats)
                else:
                    raise RuntimeError(f"未知 statistic_type：{statistic_type}")

                record["display_value"] = display
                display_values[key] = display
                summary_records.append(record)

        display_rows.append(
            {
                "row_order": row_order,
                "row_id": row_id,
                "row_type": "data",
                "label": label,
                "indent": indent,
                "values": display_values,
            }
        )

    add_section("Participant characteristics")
    add_data_row(
        row_id="age_at_shock",
        section="Participant characteristics",
        label="Age at shock interview, years",
        characteristic="Age at shock interview, years",
        level=None,
        statistic_type="mean_sd",
        source_column="age",
        source_time="t",
        precision=1,
    )
    add_data_row(
        row_id="female",
        section="Participant characteristics",
        label="Female, n (%)",
        characteristic="Sex",
        level="Female",
        statistic_type="count_percent",
        source_column="sex",
        source_time="fixed",
        target="Female",
    )
    add_category_header("Education, n (%)")
    for level in ALLOWED_LEVELS["education"]:
        add_data_row(
            row_id=f"education_{str(level).lower()}",
            section="Participant characteristics",
            label=str(level),
            characteristic="Education",
            level=str(level),
            statistic_type="count_percent",
            source_column="education",
            source_time="fixed",
            target=level,
            indent=1,
        )
    add_category_header("Marital status at shock interview, n (%)")
    marital_ids = {
        "Married/partnered": "married_partnered",
        "Separated/divorced": "separated_divorced",
        "Widowed": "widowed",
        "Never married": "never_married",
    }
    for level in ALLOWED_LEVELS["marital_status"]:
        add_data_row(
            row_id=f"marital_{marital_ids[level]}",
            section="Participant characteristics",
            label=str(level),
            characteristic="Marital status",
            level=str(level),
            statistic_type="count_percent",
            source_column="marital_status",
            source_time="t",
            target=level,
            indent=1,
        )
    add_data_row(
        row_id="household_size",
        section="Participant characteristics",
        label="Household size at shock interview",
        characteristic="Household size at shock interview",
        level=None,
        statistic_type="median_iqr",
        source_column="household_size",
        source_time="t",
        precision=1,
    )
    add_category_header("Labour-force status at shock interview, n (%)")
    labour_ids = {
        "Working": "working",
        "Unemployed": "unemployed",
        "Retired or partly retired": "retired_partly_retired",
        "Other economically inactive": "other_inactive",
    }
    for level in ALLOWED_LEVELS["labour_force_status"]:
        add_data_row(
            row_id=f"labour_{labour_ids[level]}",
            section="Participant characteristics",
            label=str(level),
            characteristic="Labour-force status",
            level=str(level),
            statistic_type="count_percent",
            source_column="labour_force_status",
            source_time="t",
            target=level,
            indent=1,
        )
    add_data_row(
        row_id="household_income_rank",
        section="Participant characteristics",
        label="Household income rank at shock interview (0–1)",
        characteristic="Household income rank at shock interview",
        level=None,
        statistic_type="mean_sd",
        source_column="household_income_rank",
        source_time="t",
        precision=2,
    )
    add_data_row(
        row_id="current_smoking",
        section="Participant characteristics",
        label="Current smoking at shock interview, n (%)",
        characteristic="Current smoking",
        level="Yes",
        statistic_type="count_percent",
        source_column="current_smoking",
        source_time="t",
        target=True,
    )

    add_section("Wealth and frailty")
    add_category_header("Pre-shock wealth quintile, n (%)")
    wealth_labels = {
        1: "Q1 (lowest)",
        2: "Q2",
        3: "Q3",
        4: "Q4",
        5: "Q5 (highest)",
    }
    for level in ALLOWED_LEVELS["wealth_quintile"]:
        add_data_row(
            row_id=f"wealth_quintile_q{level}",
            section="Wealth and frailty",
            label=wealth_labels[level],
            characteristic="Pre-shock wealth quintile",
            level=wealth_labels[level],
            statistic_type="count_percent",
            source_column="wealth_quintile",
            source_time="t-1",
            target=level,
            indent=1,
        )
    add_data_row(
        row_id="wealth_lost_percent",
        section="Wealth and frailty",
        label="Wealth lost during the index shock, %",
        characteristic="Wealth lost during the index shock, %",
        level=None,
        statistic_type="median_iqr",
        source_column="wealth_loss_fraction",
        source_time="t-1 to t",
        precision=1,
        transform=lambda values: values * 100.0,
    )
    add_data_row(
        row_id="fi_at_shock",
        section="Wealth and frailty",
        label="FI at shock interview",
        characteristic="FI at shock interview",
        level=None,
        statistic_type="mean_sd",
        source_column="fi_t",
        source_time="t",
        precision=3,
    )
    add_data_row(
        row_id="fi_at_landmark",
        section="Wealth and frailty",
        label="FI at landmark interview",
        characteristic="FI at landmark interview",
        level=None,
        statistic_type="mean_sd",
        source_column="fi_primary",
        source_time="t+1",
        precision=3,
    )

    add_section("Follow-up")
    add_data_row(
        row_id="participants",
        section="Follow-up",
        label="Participants, n (% within cohort)",
        characteristic="Participants",
        level=None,
        statistic_type="participants_count_percent",
        source_column="person_id",
        source_time="participant",
    )
    add_data_row(
        row_id="total_post_landmark_fi_observations",
        section="Follow-up",
        label="Total post-landmark FI observations, n",
        characteristic="Total post-landmark FI observations",
        level=None,
        statistic_type="total_later_count",
        source_column="fi_primary",
        source_time="t+2 onward",
    )
    add_data_row(
        row_id="follow_up_duration",
        section="Follow-up",
        label="Follow-up duration, years",
        characteristic="Follow-up duration, years",
        level=None,
        statistic_type="follow_up_median_iqr",
        source_column="follow_up_duration_years",
        source_time="landmark to last valid FI",
        precision=1,
    )
    add_data_row(
        row_id="post_landmark_assessments_per_participant",
        section="Follow-up",
        label="Post-landmark FI assessments per participant",
        characteristic="Post-landmark FI assessments per participant",
        level=None,
        statistic_type="follow_up_median_iqr",
        source_column="post_landmark_assessments",
        source_time="t+2 onward",
        precision=0,
    )

    summary = pd.DataFrame.from_records(summary_records)
    if "person_id" in summary.columns:
        raise RuntimeError("聚合 CSV 不得包含 person_id 列")

    spec = {
        "schema_version": "table_1_display_spec_v1",
        "caption": CAPTION,
        "notes": NOTES,
        "table": {
            "cohort_order": list(COHORTS),
            "group_order": list(GROUPS),
            "columns": columns,
            "rows": display_rows,
        },
        "layout": {
            "paper": "A4",
            "orientation": "landscape",
            "font": "Times New Roman",
            "body_font_pt": 8.5,
            "note_font_pt": 8.0,
            "minimum_font_pt": 8.0,
            "preferred_pages": 1,
            "maximum_pages": 2,
            "vertical_gridlines": False,
            "background_shading": False,
        },
        "reporting": {
            "weighted": False,
            "participant_percentage_denominator": (
                "cohort-specific primary complete-case sample"
            ),
            "categorical_percentage_denominator": (
                "cohort by recovery-status group"
            ),
            "post_landmark_excludes_landmark": True,
            "wealth_loss_above_100_allowed": True,
        },
    }
    return summary, spec, diagnostics


def _validate_category_reconciliation(summary: pd.DataFrame) -> list[dict[str, Any]]:
    """验证多分类 level 计数与列分母一致。"""
    checks: list[dict[str, Any]] = []
    categories = {
        "Education": 3,
        "Marital status": 4,
        "Labour-force status": 4,
        "Pre-shock wealth quintile": 5,
    }
    for characteristic, level_count in categories.items():
        subset = summary.loc[summary["characteristic"].eq(characteristic)]
        for cohort in COHORTS:
            for group in GROUPS:
                cells = subset.loc[
                    subset["cohort"].eq(cohort)
                    & subset["recovery_status"].eq(group)
                ]
                if len(cells) != level_count:
                    raise RuntimeError(
                        f"{cohort}/{group}/{characteristic}: level 数量不一致"
                    )
                numerator = int(cells["numerator"].sum())
                denominator = int(cells["denominator"].iloc[0])
                if numerator != denominator:
                    raise RuntimeError(
                        f"{cohort}/{group}/{characteristic}: level 计数未归一"
                    )
                checks.append(
                    {
                        "check": "categorical_levels_sum_to_denominator",
                        "cohort": cohort,
                        "recovery_status": group,
                        "characteristic": characteristic,
                        "status": "PASS",
                        "numerator": numerator,
                        "denominator": denominator,
                    }
                )
    return checks


def _validate_docx(path: Path, expected_row_count: int) -> dict[str, Any]:
    """检查 DOCX 原生表格结构与禁用样式。"""
    if not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError("缺少非空 table_1_editable.docx")
    with zipfile.ZipFile(path) as archive:
        names = set(archive.namelist())
        if "word/document.xml" not in names:
            raise RuntimeError("DOCX 缺少 word/document.xml")
        document_xml = archive.read("word/document.xml")
    root = ET.fromstring(document_xml)
    ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    tables = root.findall(".//w:tbl", ns)
    if len(tables) != 1:
        raise RuntimeError(f"DOCX 必须恰有一个原生表格，实际 {len(tables)}")
    table = tables[0]
    grid_columns = table.findall("./w:tblGrid/w:gridCol", ns)
    rows = table.findall("./w:tr", ns)
    if len(grid_columns) != 7:
        raise RuntimeError(f"DOCX 表格必须为 7 列，实际 {len(grid_columns)}")
    if len(rows) != expected_row_count:
        raise RuntimeError(
            f"DOCX 表格行数不一致：expected={expected_row_count}, observed={len(rows)}"
        )
    if len(table.findall(".//w:gridSpan", ns)) < 3:
        raise RuntimeError("DOCX 缺少三组 cohort 横向合并表头")
    if len(table.findall(".//w:vMerge", ns)) < 2:
        raise RuntimeError("DOCX 缺少 Characteristic 纵向合并表头")
    if len(rows[0].findall("./w:trPr/w:tblHeader", ns)) != 1:
        raise RuntimeError("DOCX 第一行未设置重复表头")
    if len(rows[1].findall("./w:trPr/w:tblHeader", ns)) != 1:
        raise RuntimeError("DOCX 第二行未设置重复表头")
    if root.findall(".//w:drawing", ns) or root.findall(".//w:pict", ns):
        raise RuntimeError("DOCX 不得包含图片化表格")

    texts = [node.text or "" for node in root.findall(".//w:t", ns)]
    full_text = "".join(texts)
    for required in (
        CAPTION,
        "Substantial recovery",
        "Persistent/limited recovery",
        "Total post-landmark FI observations, n",
        "Post-landmark FI assessments per participant",
        "Wealth loss may exceed 100%",
    ):
        if required not in full_text:
            raise RuntimeError(f"DOCX 缺少必需文本：{required}")
    for prohibited in ("P value", "SMD", "Pooled", "pooled"):
        if prohibited in full_text:
            raise RuntimeError(f"DOCX 出现禁用文本：{prohibited}")

    word_ns = f"{{{ns['w']}}}"
    vertical_border_violations: list[str] = []
    for tag in ("left", "right", "insideV"):
        for node in root.findall(f".//w:{tag}", ns):
            value = node.attrib.get(f"{word_ns}val", "")
            if value not in ("nil", "none", ""):
                vertical_border_violations.append(f"{tag}:{value}")
    if vertical_border_violations:
        raise RuntimeError(
            "DOCX 出现竖线：" + ", ".join(vertical_border_violations[:10])
        )

    shading_violations: list[str] = []
    for node in root.findall(".//w:shd", ns):
        fill = node.attrib.get(f"{word_ns}fill", "")
        if fill.lower() not in ("", "auto", "ffffff"):
            shading_violations.append(fill)
    if shading_violations:
        raise RuntimeError("DOCX 出现背景色：" + ", ".join(shading_violations[:10]))

    sizes = []
    for node in root.findall(".//w:sz", ns):
        value = node.attrib.get(f"{word_ns}val")
        if value:
            sizes.append(float(value) / 2.0)
    if sizes and min(sizes) < 8.0:
        raise RuntimeError(f"DOCX 最小字号低于 8 pt：{min(sizes)}")
    section = root.find(".//w:sectPr", ns)
    if section is None:
        raise RuntimeError("DOCX 缺少 section properties")
    page_size = section.find("./w:pgSz", ns)
    if page_size is None or page_size.attrib.get(f"{word_ns}orient") != "landscape":
        raise RuntimeError("DOCX 必须为 landscape")
    page_width = int(page_size.attrib.get(f"{word_ns}w", "0"))
    page_height = int(page_size.attrib.get(f"{word_ns}h", "0"))
    if page_width <= page_height:
        raise RuntimeError(
            f"DOCX 横向页面尺寸无效：width={page_width}, height={page_height}"
        )

    return {
        "native_table_count": len(tables),
        "table_columns": len(grid_columns),
        "table_rows": len(rows),
        "grid_span_count": len(table.findall(".//w:gridSpan", ns)),
        "vertical_merge_count": len(table.findall(".//w:vMerge", ns)),
        "image_count": 0,
        "minimum_explicit_font_pt": min(sizes) if sizes else None,
        "orientation": "landscape",
        "page_width_twips": page_width,
        "page_height_twips": page_height,
        "vertical_gridlines": False,
        "background_shading": False,
    }


def prepare(staging_dir: Path) -> None:
    """生成聚合长表、显示规格与只读输入快照。"""
    staging_dir.mkdir(parents=True, exist_ok=False)
    freeze = analysis_panel.verify_analysis_panel_freeze()
    summary, spec, diagnostics = build_table_data(freeze)
    category_checks = _validate_category_reconciliation(summary)
    summary.to_csv(
        staging_dir / SUMMARY_NAME,
        index=False,
        encoding="utf-8-sig",
        lineterminator="\n",
        float_format="%.17g",
    )
    _write_json(staging_dir / SPEC_NAME, spec)
    context = {
        "schema_version": "table_1_build_context_v1",
        "generated_utc": datetime.now(UTC).isoformat(),
        "source_snapshot": _source_snapshot(freeze),
        "protected_snapshot_before": _protected_snapshot(),
        "diagnostics": diagnostics,
        "category_checks": category_checks,
        "git": _git_state(),
        "software": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "pyarrow": pyarrow.__version__,
        },
        "builder": {
            "path": _relative(Path(__file__)),
            "sha256": _sha256_file(Path(__file__)),
        },
        "plan": {
            "path": _relative(PLAN_PATH),
            "sha256": _sha256_file(PLAN_PATH),
        },
    }
    _write_json(staging_dir / CONTEXT_NAME, context)
    print(json.dumps({"status": "PREPARED", "diagnostics": diagnostics}, ensure_ascii=False))


def _output_hashes(directory: Path) -> dict[str, Any]:
    """计算除 manifest 自身以外的正式输出哈希。"""
    names = (SUMMARY_NAME, SPEC_NAME, DOCX_NAME, VALIDATION_NAME)
    output: dict[str, Any] = {}
    for name in names:
        path = directory / name
        output[name] = {"sha256": _sha256_file(path), "bytes": path.stat().st_size}
    return output


def _build_manifest(
    directory: Path,
    context: dict[str, Any],
    validation: dict[str, Any],
) -> dict[str, Any]:
    """构建 Table 1 输出 manifest。"""
    return {
        "manifest_schema_version": "table_1_manifest_v1",
        "release_status": validation["release_status"],
        "generated_utc": context["generated_utc"],
        "source_snapshot": context["source_snapshot"],
        "protected_snapshot": context["protected_snapshot_before"],
        "sample_counts": context["diagnostics"],
        "reporting_contract": {
            "sample": "primary_complete_case_eligible",
            "descriptive_statistics": "unweighted",
            "participant_unit": "unique participant",
            "later_fi_excludes_landmark": True,
            "wealth_loss_above_100_allowed": True,
            "p_values": False,
            "standardised_differences": False,
            "pooled_column": False,
        },
        "builder": context["builder"],
        "plan": context["plan"],
        "git": context["git"],
        "software": context["software"],
        "outputs": _output_hashes(directory),
        "render_qa": validation["render_qa"],
        "contains_participant_identifiers": False,
    }


def finalize(staging_dir: Path) -> None:
    """完成结构验证并写入待视觉复核的 validation 与 manifest。"""
    context = _read_json(staging_dir / CONTEXT_NAME)
    current_protected = _protected_snapshot()
    if current_protected != context["protected_snapshot_before"]:
        raise RuntimeError("生成期间正文或 Supplementary Materials 发生变化")

    summary = pd.read_csv(staging_dir / SUMMARY_NAME, encoding="utf-8-sig")
    spec = _read_json(staging_dir / SPEC_NAME)
    category_checks = _validate_category_reconciliation(summary)
    docx_structure = _validate_docx(
        staging_dir / DOCX_NAME,
        expected_row_count=2 + len(spec["table"]["rows"]),
    )
    validation = {
        "validation_schema_version": "table_1_validation_v1",
        "release_status": "PENDING_RENDER_QA",
        "validated_utc": datetime.now(UTC).isoformat(),
        "data_checks": {
            "analysis_panel_freeze": "PASS",
            "sample_counts": "PASS",
            "one_landmark_per_participant": "PASS",
            "at_least_one_later_fi_per_participant": "PASS",
            "required_fields_complete": "PASS",
            "numeric_values_finite": "PASS",
            "later_time_positive": "PASS",
            "person_wave_unique": "PASS",
            "categorical_reconciliation": "PASS",
            "categorical_check_count": len(category_checks),
            "participant_identifiers_in_outputs": False,
        },
        "docx_structure": {"status": "PASS", **docx_structure},
        "protected_assets": {
            "status": "PASS",
            "snapshot_unchanged_during_build": True,
        },
        "render_qa": {
            "status": "PENDING",
            "page_count": None,
            "method": None,
            "inspected_utc": None,
            "checks": [],
        },
    }
    _write_json(staging_dir / VALIDATION_NAME, validation)
    manifest = _build_manifest(staging_dir, context, validation)
    _write_json(staging_dir / MANIFEST_NAME, manifest)
    print(json.dumps({"status": "FINALIZED_PENDING_RENDER_QA"}, ensure_ascii=False))


def _compare_summary(expected: pd.DataFrame, observed: pd.DataFrame) -> None:
    """比较正式 CSV 与冻结输入的内存重建结果。"""
    expected = expected.copy()
    observed = observed.copy()
    if list(expected.columns) != list(observed.columns):
        raise RuntimeError("summary_long CSV 列顺序漂移")
    text_columns = [
        "row_id",
        "section",
        "characteristic",
        "level",
        "cohort",
        "recovery_status",
        "display_group_label",
        "statistic_type",
        "source_column",
        "source_time",
        "display_value",
    ]
    for column in text_columns:
        left = expected[column].fillna("").astype(str).tolist()
        right = observed[column].fillna("").astype(str).tolist()
        if left != right:
            raise RuntimeError(f"summary_long CSV 文本列漂移：{column}")
    numeric_columns = [column for column in expected.columns if column not in text_columns]
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
    """只读重建并验证全部正式输出。"""
    missing = [name for name in KNOWN_OUTPUTS if not (output_dir / name).is_file()]
    if missing:
        raise RuntimeError("缺少正式输出：" + ", ".join(missing))
    freeze = analysis_panel.verify_analysis_panel_freeze()
    expected_summary, expected_spec, expected_diagnostics = build_table_data(freeze)
    observed_summary = pd.read_csv(output_dir / SUMMARY_NAME, encoding="utf-8-sig")
    _compare_summary(expected_summary, observed_summary)
    observed_spec = _read_json(output_dir / SPEC_NAME)
    if observed_spec != expected_spec:
        raise RuntimeError("display spec 与冻结输入重建结果不一致")
    _validate_category_reconciliation(observed_summary)
    docx_structure = _validate_docx(
        output_dir / DOCX_NAME,
        expected_row_count=2 + len(observed_spec["table"]["rows"]),
    )

    validation = _read_json(output_dir / VALIDATION_NAME)
    manifest = _read_json(output_dir / MANIFEST_NAME)
    if manifest["sample_counts"] != expected_diagnostics:
        raise RuntimeError("manifest sample counts 漂移")
    for name, record in manifest["outputs"].items():
        if _sha256_file(output_dir / name) != record["sha256"]:
            raise RuntimeError(f"输出哈希漂移：{name}")
    current_protected = _protected_snapshot()
    if current_protected != manifest["protected_snapshot"]:
        raise RuntimeError("正文或 Supplementary Materials 与生成快照不一致")
    if _sha256_file(FREEZE_PATH) != manifest["source_snapshot"]["analysis_panel_freeze"]["sha256"]:
        raise RuntimeError("Analysis Panel freeze 哈希漂移")
    if validation["release_status"] != "PASS" and not allow_pending_render:
        raise RuntimeError("render QA 尚未完成，正式输出不能标记为 PASS")
    result = {
        "status": "PASS" if validation["release_status"] == "PASS" else "PASS_PENDING_RENDER",
        "release_status": validation["release_status"],
        "sample_counts": expected_diagnostics,
        "docx_structure": docx_structure,
        "protected_assets_unchanged": True,
    }
    if verify_frozen and (output_dir / TABLE_FREEZE_NAME).is_file():
        result["table_1_freeze"] = verify_table_1_freeze(output_dir, emit=False)
    print(json.dumps(result, ensure_ascii=False))
    return result


def record_render_pass(output_dir: Path, page_count: int, method: str) -> None:
    """在人工逐页检查后登记 render QA PASS。"""
    if (output_dir / TABLE_FREEZE_NAME).exists():
        raise RuntimeError("Table 1 已冻结，不能改写 render QA 记录")
    if page_count not in (1, 2):
        raise RuntimeError("Table 1 渲染页数必须为 1 或 2")
    validation = _read_json(output_dir / VALIDATION_NAME)
    manifest = _read_json(output_dir / MANIFEST_NAME)
    if validation["docx_structure"]["status"] != "PASS":
        raise RuntimeError("DOCX 结构验证未通过，不能登记 render PASS")
    checks = [
        "no_clipping_or_overlap",
        "no_missing_glyphs",
        "no_abnormal_wrapping",
        "header_hierarchy_clear",
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
    print(
        json.dumps(
            {"status": "RENDER_QA_RECORDED", "page_count": page_count, "method": method},
            ensure_ascii=False,
        )
    )


def _release_asset_hashes(output_dir: Path) -> dict[str, dict[str, Any]]:
    """计算冻结所覆盖的五个正式资产哈希。"""
    return {
        name: {
            "sha256": _sha256_file(output_dir / name),
            "bytes": (output_dir / name).stat().st_size,
        }
        for name in KNOWN_OUTPUTS
    }


def _scientific_freeze_audit(output_dir: Path) -> dict[str, Any]:
    """复核两个显眼数字模式，仅输出汇总证据。"""
    analysis_freeze = analysis_panel.verify_analysis_panel_freeze()
    exposure_freeze = _read_json(EXPOSURE_FREEZE_PATH)
    covariate_manifest = _read_json(KLOSA_COVARIATE_MANIFEST_PATH)

    for cohort in ("HRS", "KLoSA"):
        record = exposure_freeze["cohorts"][cohort]
        if _sha256_file(ROOT / record["parquet_path"]) != record["parquet_sha256"]:
            raise RuntimeError(f"{cohort}: Exposure Panel freeze 哈希漂移")

    klosa_path = ROOT / analysis_freeze["cohorts"]["KLoSA"]["parquet_path"]
    klosa = pd.read_parquet(
        klosa_path,
        columns=[
            "person_id",
            "tminus1_wave",
            "fi_time_role",
            "primary_complete_case_eligible",
            "wealth_quintile",
        ],
    )
    klosa = klosa.loc[
        klosa["primary_complete_case_eligible"] & klosa["fi_time_role"].eq("landmark")
    ].copy()
    klosa_exposure_record = exposure_freeze["cohorts"]["KLoSA"]
    klosa_exposure = pd.read_parquet(
        ROOT / klosa_exposure_record["parquet_path"],
        columns=["person_id", "is_index_shock", "tminus1_wave", "wealth_tminus1"],
    )
    klosa_exposure = klosa_exposure.loc[klosa_exposure["is_index_shock"]].copy()
    klosa_joined = klosa.merge(
        klosa_exposure,
        on="person_id",
        how="left",
        validate="one_to_one",
        suffixes=("_analysis", "_exposure"),
    )
    if not klosa_joined["wealth_tminus1"].gt(0).all():
        raise RuntimeError("KLoSA: Table 1 主样本 pre-shock wealth 必须全部为正")
    klosa_q1 = klosa_joined.loc[klosa_joined["wealth_quintile"].eq(1)].copy()
    q1_wave_counts = {
        str(int(wave)): int(count)
        for wave, count in klosa_q1["tminus1_wave_analysis"]
        .value_counts()
        .sort_index()
        .items()
    }
    if len(klosa_q1) != 3 or q1_wave_counts != {"6": 3}:
        raise RuntimeError("KLoSA: Table 1 Q1 稀少模式与批准复核不一致")

    wave_reference = {
        int(row["wave"]): row
        for row in covariate_manifest["wealth_quintile_reference_by_wave"]
    }
    reference_checks: dict[str, Any] = {}
    for wave in range(1, 7):
        row = wave_reference[wave]
        q1_n = int(row["quintile_counts"]["1"])
        nonpositive_n = int(row["zero_wealth_n"]) + int(row["negative_wealth_n"])
        positive_q1_n = q1_n - nonpositive_n
        if wave <= 5 and positive_q1_n != 0:
            raise RuntimeError(f"KLoSA W{wave}: Q1 不应包含正财富")
        if wave == 6 and positive_q1_n != 41:
            raise RuntimeError("KLoSA W6: Q1 正财富人数与冻结 manifest 不一致")
        reference_checks[str(wave)] = {
            "reference_n": int(row["reference_n"]),
            "q1_n": q1_n,
            "zero_wealth_n": int(row["zero_wealth_n"]),
            "negative_wealth_n": int(row["negative_wealth_n"]),
            "positive_q1_n": positive_q1_n,
        }

    hrs_path = ROOT / analysis_freeze["cohorts"]["HRS"]["parquet_path"]
    hrs = pd.read_parquet(
        hrs_path,
        columns=[
            "person_id",
            "fi_time_role",
            "primary_complete_case_eligible",
            "recovery_status",
            "wealth_loss_fraction",
        ],
    )
    hrs = hrs.loc[
        hrs["primary_complete_case_eligible"]
        & hrs["fi_time_role"].eq("landmark")
        & hrs["recovery_status"].eq("substantial_recovery")
    ].copy()
    hrs_exposure_record = exposure_freeze["cohorts"]["HRS"]
    hrs_exposure = pd.read_parquet(
        ROOT / hrs_exposure_record["parquet_path"],
        columns=[
            "person_id",
            "is_index_shock",
            "wealth_tminus1",
            "wealth_t",
            "wealth_loss_fraction",
        ],
    )
    hrs_exposure = hrs_exposure.loc[hrs_exposure["is_index_shock"]].copy()
    hrs_joined = hrs.merge(
        hrs_exposure,
        on="person_id",
        how="left",
        validate="one_to_one",
        suffixes=("_analysis", "_exposure"),
    )
    loss = hrs_joined["wealth_loss_fraction_analysis"].astype(float)
    formula = (
        hrs_joined["wealth_tminus1"].astype(float)
        - hrs_joined["wealth_t"].astype(float)
    ) / hrs_joined["wealth_tminus1"].astype(float)
    formula_diff = float(np.max(np.abs(loss - formula)))
    exposure_diff = float(
        np.max(
            np.abs(
                loss - hrs_joined["wealth_loss_fraction_exposure"].astype(float)
            )
        )
    )
    above_one_n = int(loss.gt(1).sum())
    negative_shock_n = int(hrs_joined["wealth_t"].lt(0).sum())
    q3_percent = float(loss.quantile(0.75) * 100.0)
    if formula_diff != 0.0 or exposure_diff != 0.0:
        raise RuntimeError("HRS: wealth loss 并非原始公式值")
    if above_one_n != 589 or negative_shock_n != 589:
        raise RuntimeError("HRS: wealth loss >100% 模式与批准复核不一致")

    summary = pd.read_csv(output_dir / SUMMARY_NAME, encoding="utf-8-sig")
    hrs_table_row = summary.loc[
        summary["row_id"].eq("wealth_lost_percent")
        & summary["cohort"].eq("HRS")
        & summary["recovery_status"].eq("substantial_recovery")
    ]
    if len(hrs_table_row) != 1 or not np.isclose(
        float(hrs_table_row.iloc[0]["q3"]), q3_percent, rtol=0.0, atol=1e-12
    ):
        raise RuntimeError("HRS: Table 1 wealth-loss Q3 与原始连续值不一致")

    return {
        "klosa_pre_shock_wealth_quintile": {
            "status": "PASS",
            "reference_population": (
                "cohort × wave observed participants with finite nonmissing wealth; "
                "unweighted midrank; ties preserved; zero and negative wealth retained"
            ),
            "positive_wealth_gate_applied_after_quintile_construction": True,
            "primary_sample_q1_n": int(len(klosa_q1)),
            "primary_sample_q1_tminus1_wave_counts": q1_wave_counts,
            "primary_sample_q1_preshock_wealth_min": float(
                klosa_q1["wealth_tminus1"].min()
            ),
            "primary_sample_q1_preshock_wealth_median": float(
                klosa_q1["wealth_tminus1"].median()
            ),
            "primary_sample_q1_preshock_wealth_max": float(
                klosa_q1["wealth_tminus1"].max()
            ),
            "wave_1_to_6_reference_checks": reference_checks,
        },
        "hrs_substantial_recovery_wealth_loss": {
            "status": "PASS",
            "unclipped_unwinsorised_continuous_fraction": True,
            "n": int(len(hrs_joined)),
            "q1_percent": float(loss.quantile(0.25) * 100.0),
            "median_percent": float(loss.median() * 100.0),
            "q3_percent": q3_percent,
            "loss_above_100_n": above_one_n,
            "shock_wealth_negative_n": negative_shock_n,
            "formula_max_abs_difference": formula_diff,
            "analysis_vs_exposure_max_abs_difference": exposure_diff,
        },
    }


def verify_table_1_freeze(output_dir: Path, emit: bool = True) -> dict[str, Any]:
    """验证 Table 1 正式冻结记录及其全部资产。"""
    freeze_path = output_dir / TABLE_FREEZE_NAME
    if not freeze_path.is_file():
        raise RuntimeError("Table 1 freeze 记录不存在")
    freeze = _read_json(freeze_path)
    if freeze.get("freeze_status") != "FROZEN":
        raise RuntimeError("Table 1 freeze_status 不是 FROZEN")
    if freeze.get("freeze_record_schema_version") != "table_1_freeze_record_v1":
        raise RuntimeError("Table 1 freeze schema 不一致")
    for name, record in freeze["release_assets"].items():
        path = output_dir / name
        if not path.is_file() or _sha256_file(path) != record["sha256"]:
            raise RuntimeError(f"Table 1 冻结资产漂移：{name}")
    for record in freeze["dependencies"].values():
        path = ROOT / record["path"]
        if _sha256_file(path) != record["sha256"]:
            raise RuntimeError(f"Table 1 冻结依赖漂移：{record['path']}")
    if _protected_snapshot() != freeze["protected_snapshot"]:
        raise RuntimeError("Table 1 冻结时的正文或 Supplementary Materials 快照已变化")
    result = {
        "status": "PASS",
        "freeze_status": "FROZEN",
        "release_version": freeze["release_version"],
        "release_asset_count": len(freeze["release_assets"]),
    }
    if emit:
        print(json.dumps(result, ensure_ascii=False))
    return result


def freeze_release(output_dir: Path) -> None:
    """在严格检查和科学复核通过后冻结 Table 1 v1。"""
    freeze_path = output_dir / TABLE_FREEZE_NAME
    if freeze_path.exists():
        verify_table_1_freeze(output_dir)
        return
    check_only(output_dir, verify_frozen=False)
    validation = _read_json(output_dir / VALIDATION_NAME)
    if validation["release_status"] != "PASS":
        raise RuntimeError("Table 1 尚未通过 render QA，不能冻结")
    scientific_audits = _scientific_freeze_audit(output_dir)
    dependency_paths = {
        "analysis_panel_freeze": FREEZE_PATH,
        "exposure_panel_freeze": EXPOSURE_FREEZE_PATH,
        "covariate_panel_freeze": COVARIATE_FREEZE_PATH,
        "klosa_covariate_manifest": KLOSA_COVARIATE_MANIFEST_PATH,
        "covariate_builder": COVARIATE_CODE_PATH,
        "table_1_builder": Path(__file__),
        "generation_plan": PLAN_PATH,
    }
    freeze = {
        "freeze_record_schema_version": "table_1_freeze_record_v1",
        "freeze_status": "FROZEN",
        "release_version": "table_1_v1",
        "frozen_utc": datetime.now(UTC).isoformat(),
        "source_sample": "primary_complete_case_eligible == TRUE",
        "descriptive_statistics": "unweighted",
        "release_assets": _release_asset_hashes(output_dir),
        "dependencies": {
            name: {"path": _relative(path), "sha256": _sha256_file(path)}
            for name, path in dependency_paths.items()
        },
        "scientific_audits": scientific_audits,
        "render_qa": validation["render_qa"],
        "protected_snapshot": _protected_snapshot(),
        "git": _git_state(),
        "overwrite_policy": (
            "fail closed; frozen assets must not be overwritten; a substantive or "
            "schema change requires a new release version"
        ),
    }
    temporary = freeze_path.with_suffix(".json.tmp")
    _write_json(temporary, freeze)
    temporary.replace(freeze_path)
    verify_table_1_freeze(output_dir)


def parse_args() -> argparse.Namespace:
    """解析生成与验证子命令。"""
    parser = argparse.ArgumentParser(description="生成并验证正式 Table 1 聚合资产")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--staging-dir", type=Path, required=True)

    finalize_parser = subparsers.add_parser("finalize")
    finalize_parser.add_argument("--staging-dir", type=Path, required=True)

    check_parser = subparsers.add_parser("check-only")
    check_parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    check_parser.add_argument("--allow-pending-render", action="store_true")

    render_parser = subparsers.add_parser("record-render-pass")
    render_parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    render_parser.add_argument("--page-count", type=int, required=True)
    render_parser.add_argument("--method", required=True)

    freeze_parser = subparsers.add_parser("freeze")
    freeze_parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)

    verify_freeze_parser = subparsers.add_parser("verify-freeze")
    verify_freeze_parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
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
    elif args.command == "record-render-pass":
        record_render_pass(args.output_dir.resolve(), args.page_count, args.method)
    elif args.command == "freeze":
        freeze_release(args.output_dir.resolve())
    elif args.command == "verify-freeze":
        verify_table_1_freeze(args.output_dir.resolve())
    else:
        raise RuntimeError(f"未知命令：{args.command}")


if __name__ == "__main__":
    main()
