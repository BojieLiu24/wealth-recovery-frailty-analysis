#!/usr/bin/env python3
"""在正式分析面板之前审计实际访谈时间轴，不生成任何模型输入。"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[2]
COHORTS = ("HRS", "ELSA", "KLoSA")
EXPECTED_COUNTS = {
    "HRS": {"index_shocks": 8_204, "classifiable": 6_547},
    "ELSA": {"index_shocks": 1_284, "classifiable": 968},
    "KLoSA": {"index_shocks": 3_125, "classifiable": 2_755},
}
FREEZE_PATHS = {
    "spine": Path("00_admin/panel_manifests/spine/spine_v2_freeze.json"),
    "candidate": Path(
        "00_admin/panel_manifests/candidate_intervals/"
        "candidate_intervals_v3_freeze.json"
    ),
    "exposure": Path(
        "00_admin/panel_manifests/exposure_panel/exposure_panel_v1_freeze.json"
    ),
    "fi": Path("00_admin/panel_manifests/fi_panel/fi_panel_v2_freeze.json"),
}
SUMMARY_RELATIVE = Path("07_manuscript/preanalysis_timing_audit_summary.csv")
DECISION_RELATIVE = Path("07_manuscript/preanalysis_timing_audit_decision.md")
DIAGNOSTIC_FILENAME = "preanalysis_timing_audit_flagged_rows.csv"
KEY = ["cohort", "person_id", "wave"]
AUDIT_KEY = ["cohort", "person_id", "shock_definition"]

# Spine freeze 没有逐列 nullable 字段；这里按冻结数据字典锁定必须非空的结构列。
SPINE_NONNULL = {
    "cohort",
    "person_id",
    "wave",
    "wave_order",
    "nominal_year",
    "observation_status",
    "person_wave_scope_status",
    "scope_source_variable",
    "sample_cohort_source_variable",
    "interview_indicator_source_variable",
    "source_dataset_version",
    "spine_schema_version",
}

FLAG_COLUMNS = [
    "record_type",
    "cohort",
    "shock_definition",
    "person_id",
    "flag_reasons",
    "landmark_wave",
    "current_wave",
    "earlier_wave",
    "later_wave",
    "earlier_interview_year",
    "earlier_interview_month",
    "later_interview_year",
    "later_interview_month",
    "elapsed_months",
    "elapsed_years",
    "scheduled_wave_order_gap",
    "earlier_nominal_year",
    "later_nominal_year",
]

FORBIDDEN_OUTPUT_COLUMNS = {
    "fi_primary",
    "fi_common22",
    "recovery_status",
    "recovery_fraction_raw",
    "wealth_tminus1",
    "wealth_t",
    "wealth_tplus1",
}


class UpstreamIntegrityError(RuntimeError):
    """表示活动输入不再符合冻结契约，必须在输出前停止。"""


@dataclass(frozen=True)
class FrozenAsset:
    """记录一次已核验的冻结文件及其来源 manifest。"""

    component: str
    cohort: str
    data_path: Path
    data_sha256: str
    manifest_path: Path
    manifest_sha256: str
    freeze_path: Path
    freeze_sha256: str


@dataclass
class CohortAudit:
    """保存一个 cohort 与 shock definition 的聚合审计结果。"""

    summary: dict[str, Any]
    by_wave: pd.DataFrame
    flagged: pd.DataFrame


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    """流式计算文件 SHA-256。"""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    """以 UTF-8 读取 JSON，并要求顶层为对象。"""
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise UpstreamIntegrityError(f"JSON 顶层不是对象：{path}")
    return value


def _repo_path(root: Path, value: str | Path) -> Path:
    """把 freeze 中的相对或绝对路径解析为规范绝对路径。"""
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _component_record_paths(
    root: Path, component: str, record: dict[str, Any]
) -> tuple[Path, str, Path, str]:
    """兼容四种历史 freeze schema 的路径字段。"""
    if component == "spine":
        data_value = record["path"]
        data_hash = record["sha256"]
        manifest_value = record["build_manifest_path"]
        manifest_hash = record["build_manifest_sha256"]
    elif component == "candidate":
        data_value = record["path"]
        data_hash = record["sha256"]
        manifest_value = record["active_manifest_path"]
        manifest_hash = record["active_manifest_sha256"]
    else:
        data_value = record["parquet_path"]
        data_hash = record["parquet_sha256"]
        manifest_value = record["manifest_path"]
        manifest_hash = record["manifest_sha256"]
    return (
        _repo_path(root, data_value),
        str(data_hash),
        _repo_path(root, manifest_value),
        str(manifest_hash),
    )


def _manifest_output(manifest: dict[str, Any]) -> dict[str, Any]:
    """统一取得 cohort manifest 的 output 对象。"""
    output = manifest.get("output")
    if not isinstance(output, dict):
        raise UpstreamIntegrityError("cohort manifest 缺少 output 对象")
    return output


def _expected_schema_and_nullable(
    component: str,
    freeze: dict[str, Any],
    manifest: dict[str, Any],
) -> tuple[dict[str, str], dict[str, bool], list[str]]:
    """从活动 freeze/manifest 读取列顺序、dtype 与 nullable 契约。"""
    if component == "spine":
        schema = dict(freeze["shared_pandas_schema"])
        nullable = {name: name not in SPINE_NONNULL for name in schema}
        key = list(freeze["scope"]["primary_key"])
    elif component == "candidate":
        contract = list(freeze["schema_contract"])
        schema = {item["column"]: item["pandas_dtype"] for item in contract}
        nullable = {item["column"]: bool(item["nullable"]) for item in contract}
        key = list(freeze["scope"]["primary_key"])
    else:
        contract = list(manifest["schema"])
        schema = {item["name"]: item["dtype"] for item in contract}
        nullable = {item["name"]: bool(item["nullable"]) for item in contract}
        key = list(manifest["key"])
    return schema, nullable, key


def _parquet_pandas_schema(path: Path) -> dict[str, str]:
    """从 Parquet pandas metadata 读取原始列顺序与 pandas dtype。"""
    schema = pq.read_schema(path)
    metadata = schema.pandas_metadata
    if not metadata or "columns" not in metadata:
        raise UpstreamIntegrityError(f"Parquet 缺少 pandas schema metadata：{path}")
    return {
        str(item["field_name"]): str(item["numpy_type"])
        for item in metadata["columns"]
        if item.get("field_name") is not None
    }


def extract_shock_definition(exposure_freeze: dict[str, Any]) -> str:
    """从 Exposure freeze 的锁定 design metadata 取得 shock definition。"""
    try:
        decision = exposure_freeze["design_register"]["locked_decisions"][
            "shock_definition"
        ]
    except (KeyError, TypeError) as exc:
        raise UpstreamIntegrityError(
            "Exposure freeze 缺少锁定 shock_definition"
        ) from exc
    selected = str(decision.get("selected_value", "")).strip()
    if decision.get("status") != "locked" or not selected:
        raise UpstreamIntegrityError("Exposure freeze 的 shock_definition 未锁定")
    allowed = {
        item.strip()
        for item in str(decision.get("allowed_values", "")).split(";")
        if item.strip()
    }
    if allowed and selected not in allowed:
        raise UpstreamIntegrityError("锁定 shock_definition 不在 allowed_values 中")
    return selected


def verify_frozen_assets(
    root: Path,
) -> tuple[dict[str, dict[str, Any]], dict[tuple[str, str], FrozenAsset]]:
    """核对四类 freeze、数据、manifest、schema、nullability 与主键。"""
    root = root.resolve()
    freezes: dict[str, dict[str, Any]] = {}
    assets: dict[tuple[str, str], FrozenAsset] = {}
    for component, relative_freeze in FREEZE_PATHS.items():
        freeze_path = (root / relative_freeze).resolve()
        if not freeze_path.is_file():
            raise UpstreamIntegrityError(f"缺少 freeze：{freeze_path}")
        freeze = _read_json(freeze_path)
        if freeze.get("freeze_status") != "FROZEN":
            raise UpstreamIntegrityError(f"{component}: freeze_status 不是 FROZEN")
        if set(freeze.get("cohorts", {})) != set(COHORTS):
            raise UpstreamIntegrityError(f"{component}: freeze cohort 集合异常")
        freezes[component] = freeze
        freeze_hash = sha256_file(freeze_path)

        reference_schema: dict[str, str] | None = None
        for cohort in COHORTS:
            record = freeze["cohorts"][cohort]
            data_path, expected_data_hash, manifest_path, expected_manifest_hash = (
                _component_record_paths(root, component, record)
            )
            if not data_path.is_file() or not manifest_path.is_file():
                raise UpstreamIntegrityError(
                    f"{component}/{cohort}: freeze 登记文件缺失"
                )
            actual_data_hash = sha256_file(data_path)
            actual_manifest_hash = sha256_file(manifest_path)
            if actual_data_hash != expected_data_hash:
                raise UpstreamIntegrityError(
                    f"{component}/{cohort}: Parquet SHA-256 与 freeze 不一致"
                )
            if actual_manifest_hash != expected_manifest_hash:
                raise UpstreamIntegrityError(
                    f"{component}/{cohort}: manifest SHA-256 与 freeze 不一致"
                )

            manifest = _read_json(manifest_path)
            output = _manifest_output(manifest)
            manifest_output_path = _repo_path(root, output["path"])
            if manifest_output_path != data_path:
                raise UpstreamIntegrityError(
                    f"{component}/{cohort}: manifest 与 freeze 的活动路径不一致"
                )
            if str(output["sha256"]) != expected_data_hash:
                raise UpstreamIntegrityError(
                    f"{component}/{cohort}: manifest output hash 与 freeze 不一致"
                )
            expected_rows = int(record["row_count"])
            manifest_rows = int(output.get("row_count", manifest.get("row_count", -1)))
            parquet_rows = int(pq.ParquetFile(data_path).metadata.num_rows)
            if manifest_rows != expected_rows or parquet_rows != expected_rows:
                raise UpstreamIntegrityError(
                    f"{component}/{cohort}: frozen row count 不一致"
                )

            schema, nullable, key = _expected_schema_and_nullable(
                component, freeze, manifest
            )
            actual_schema = _parquet_pandas_schema(data_path)
            if list(actual_schema) != list(schema) or actual_schema != schema:
                raise UpstreamIntegrityError(
                    f"{component}/{cohort}: Parquet schema 与活动契约不一致"
                )
            if reference_schema is None:
                reference_schema = schema
            elif schema != reference_schema:
                raise UpstreamIntegrityError(f"{component}: cohort schema 不一致")
            if list(output.get("key", manifest.get("key", key))) != key:
                raise UpstreamIntegrityError(f"{component}/{cohort}: manifest key 异常")

            required_columns = list(dict.fromkeys(key + [
                name for name, is_nullable in nullable.items() if not is_nullable
            ]))
            structural = pd.read_parquet(data_path, columns=required_columns)
            if structural.duplicated(key).any():
                raise UpstreamIntegrityError(f"{component}/{cohort}: 主键不唯一")
            missing_required = {
                name: int(structural[name].isna().sum())
                for name in required_columns
                if structural[name].isna().any()
            }
            if missing_required:
                raise UpstreamIntegrityError(
                    f"{component}/{cohort}: 非空列出现缺失 {missing_required}"
                )
            if set(structural["cohort"].astype(str).unique()) != {cohort}:
                raise UpstreamIntegrityError(f"{component}/{cohort}: cohort 值异常")

            assets[(component, cohort)] = FrozenAsset(
                component=component,
                cohort=cohort,
                data_path=data_path,
                data_sha256=actual_data_hash,
                manifest_path=manifest_path,
                manifest_sha256=actual_manifest_hash,
                freeze_path=freeze_path,
                freeze_sha256=freeze_hash,
            )

    extract_shock_definition(freezes["exposure"])
    return freezes, assets


def verify_assets_unchanged(assets: dict[tuple[str, str], FrozenAsset]) -> None:
    """在发布审计结果前复核所有输入仍与 preflight 快照一致。"""
    for asset in assets.values():
        if sha256_file(asset.data_path) != asset.data_sha256:
            raise UpstreamIntegrityError(
                f"{asset.component}/{asset.cohort}: 审计期间数据发生变化"
            )
        if sha256_file(asset.manifest_path) != asset.manifest_sha256:
            raise UpstreamIntegrityError(
                f"{asset.component}/{asset.cohort}: 审计期间 manifest 发生变化"
            )
        if sha256_file(asset.freeze_path) != asset.freeze_sha256:
            raise UpstreamIntegrityError(
                f"{asset.component}: 审计期间 freeze 发生变化"
            )


def _nullable_int_equal(left: pd.Series, right: pd.Series) -> bool:
    """比较可能含 NA 的波次整数列。"""
    return left.astype("Int64").reset_index(drop=True).equals(
        right.astype("Int64").reset_index(drop=True)
    )


def _valid_year(series: pd.Series) -> pd.Series:
    """判断实际访谈年份是否在允许范围内。"""
    numeric = pd.to_numeric(series, errors="coerce")
    return numeric.notna() & numeric.between(1900, 2100, inclusive="both")


def _valid_month(series: pd.Series) -> pd.Series:
    """判断实际访谈月份是否为 1 至 12。"""
    numeric = pd.to_numeric(series, errors="coerce")
    return numeric.notna() & numeric.between(1, 12, inclusive="both")


def _percent(numerator: int | float, denominator: int | float) -> float:
    """返回 0--100 百分比；空分母返回 NaN。"""
    if not denominator:
        return float("nan")
    return 100.0 * float(numerator) / float(denominator)


def _distribution(prefix: str, months: Iterable[Any]) -> dict[str, Any]:
    """按整数月计算并输出以年为单位的描述统计。"""
    numeric = pd.to_numeric(pd.Series(list(months), dtype="Float64"), errors="coerce")
    values = numeric.dropna().astype(float) / 12.0
    result: dict[str, Any] = {f"{prefix}_n": int(len(values))}
    statistics = {
        "mean_years": float(values.mean()) if len(values) else math.nan,
        "sd_years": float(values.std(ddof=1)) if len(values) > 1 else math.nan,
        "median_years": float(values.quantile(0.50, interpolation="linear"))
        if len(values)
        else math.nan,
        "q1_years": float(values.quantile(0.25, interpolation="linear"))
        if len(values)
        else math.nan,
        "q3_years": float(values.quantile(0.75, interpolation="linear"))
        if len(values)
        else math.nan,
        "min_years": float(values.min()) if len(values) else math.nan,
        "p5_years": float(values.quantile(0.05, interpolation="linear"))
        if len(values)
        else math.nan,
        "p95_years": float(values.quantile(0.95, interpolation="linear"))
        if len(values)
        else math.nan,
        "max_years": float(values.max()) if len(values) else math.nan,
    }
    if len(values):
        statistics["iqr_years"] = statistics["q3_years"] - statistics["q1_years"]
    else:
        statistics["iqr_years"] = math.nan
    result.update({f"{prefix}_{key}": value for key, value in statistics.items()})
    return result


def classify_decision(
    *,
    timing_support_candidate_n: int,
    timing_complete_support_n: int,
    affected_participant_rate: float,
    affected_row_rate: float,
    max_wave_missing_rate: float,
    chronology_failure: bool,
) -> str:
    """按预先锁定的双阈值规则给出 cohort-specific 决策。"""
    if timing_support_candidate_n <= 0 or timing_complete_support_n <= 0:
        return "NO"
    if chronology_failure:
        return "NO"
    missing_exists = affected_participant_rate > 0 or affected_row_rate > 0
    if not missing_exists:
        return "YES"
    if (
        affected_participant_rate < 0.01
        and affected_row_rate < 0.01
        and max_wave_missing_rate < 0.05
    ):
        return "YES WITH DOCUMENTED LIMITATION"
    return "NO"


def _observation_flags(audit_rows: pd.DataFrame) -> list[dict[str, Any]]:
    """生成实际 year/month 缺失或非法的 observation-level flags。"""
    flags: list[dict[str, Any]] = []
    incomplete = audit_rows.loc[~audit_rows["current_timing_complete"]]
    for row in incomplete.itertuples(index=False):
        reasons: list[str] = []
        if pd.isna(row.interview_year):
            reasons.append("missing_interview_year")
        elif not 1900 <= int(row.interview_year) <= 2100:
            reasons.append("invalid_interview_year")
        if pd.isna(row.interview_month):
            reasons.append("missing_interview_month")
        elif not 1 <= int(row.interview_month) <= 12:
            reasons.append("invalid_interview_month")
        reasons.append(
            "landmark_timing_incomplete"
            if row.wave_order == row.landmark_wave_order
            else "followup_timing_incomplete"
        )
        flags.append(
            {
                "record_type": "observation",
                "cohort": row.cohort,
                "shock_definition": row.shock_definition,
                "person_id": row.person_id,
                "flag_reasons": ";".join(sorted(set(reasons))),
                "landmark_wave": row.landmark_wave,
                "current_wave": row.wave,
                "earlier_wave": pd.NA,
                "later_wave": pd.NA,
                "earlier_interview_year": pd.NA,
                "earlier_interview_month": pd.NA,
                "later_interview_year": row.interview_year,
                "later_interview_month": row.interview_month,
                "elapsed_months": pd.NA,
                "elapsed_years": pd.NA,
                "scheduled_wave_order_gap": pd.NA,
                "earlier_nominal_year": pd.NA,
                "later_nominal_year": row.nominal_year,
            }
        )
    return flags


def _interval_record(row: pd.Series) -> dict[str, Any]:
    """把 consecutive interval 行转换为统一诊断结构。"""
    return {
        "record_type": "interval",
        "cohort": row["cohort"],
        "shock_definition": row["shock_definition"],
        "person_id": row["person_id"],
        "landmark_wave": row["landmark_wave"],
        "current_wave": pd.NA,
        "earlier_wave": row["earlier_wave"],
        "later_wave": row["wave"],
        "earlier_interview_year": row["earlier_interview_year"],
        "earlier_interview_month": row["earlier_interview_month"],
        "later_interview_year": row["interview_year"],
        "later_interview_month": row["interview_month"],
        "elapsed_months": row["interval_months"],
        "elapsed_years": float(row["interval_months"]) / 12.0,
        "scheduled_wave_order_gap": row["scheduled_wave_order_gap"],
        "earlier_nominal_year": row["earlier_nominal_year"],
        "later_nominal_year": row["nominal_year"],
    }


def _combine_interval_flags(
    intervals: pd.DataFrame,
    nonpositive_rows: pd.DataFrame,
) -> list[dict[str, Any]]:
    """把 long、top-20 与 chronology flags 合并到唯一 interval 记录。"""
    records: dict[tuple[str, str, str, int, int], dict[str, Any]] = {}

    def add(record: dict[str, Any], reason: str) -> None:
        key = (
            str(record["cohort"]),
            str(record["shock_definition"]),
            str(record["person_id"]),
            int(record["earlier_wave"]),
            int(record["later_wave"]),
        )
        if key not in records:
            records[key] = {**record, "_reasons": set()}
        records[key]["_reasons"].add(reason)

    if not intervals.empty:
        for _, row in intervals.loc[intervals["interval_months"] > 42].iterrows():
            add(_interval_record(row), "interval_over_42_months")
        longest = intervals.sort_values(
            ["interval_months", "person_id", "earlier_wave", "wave"],
            ascending=[False, True, True, True],
            kind="mergesort",
        ).head(20)
        for _, row in longest.iterrows():
            add(_interval_record(row), "top20_longest_interval")
        for _, row in intervals.loc[intervals["interval_months"] <= 0].iterrows():
            add(_interval_record(row), "nonincreasing_interview_time")

    for _, row in nonpositive_rows.iterrows():
        record = {
            "record_type": "interval",
            "cohort": row["cohort"],
            "shock_definition": row["shock_definition"],
            "person_id": row["person_id"],
            "landmark_wave": row["landmark_wave"],
            "current_wave": pd.NA,
            "earlier_wave": row["landmark_wave"],
            "later_wave": row["wave"],
            "earlier_interview_year": row["landmark_interview_year"],
            "earlier_interview_month": row["landmark_interview_month"],
            "later_interview_year": row["interview_year"],
            "later_interview_month": row["interview_month"],
            "elapsed_months": row["elapsed_months"],
            "elapsed_years": float(row["elapsed_months"]) / 12.0,
            "scheduled_wave_order_gap": (
                row["wave_order"] - row["landmark_wave_order"]
            ),
            "earlier_nominal_year": row["landmark_nominal_year"],
            "later_nominal_year": row["nominal_year"],
        }
        reason = "negative_post_landmark_time" if row["elapsed_months"] < 0 else "zero_post_landmark_time"
        add(record, reason)

    combined: list[dict[str, Any]] = []
    for record in records.values():
        reasons = record.pop("_reasons")
        record["flag_reasons"] = ";".join(sorted(reasons))
        combined.append(record)
    return combined


def audit_cohort(
    *,
    cohort: str,
    shock_definition: str,
    exposure: pd.DataFrame,
    candidate: pd.DataFrame,
    fi: pd.DataFrame,
    spine: pd.DataFrame,
    expected_counts: dict[str, int] | None = None,
) -> CohortAudit:
    """对一个 cohort 的冻结 index shocks 构造只读 timing audit。"""
    index = exposure.loc[
        exposure["is_index_shock"].eq(True),
        [
            "cohort",
            "person_id",
            "candidate_shock_wave",
            "tminus1_wave",
            "tplus1_wave",
            "recovery_status",
        ],
    ].copy()
    if index.duplicated(["cohort", "person_id"]).any():
        raise UpstreamIntegrityError(f"{cohort}: index shock 不是一人一行")
    if index["tplus1_wave"].isna().any():
        raise UpstreamIntegrityError(f"{cohort}: index shock 缺少 tplus1_wave")
    index["shock_definition"] = shock_definition

    index_n = int(len(index))
    classifiable_n = int(index["recovery_status"].notna().sum())
    if expected_counts is not None:
        if index_n != int(expected_counts["index_shocks"]):
            raise UpstreamIntegrityError(f"{cohort}: index-shock count 未复现")
        if classifiable_n != int(expected_counts["classifiable"]):
            raise UpstreamIntegrityError(f"{cohort}: recovery-classifiable count 未复现")

    candidate_check = candidate.loc[
        :,
        [
            "cohort",
            "person_id",
            "candidate_shock_wave",
            "tminus1_wave",
            "tplus1_wave",
        ],
    ]
    checked = index.merge(
        candidate_check,
        on=["cohort", "person_id", "candidate_shock_wave"],
        how="left",
        validate="one_to_one",
        indicator=True,
        suffixes=("_exposure", "_candidate"),
    )
    if not checked["_merge"].eq("both").all():
        raise UpstreamIntegrityError(f"{cohort}: index shock 无精确 Candidate 匹配")
    for field in ("tminus1_wave", "tplus1_wave"):
        if not _nullable_int_equal(
            checked[f"{field}_exposure"], checked[f"{field}_candidate"]
        ):
            raise UpstreamIntegrityError(
                f"{cohort}: Exposure 与 Candidate 的 {field} 不一致"
            )
    index = checked.loc[
        :,
        [
            "cohort",
            "person_id",
            "shock_definition",
            "candidate_shock_wave",
            "tminus1_wave_exposure",
            "tplus1_wave_exposure",
            "recovery_status",
        ],
    ].rename(
        columns={
            "tminus1_wave_exposure": "tminus1_wave",
            "tplus1_wave_exposure": "landmark_wave",
        }
    )

    spine_fields = [
        "cohort",
        "person_id",
        "wave",
        "wave_order",
        "nominal_year",
        "successor_wave",
        "observed_in_wave",
        "interview_year",
        "interview_month",
        "interview_date",
    ]
    shock_spine = spine.loc[:, spine_fields].rename(
        columns={
            "wave": "candidate_shock_wave",
            "successor_wave": "shock_successor_wave",
        }
    )
    index = index.merge(
        shock_spine[
            ["cohort", "person_id", "candidate_shock_wave", "shock_successor_wave"]
        ],
        on=["cohort", "person_id", "candidate_shock_wave"],
        how="left",
        validate="one_to_one",
        indicator="shock_spine_merge",
    )
    if not index["shock_spine_merge"].eq("both").all():
        raise UpstreamIntegrityError(f"{cohort}: shock wave 无精确 Spine 匹配")
    if not _nullable_int_equal(index["landmark_wave"], index["shock_successor_wave"]):
        raise UpstreamIntegrityError(f"{cohort}: landmark 不是 shock wave 的 successor")
    index = index.drop(columns=["shock_successor_wave", "shock_spine_merge"])

    landmark_spine = spine.loc[:, spine_fields].rename(
        columns={
            "wave": "landmark_wave",
            "wave_order": "landmark_wave_order",
            "nominal_year": "landmark_nominal_year",
            "observed_in_wave": "landmark_observed_in_wave",
            "interview_year": "landmark_interview_year",
            "interview_month": "landmark_interview_month",
            "interview_date": "landmark_interview_date",
        }
    )
    index = index.merge(
        landmark_spine.drop(columns=["successor_wave"]),
        on=["cohort", "person_id", "landmark_wave"],
        how="left",
        validate="one_to_one",
        indicator="landmark_spine_merge",
    )
    if not index["landmark_spine_merge"].eq("both").all():
        raise UpstreamIntegrityError(f"{cohort}: landmark 无精确 Spine 匹配")
    index = index.drop(columns=["landmark_spine_merge"])
    index["landmark_timing_complete"] = _valid_year(
        index["landmark_interview_year"]
    ) & _valid_month(index["landmark_interview_month"])

    # FI 数值只用于判断是否有效；后续立即只保留键。
    valid_fi_keys = fi.loc[fi["fi_primary"].notna(), KEY].copy()
    valid_rows = valid_fi_keys.merge(
        index,
        on=["cohort", "person_id"],
        how="inner",
        validate="many_to_one",
    )
    current_spine = spine.loc[:, spine_fields].drop(columns=["successor_wave"])
    valid_rows = valid_rows.merge(
        current_spine,
        on=KEY,
        how="left",
        validate="one_to_one",
        indicator="current_spine_merge",
    )
    if not valid_rows["current_spine_merge"].eq("both").all():
        raise UpstreamIntegrityError(f"{cohort}: valid FI row 无精确 Spine 匹配")
    valid_rows = valid_rows.drop(columns=["current_spine_merge"])
    if not valid_rows["observed_in_wave"].eq(True).all():
        raise UpstreamIntegrityError(f"{cohort}: valid FI 出现在未确认访谈行")

    valid_rows["is_landmark"] = valid_rows["wave_order"].eq(
        valid_rows["landmark_wave_order"]
    ) & valid_rows["wave"].astype("Int64").eq(
        valid_rows["landmark_wave"].astype("Int64")
    )
    valid_rows["is_later"] = valid_rows["wave_order"] > valid_rows[
        "landmark_wave_order"
    ]
    landmark_valid_people = set(
        valid_rows.loc[valid_rows["is_landmark"], "person_id"].astype(str)
    )
    later_valid_people = set(
        valid_rows.loc[valid_rows["is_later"], "person_id"].astype(str)
    )
    index["landmark_fi_valid"] = index["person_id"].astype(str).isin(
        landmark_valid_people
    )
    index["has_later_valid_fi"] = index["person_id"].astype(str).isin(
        later_valid_people
    )
    index["timing_support_candidate"] = index["landmark_fi_valid"] & index[
        "has_later_valid_fi"
    ]
    candidate_people = index.loc[index["timing_support_candidate"], AUDIT_KEY].copy()
    audit_rows = valid_rows.loc[
        valid_rows["person_id"].astype(str).isin(
            set(candidate_people["person_id"].astype(str))
        )
        & (valid_rows["wave_order"] >= valid_rows["landmark_wave_order"])
    ].copy()
    if audit_rows.duplicated(AUDIT_KEY + ["wave"]).any():
        raise UpstreamIntegrityError(f"{cohort}: timing audit row key 不唯一")

    audit_rows["year_available"] = audit_rows["interview_year"].notna()
    audit_rows["month_available"] = audit_rows["interview_month"].notna()
    audit_rows["current_timing_complete"] = _valid_year(
        audit_rows["interview_year"]
    ) & _valid_month(audit_rows["interview_month"])
    audit_rows["time_calculable"] = audit_rows["current_timing_complete"] & audit_rows[
        "landmark_timing_complete"
    ]
    audit_rows["elapsed_months"] = pd.Series(pd.NA, index=audit_rows.index, dtype="Int64")
    calculable = audit_rows["time_calculable"]
    audit_rows.loc[calculable, "elapsed_months"] = (
        12
        * (
            audit_rows.loc[calculable, "interview_year"].astype("Int64")
            - audit_rows.loc[calculable, "landmark_interview_year"].astype("Int64")
        )
        + (
            audit_rows.loc[calculable, "interview_month"].astype("Int64")
            - audit_rows.loc[calculable, "landmark_interview_month"].astype("Int64")
        )
    ).astype("Int64")
    audit_rows["time_years"] = audit_rows["elapsed_months"].astype("Float64") / 12.0
    audit_rows["nominal_elapsed_years"] = (
        audit_rows["nominal_year"].astype("Int64")
        - audit_rows["landmark_nominal_year"].astype("Int64")
    ).astype("Int64")

    participant_incomplete = (
        audit_rows.assign(incomplete=~audit_rows["current_timing_complete"])
        .groupby(AUDIT_KEY, observed=True)["incomplete"]
        .any()
        .reset_index()
    )
    affected_participants_n = int(participant_incomplete["incomplete"].sum())
    affected_rows_n = int((~audit_rows["current_timing_complete"]).sum())
    candidate_n = int(len(candidate_people))
    valid_fi_rows_n = int(len(audit_rows))

    later_complete_people = set(
        audit_rows.loc[
            audit_rows["is_later"] & audit_rows["time_calculable"], "person_id"
        ].astype(str)
    )
    timing_complete_people = set(
        index.loc[
            index["timing_support_candidate"]
            & index["landmark_timing_complete"]
            & index["person_id"].astype(str).isin(later_complete_people),
            "person_id",
        ].astype(str)
    )
    index["timing_complete_support"] = index["person_id"].astype(str).isin(
        timing_complete_people
    ) & index["timing_support_candidate"]
    timing_complete_support_n = int(index["timing_complete_support"].sum())

    analysis_rows = audit_rows.loc[
        audit_rows["person_id"].astype(str).isin(timing_complete_people)
        & audit_rows["time_calculable"]
    ].copy()
    analysis_rows = analysis_rows.sort_values(
        AUDIT_KEY + ["wave_order"], kind="mergesort"
    )
    complete_landmark = analysis_rows["is_landmark"]
    landmark_zero_violations = int(
        analysis_rows.loc[complete_landmark, "elapsed_months"].ne(0).sum()
    )
    post_complete = analysis_rows.loc[analysis_rows["is_later"]].copy()
    nonpositive_rows = post_complete.loc[post_complete["elapsed_months"] <= 0].copy()

    grouped = analysis_rows.groupby(AUDIT_KEY, observed=True, sort=False)
    analysis_rows["earlier_wave"] = grouped["wave"].shift(1)
    analysis_rows["earlier_wave_order"] = grouped["wave_order"].shift(1)
    analysis_rows["earlier_interview_year"] = grouped["interview_year"].shift(1)
    analysis_rows["earlier_interview_month"] = grouped["interview_month"].shift(1)
    analysis_rows["earlier_nominal_year"] = grouped["nominal_year"].shift(1)
    analysis_rows["earlier_elapsed_months"] = grouped["elapsed_months"].shift(1)
    intervals = analysis_rows.loc[analysis_rows["earlier_wave"].notna()].copy()
    intervals["interval_months"] = (
        intervals["elapsed_months"].astype("Int64")
        - intervals["earlier_elapsed_months"].astype("Int64")
    )
    intervals["scheduled_wave_order_gap"] = (
        intervals["wave_order"].astype("Int64")
        - intervals["earlier_wave_order"].astype("Int64")
    )
    nonincreasing = intervals.loc[intervals["interval_months"] <= 0]
    nonincreasing_participants = int(
        nonincreasing[AUDIT_KEY].drop_duplicates().shape[0]
    )

    first_followup = (
        post_complete.sort_values(AUDIT_KEY + ["wave_order"], kind="mergesort")
        .groupby(AUDIT_KEY, observed=True, as_index=False)
        .first()
    )
    last_followup = (
        post_complete.sort_values(AUDIT_KEY + ["wave_order"], kind="mergesort")
        .groupby(AUDIT_KEY, observed=True, as_index=False)
        .last()
    )

    by_wave = (
        audit_rows.groupby(["cohort", "shock_definition", "wave"], observed=True)
        .agg(
            valid_fi_rows=("person_id", "size"),
            landmark_rows=("is_landmark", "sum"),
            followup_rows=("is_later", "sum"),
            year_available_n=("year_available", "sum"),
            month_available_n=("month_available", "sum"),
            timing_complete_n=("current_timing_complete", "sum"),
        )
        .reset_index()
    )
    by_wave["missing_year_n"] = by_wave["valid_fi_rows"] - by_wave[
        "year_available_n"
    ]
    by_wave["missing_month_n"] = by_wave["valid_fi_rows"] - by_wave[
        "month_available_n"
    ]
    by_wave["timing_missing_rows_n"] = by_wave["valid_fi_rows"] - by_wave[
        "timing_complete_n"
    ]
    by_wave["timing_missing_rate"] = (
        by_wave["timing_missing_rows_n"] / by_wave["valid_fi_rows"]
    )
    max_wave_missing_rate = (
        float(by_wave["timing_missing_rate"].max()) if len(by_wave) else 1.0
    )

    landmark_rows = audit_rows.loc[audit_rows["is_landmark"]]
    followup_rows = audit_rows.loc[audit_rows["is_later"]]
    affected_participant_rate = (
        affected_participants_n / candidate_n if candidate_n else 1.0
    )
    affected_row_rate = affected_rows_n / valid_fi_rows_n if valid_fi_rows_n else 1.0
    chronology_failure = bool(
        landmark_zero_violations
        or len(nonpositive_rows)
        or nonincreasing_participants
    )
    decision = classify_decision(
        timing_support_candidate_n=candidate_n,
        timing_complete_support_n=timing_complete_support_n,
        affected_participant_rate=affected_participant_rate,
        affected_row_rate=affected_row_rate,
        max_wave_missing_rate=max_wave_missing_rate,
        chronology_failure=chronology_failure,
    )

    nominal_comparison = post_complete.copy()
    nominal_comparison["actual_minus_nominal_years"] = (
        nominal_comparison["time_years"].astype(float)
        - nominal_comparison["nominal_elapsed_years"].astype(float)
    )
    nominal_diff = nominal_comparison["actual_minus_nominal_years"]
    nominal_abs = nominal_diff.abs()

    if cohort == "HRS":
        hrs_exact = post_complete.loc[
            post_complete["interview_date"].notna()
            & post_complete["landmark_interview_date"].notna()
        ].copy()
        hrs_exact["exact_time_years"] = (
            hrs_exact["interview_date"] - hrs_exact["landmark_interview_date"]
        ).dt.days / 365.25
        hrs_abs_diff = (
            hrs_exact["exact_time_years"] - hrs_exact["time_years"].astype(float)
        ).abs()
    else:
        hrs_abs_diff = pd.Series(dtype=float)

    summary: dict[str, Any] = {
        "cohort": cohort,
        "shock_definition": shock_definition,
        "decision": decision,
        "index_shock_n": index_n,
        "recovery_classifiable_n": classifiable_n,
        "landmark_actual_timing_complete_index_n": int(
            index["landmark_timing_complete"].sum()
        ),
        "landmark_fi_valid_n": int(index["landmark_fi_valid"].sum()),
        "timing_support_candidate_n": candidate_n,
        "timing_support_candidate_recovery_classifiable_n": int(
            (
                index["timing_support_candidate"]
                & index["recovery_status"].notna()
            ).sum()
        ),
        "timing_complete_support_n": timing_complete_support_n,
        "timing_complete_support_recovery_classifiable_n": int(
            (
                index["timing_complete_support"]
                & index["recovery_status"].notna()
            ).sum()
        ),
        "valid_fi_person_waves_n": valid_fi_rows_n,
        "followup_valid_fi_person_waves_n": int(len(followup_rows)),
        "current_actual_timing_complete_fi_rows_n": int(
            audit_rows["current_timing_complete"].sum()
        ),
        "time_calculable_fi_rows_n": int(audit_rows["time_calculable"].sum()),
        "landmark_year_available_n": int(landmark_rows["year_available"].sum()),
        "landmark_month_available_n": int(landmark_rows["month_available"].sum()),
        "landmark_year_month_complete_n": int(
            landmark_rows["current_timing_complete"].sum()
        ),
        "landmark_year_month_complete_pct": _percent(
            int(landmark_rows["current_timing_complete"].sum()), len(landmark_rows)
        ),
        "followup_year_available_n": int(followup_rows["year_available"].sum()),
        "followup_month_available_n": int(followup_rows["month_available"].sum()),
        "followup_year_month_complete_n": int(
            followup_rows["current_timing_complete"].sum()
        ),
        "followup_year_month_complete_pct": _percent(
            int(followup_rows["current_timing_complete"].sum()), len(followup_rows)
        ),
        "affected_timing_participants_n": affected_participants_n,
        "affected_timing_participants_pct": _percent(
            affected_participants_n, candidate_n
        ),
        "affected_timing_fi_rows_n": affected_rows_n,
        "affected_timing_fi_rows_pct": _percent(affected_rows_n, valid_fi_rows_n),
        "max_wave_timing_missing_pct": 100.0 * max_wave_missing_rate,
        "landmark_zero_time_violations": landmark_zero_violations,
        "nonpositive_post_landmark_time_rows": int(len(nonpositive_rows)),
        "nonincreasing_participants": nonincreasing_participants,
        "long_interval_over_42_months_n": int(
            (intervals["interval_months"] > 42).sum()
        ),
        "long_interval_adjacent_scheduled_waves_n": int(
            (
                (intervals["interval_months"] > 42)
                & intervals["scheduled_wave_order_gap"].eq(1)
            ).sum()
        ),
        "long_interval_skipped_scheduled_waves_n": int(
            (
                (intervals["interval_months"] > 42)
                & intervals["scheduled_wave_order_gap"].gt(1)
            ).sum()
        ),
    }
    summary.update(_distribution("consecutive_interval", intervals["interval_months"]))
    summary.update(
        _distribution(
            "landmark_to_first_followup", first_followup["elapsed_months"]
        )
    )
    summary.update(
        _distribution("landmark_to_last_followup", last_followup["elapsed_months"])
    )
    summary.update(
        {
            "actual_vs_nominal_n": int(len(nominal_diff)),
            "actual_minus_nominal_median_years": float(nominal_diff.median())
            if len(nominal_diff)
            else math.nan,
            "actual_minus_nominal_q1_years": float(
                nominal_diff.quantile(0.25, interpolation="linear")
            )
            if len(nominal_diff)
            else math.nan,
            "actual_minus_nominal_q3_years": float(
                nominal_diff.quantile(0.75, interpolation="linear")
            )
            if len(nominal_diff)
            else math.nan,
            "actual_minus_nominal_iqr_years": float(
                nominal_diff.quantile(0.75, interpolation="linear")
                - nominal_diff.quantile(0.25, interpolation="linear")
            )
            if len(nominal_diff)
            else math.nan,
            "actual_minus_nominal_min_years": float(nominal_diff.min())
            if len(nominal_diff)
            else math.nan,
            "actual_minus_nominal_max_years": float(nominal_diff.max())
            if len(nominal_diff)
            else math.nan,
            "actual_nominal_abs_diff_gt_0_25_n": int((nominal_abs > 0.25).sum()),
            "actual_nominal_abs_diff_gt_0_25_pct": _percent(
                int((nominal_abs > 0.25).sum()), len(nominal_abs)
            ),
            "actual_nominal_abs_diff_gt_0_5_n": int((nominal_abs > 0.5).sum()),
            "actual_nominal_abs_diff_gt_0_5_pct": _percent(
                int((nominal_abs > 0.5).sum()), len(nominal_abs)
            ),
            "hrs_exact_date_validation_n": int(len(hrs_abs_diff)),
            "hrs_month_exact_median_abs_diff_years": float(hrs_abs_diff.median())
            if len(hrs_abs_diff)
            else math.nan,
            "hrs_month_exact_p95_abs_diff_years": float(
                hrs_abs_diff.quantile(0.95, interpolation="linear")
            )
            if len(hrs_abs_diff)
            else math.nan,
            "hrs_month_exact_max_abs_diff_years": float(hrs_abs_diff.max())
            if len(hrs_abs_diff)
            else math.nan,
        }
    )

    flags = _observation_flags(audit_rows)
    flags.extend(_combine_interval_flags(intervals, nonpositive_rows))
    flagged = pd.DataFrame(flags, columns=FLAG_COLUMNS)
    if not flagged.empty:
        flagged = flagged.sort_values(
            ["record_type", "person_id", "earlier_wave", "later_wave", "current_wave"],
            kind="mergesort",
            na_position="last",
        ).reset_index(drop=True)

    if FORBIDDEN_OUTPUT_COLUMNS.intersection(summary):
        raise AssertionError("summary 意外包含禁止输出字段")
    if FORBIDDEN_OUTPUT_COLUMNS.intersection(flagged.columns):
        raise AssertionError("diagnostic 意外包含禁止输出字段")
    return CohortAudit(summary=summary, by_wave=by_wave, flagged=flagged)


def _format_value(value: Any, digits: int = 3) -> str:
    """为 Markdown 表格稳定格式化数值。"""
    if value is None or value is pd.NA:
        return "NA"
    if isinstance(value, (float, np.floating)):
        if math.isnan(float(value)):
            return "NA"
        return f"{float(value):.{digits}f}"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    return str(value).replace("|", "\\|")


def _markdown_table(rows: list[dict[str, Any]], columns: list[tuple[str, str]]) -> str:
    """生成紧凑的 GitHub Markdown 表格。"""
    headers = [label for _, label in columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(_format_value(row.get(name)) for name, _ in columns)
            + " |"
        )
    return "\n".join(lines)


def _overall_decision(summaries: list[dict[str, Any]]) -> str:
    """按最不利 cohort-specific 结果确定总体结论。"""
    rank = {"YES": 0, "YES WITH DOCUMENTED LIMITATION": 1, "NO": 2}
    return max((str(item["decision"]) for item in summaries), key=rank.__getitem__)


def build_decision_markdown(
    audits: list[CohortAudit],
    assets: dict[tuple[str, str], FrozenAsset],
    generated_at: str,
) -> str:
    """生成不含 person identifier 的 A--G 决策文档。"""
    summaries = [audit.summary for audit in audits]
    overall = _overall_decision(summaries)
    summary_by_cohort = {str(row["cohort"]): row for row in summaries}
    all_by_wave = pd.concat([audit.by_wave for audit in audits], ignore_index=True)

    if overall == "YES":
        wording = (
            "Time was measured in years since the landmark using each participant’s "
            "interview year and month, with the landmark assessment coded as time 0."
        )
    elif overall == "YES WITH DOCUMENTED LIMITATION":
        wording = (
            "Time was measured in years since the landmark using each participant’s "
            "interview year and month, with the landmark assessment coded as time 0. "
            "Observations without a recorded interview year or month were excluded "
            "from the timing-complete analysis; nominal wave timing was not substituted."
        )
    else:
        wording = (
            "No replacement wording is approved. Section 2.6 should retain an unresolved "
            "time-scale placeholder pending author review."
        )

    completeness_rows = [
        {
            "cohort": row["cohort"],
            "shock_definition": row["shock_definition"],
            "decision": row["decision"],
            "candidate_n": row["timing_support_candidate_n"],
            "complete_n": row["timing_complete_support_n"],
            "landmark_pct": row["landmark_year_month_complete_pct"],
            "followup_pct": row["followup_year_month_complete_pct"],
            "affected_person_pct": row["affected_timing_participants_pct"],
            "affected_row_pct": row["affected_timing_fi_rows_pct"],
            "max_wave_missing_pct": row["max_wave_timing_missing_pct"],
        }
        for row in summaries
    ]
    chronology_rows = [
        {
            "cohort": row["cohort"],
            "zero_landmark": row["landmark_zero_time_violations"],
            "nonpositive": row["nonpositive_post_landmark_time_rows"],
            "nonincreasing": row["nonincreasing_participants"],
        }
        for row in summaries
    ]
    interval_rows: list[dict[str, Any]] = []
    for row in summaries:
        for prefix, label in (
            ("consecutive_interval", "Consecutive valid FI"),
            ("landmark_to_first_followup", "Landmark to first later FI"),
            ("landmark_to_last_followup", "Landmark to last FI"),
        ):
            interval_rows.append(
                {
                    "cohort": row["cohort"],
                    "interval": label,
                    "n": row[f"{prefix}_n"],
                    "mean": row[f"{prefix}_mean_years"],
                    "sd": row[f"{prefix}_sd_years"],
                    "median": row[f"{prefix}_median_years"],
                    "q1": row[f"{prefix}_q1_years"],
                    "q3": row[f"{prefix}_q3_years"],
                    "min": row[f"{prefix}_min_years"],
                    "p5": row[f"{prefix}_p5_years"],
                    "p95": row[f"{prefix}_p95_years"],
                    "max": row[f"{prefix}_max_years"],
                }
            )
    nominal_rows = [
        {
            "cohort": row["cohort"],
            "n": row["actual_vs_nominal_n"],
            "median": row["actual_minus_nominal_median_years"],
            "q1": row["actual_minus_nominal_q1_years"],
            "q3": row["actual_minus_nominal_q3_years"],
            "min": row["actual_minus_nominal_min_years"],
            "max": row["actual_minus_nominal_max_years"],
            "gt025": row["actual_nominal_abs_diff_gt_0_25_pct"],
            "gt05": row["actual_nominal_abs_diff_gt_0_5_pct"],
        }
        for row in summaries
    ]
    long_rows = [
        {
            "cohort": row["cohort"],
            "over42": row["long_interval_over_42_months_n"],
            "adjacent": row["long_interval_adjacent_scheduled_waves_n"],
            "skipped": row["long_interval_skipped_scheduled_waves_n"],
            "max": row["consecutive_interval_max_years"],
        }
        for row in summaries
    ]
    stage_rows = [
        {
            "cohort": row["cohort"],
            "index": row["index_shock_n"],
            "recovery": row["recovery_classifiable_n"],
            "landmark_fi": row["landmark_fi_valid_n"],
            "candidate": row["timing_support_candidate_n"],
            "candidate_recovery": row[
                "timing_support_candidate_recovery_classifiable_n"
            ],
            "complete": row["timing_complete_support_n"],
            "complete_recovery": row[
                "timing_complete_support_recovery_classifiable_n"
            ],
        }
        for row in summaries
    ]
    by_wave_rows = all_by_wave.to_dict(orient="records")
    freeze_rows = []
    for component in FREEZE_PATHS:
        representative = assets[(component, COHORTS[0])]
        freeze_rows.append(
            {
                "component": component,
                "freeze_sha256": representative.freeze_sha256,
                "status": "verified before and after audit",
            }
        )

    hrs = summary_by_cohort["HRS"]
    lines = [
        "# Pre Analysis Longitudinal Timing Audit Decision",
        "",
        f"Generated: {generated_at}",
        "",
        f"**Overall decision: {overall}**",
        "",
        "This audit evaluates timing support only. Recovery classification is reported "
        "as a secondary bookkeeping branch and does not gate the primary timing-support set.",
        "",
        "## Audit flow",
        "",
        _markdown_table(
            stage_rows,
            [
                ("cohort", "Cohort"),
                ("index", "Index shock"),
                ("recovery", "Recovery classifiable QC"),
                ("landmark_fi", "Valid landmark FI"),
                ("candidate", "Timing-support candidate"),
                ("candidate_recovery", "Candidate and recovery classifiable"),
                ("complete", "Timing-complete support"),
                ("complete_recovery", "Complete and recovery classifiable"),
            ],
        ),
        "",
        "## A Is actual year and month timing sufficiently complete",
        "",
        _markdown_table(
            completeness_rows,
            [
                ("cohort", "Cohort"),
                ("shock_definition", "Shock definition"),
                ("decision", "Decision"),
                ("candidate_n", "Candidate N"),
                ("complete_n", "Timing-complete N"),
                ("landmark_pct", "Landmark complete %"),
                ("followup_pct", "Follow-up complete %"),
                ("affected_person_pct", "Affected participants %"),
                ("affected_row_pct", "Affected FI rows %"),
                ("max_wave_missing_pct", "Maximum wave missing %"),
            ],
        ),
        "",
        "The decision thresholds are strictly less than 1% for affected participants "
        "and FI rows and strictly less than 5% in every contributing wave.",
        "",
        "### Completeness by contributing wave",
        "",
        _markdown_table(
            by_wave_rows,
            [
                ("cohort", "Cohort"),
                ("shock_definition", "Shock definition"),
                ("wave", "Wave"),
                ("valid_fi_rows", "Valid FI rows"),
                ("landmark_rows", "Landmark rows"),
                ("followup_rows", "Follow-up rows"),
                ("year_available_n", "Year available"),
                ("month_available_n", "Month available"),
                ("timing_complete_n", "Both complete"),
                ("timing_missing_rows_n", "Timing missing"),
            ],
        ),
        "",
        "## B Is the proposed model-time definition supported",
        "",
        "The assessed harmonised definition is:",
        "",
        "`time_years = [12 × (interview_year_current − interview_year_landmark) "
        "+ (interview_month_current − interview_month_landmark)] / 12`",
        "",
        f"Decision: **{overall}**.",
        "",
        "## C Are post-landmark times strictly positive and increasing",
        "",
        _markdown_table(
            chronology_rows,
            [
                ("cohort", "Cohort"),
                ("zero_landmark", "Landmark nonzero rows"),
                ("nonpositive", "Nonpositive post-landmark rows"),
                ("nonincreasing", "Participants with non-increasing time"),
            ],
        ),
        "",
        "## D Are long intervals plausible survey-timing features",
        "",
        _markdown_table(
            long_rows,
            [
                ("cohort", "Cohort"),
                ("over42", "Intervals over 42 months"),
                ("adjacent", "Adjacent scheduled waves"),
                ("skipped", "Spanning skipped waves"),
                ("max", "Maximum consecutive interval years"),
            ],
        ),
        "",
        "Intervals over 42 months are diagnostic only. Positive, ordered intervals are "
        "retained because they can reflect skipped observed waves, temporary non-response, "
        "extended fieldwork, or cohort-specific scheduling; ID-level rows remain available "
        "in the private diagnostics for review.",
        "",
        "### Interval distributions in years",
        "",
        _markdown_table(
            interval_rows,
            [
                ("cohort", "Cohort"),
                ("interval", "Interval"),
                ("n", "N"),
                ("mean", "Mean"),
                ("sd", "SD"),
                ("median", "Median"),
                ("q1", "Q1"),
                ("q3", "Q3"),
                ("min", "Min"),
                ("p5", "P5"),
                ("p95", "P95"),
                ("max", "Max"),
            ],
        ),
        "",
        "## E Is nominal wave timing an acceptable substitute",
        "",
        "**NO.** Nominal time is defined only as `nominal_year_current − "
        "nominal_year_landmark` from Full Spine and is used only for this comparison:",
        "",
        _markdown_table(
            nominal_rows,
            [
                ("cohort", "Cohort"),
                ("n", "Post-landmark N"),
                ("median", "Median difference years"),
                ("q1", "Q1"),
                ("q3", "Q3"),
                ("min", "Min"),
                ("max", "Max"),
                ("gt025", "Absolute difference >0.25 years %"),
                ("gt05", "Absolute difference >0.5 years %"),
            ],
        ),
        "",
        "### HRS exact-date validation",
        "",
        _markdown_table(
            [
                {
                    "n": hrs["hrs_exact_date_validation_n"],
                    "median": hrs["hrs_month_exact_median_abs_diff_years"],
                    "p95": hrs["hrs_month_exact_p95_abs_diff_years"],
                    "max": hrs["hrs_month_exact_max_abs_diff_years"],
                }
            ],
            [
                ("n", "N"),
                ("median", "Median absolute difference years"),
                ("p95", "P95 absolute difference years"),
                ("max", "Maximum absolute difference years"),
            ],
        ),
        "",
        "HRS exact dates validate the common month-level approximation only and do not "
        "define a cohort-specific production time scale.",
        "",
        "## F Manuscript wording recommendation",
        "",
        f"> {wording}",
        "",
        "The manuscript itself was not modified by this audit.",
        "",
        "## G Analysis Panel implementation recommendation",
        "",
        "If the decision is YES or YES WITH DOCUMENTED LIMITATION, a future Analysis "
        "Panel should source `interview_year` and `interview_month` from frozen Full Spine "
        "using the exact `cohort + person_id + wave` key, use the month-difference formula "
        "above, code the landmark as zero, and never substitute nominal timing. Rows with "
        "unavailable actual timing should follow the documented timing-complete restriction. "
        "No cohort-specific exception is recommended. This audit does not implement that panel.",
        "",
        "## Input integrity",
        "",
        _markdown_table(
            freeze_rows,
            [
                ("component", "Frozen component"),
                ("freeze_sha256", "Freeze SHA-256"),
                ("status", "Status"),
            ],
        ),
        "",
        "No FI values, recovery-stratified FI summaries, regression estimates, p values, "
        "confidence intervals, or predicted trajectories were produced.",
        "",
    ]
    return "\n".join(lines)


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    """以 UTF-8 和 LF 写入 CSV。"""
    frame.to_csv(path, index=False, encoding="utf-8", lineterminator="\n")


def _publish_transactionally(staged_to_final: list[tuple[Path, Path]]) -> None:
    """在所有文件准备完成后发布，并在异常时恢复已有版本。"""
    backups: list[tuple[Path, Path]] = []
    published: list[Path] = []
    backup_root = staged_to_final[0][0].parent / "backups"
    backup_root.mkdir(parents=True, exist_ok=True)
    try:
        for position, (staged, final) in enumerate(staged_to_final):
            final.parent.mkdir(parents=True, exist_ok=True)
            if final.exists():
                backup = backup_root / f"{position}_{final.name}"
                shutil.copy2(final, backup)
                backups.append((backup, final))
            os.replace(staged, final)
            published.append(final)
    except Exception:
        for final in reversed(published):
            if final.exists():
                final.unlink()
        for backup, final in backups:
            shutil.copy2(backup, final)
        raise


def run_audit(root: Path = ROOT) -> tuple[pd.DataFrame, str, list[Path]]:
    """运行三队列审计并在完整成功后发布五个输出。"""
    root = root.resolve()
    freezes, assets = verify_frozen_assets(root)
    shock_definition = extract_shock_definition(freezes["exposure"])
    audits: list[CohortAudit] = []

    for cohort in COHORTS:
        exposure = pd.read_parquet(
            assets[("exposure", cohort)].data_path,
            columns=[
                "cohort",
                "person_id",
                "candidate_shock_wave",
                "tminus1_wave",
                "tplus1_wave",
                "is_index_shock",
                "recovery_status",
            ],
        )
        candidate = pd.read_parquet(
            assets[("candidate", cohort)].data_path,
            columns=[
                "cohort",
                "person_id",
                "candidate_shock_wave",
                "tminus1_wave",
                "tplus1_wave",
            ],
        )
        fi = pd.read_parquet(
            assets[("fi", cohort)].data_path,
            columns=["cohort", "person_id", "wave", "fi_primary"],
        )
        spine = pd.read_parquet(
            assets[("spine", cohort)].data_path,
            columns=[
                "cohort",
                "person_id",
                "wave",
                "wave_order",
                "nominal_year",
                "successor_wave",
                "observed_in_wave",
                "interview_year",
                "interview_month",
                "interview_date",
            ],
        )
        audits.append(
            audit_cohort(
                cohort=cohort,
                shock_definition=shock_definition,
                exposure=exposure,
                candidate=candidate,
                fi=fi,
                spine=spine,
                expected_counts=EXPECTED_COUNTS[cohort],
            )
        )

    summary = pd.DataFrame([audit.summary for audit in audits])
    if len(summary) != len(COHORTS):
        raise AssertionError("summary 必须一行对应一个 cohort × shock definition")
    if summary[["cohort", "shock_definition"]].duplicated().any():
        raise AssertionError("summary 的 cohort × shock definition 不唯一")
    if FORBIDDEN_OUTPUT_COLUMNS.intersection(summary.columns):
        raise AssertionError("summary 包含禁止的 substantive 字段")

    generated_at = datetime.now().astimezone().isoformat(timespec="seconds")
    decision = build_decision_markdown(audits, assets, generated_at)
    verify_assets_unchanged(assets)

    staging_parent = root / "03_output" / "tmp"
    staging_parent.mkdir(parents=True, exist_ok=True)
    final_paths: list[Path] = []
    with tempfile.TemporaryDirectory(
        prefix="preanalysis_timing_", dir=staging_parent
    ) as temporary:
        staging = Path(temporary)
        staged_summary = staging / SUMMARY_RELATIVE.name
        staged_decision = staging / DECISION_RELATIVE.name
        _write_csv(summary, staged_summary)
        staged_decision.write_text(decision, encoding="utf-8", newline="\n")
        mappings: list[tuple[Path, Path]] = [
            (staged_summary, root / SUMMARY_RELATIVE),
            (staged_decision, root / DECISION_RELATIVE),
        ]
        for audit in audits:
            cohort = str(audit.summary["cohort"])
            staged_flagged = staging / f"{cohort}_{DIAGNOSTIC_FILENAME}"
            _write_csv(audit.flagged, staged_flagged)
            final_flagged = (
                root
                / "01_data"
                / "derived"
                / "panel_components"
                / cohort
                / "_diagnostics"
                / DIAGNOSTIC_FILENAME
            )
            mappings.append((staged_flagged, final_flagged))
        _publish_transactionally(mappings)
        final_paths = [final for _, final in mappings]

    verify_assets_unchanged(assets)
    return summary, decision, final_paths


def parse_args() -> argparse.Namespace:
    """解析只读审计命令行。"""
    parser = argparse.ArgumentParser(
        description="Audit actual interview timing before Analysis Panel construction."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=ROOT,
        help="Repository root; defaults to the script's repository.",
    )
    return parser.parse_args()


def main() -> None:
    """命令行入口。"""
    args = parse_args()
    summary, _decision, paths = run_audit(args.root)
    for row in summary[["cohort", "shock_definition", "decision"]].itertuples(
        index=False
    ):
        print(f"{row.cohort} {row.shock_definition}: {row.decision}")
    for path in paths:
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
