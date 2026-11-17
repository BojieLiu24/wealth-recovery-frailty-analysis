"""三队列 Exposure Panel v1：首次财富冲击、恢复分类、QC 与冻结。"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from panel_core import RUNNER_VERSION, sha256_file
from wealth_panel import (
    FREEZE_PATH as WEALTH_FREEZE_PATH,
    WEALTH_PANEL_COLUMNS,
    WEALTH_PANEL_DTYPES,
    git_provenance,
)


ROOT = Path(__file__).resolve().parents[2]
ADMIN = ROOT / "00_admin"
DESIGN_REGISTER_PATH = ADMIN / "panel_design_register.csv"
MANIFEST_ROOT = ADMIN / "panel_manifests"
EXPOSURE_MANIFEST_ROOT = MANIFEST_ROOT / "exposure_panel"
DERIVED_ROOT = ROOT / "01_data" / "derived" / "panel_components"
REVIEW_PATH = ROOT / "04_collaboration" / "exposure_panel_review_v1.xlsx"
DICTIONARY_PATH = EXPOSURE_MANIFEST_ROOT / "exposure_panel_data_dictionary.md"
FREEZE_PATH = EXPOSURE_MANIFEST_ROOT / "exposure_panel_v1_freeze.json"
ARCHIVED_FREEZE_PATH = (
    EXPOSURE_MANIFEST_ROOT / "exposure_panel_v1_pre_window_freeze_archived.json"
)
WITHDRAWAL_PATH = (
    EXPOSURE_MANIFEST_ROOT / "exposure_panel_v1_pre_window_withdrawal.json"
)
CANDIDATE_FREEZE_PATH = (
    MANIFEST_ROOT / "candidate_intervals" / "candidate_intervals_v3_freeze.json"
)
ELIGIBILITY_FREEZE_PATH = (
    MANIFEST_ROOT / "eligibility_base" / "eligibility_base_v2_freeze.json"
)

COHORTS = ("HRS", "ELSA", "KLoSA")
RELEASE_VERSION = "v1"
BUILDER_VERSION = "exposure_panel_builder_v1.1.0"
SPEC_VERSION = "wealth_recovery_exposure_spec_v1.1"
SHOCK_THRESHOLD = 0.75
PRIMARY_RECOVERY_THRESHOLD = 0.50
SENSITIVITY_THRESHOLDS = (0.25, 1.00)
AUDIT_ONLY_THRESHOLDS = (0.75,)
SHOCK_SEARCH_WINDOWS = {
    "HRS": (3, 14),
    "ELSA": (2, 8),
    "KLoSA": (2, 7),
}
EXPECTED_OLD_FREEZE_SHA256 = (
    "177b7394eae73a65cf7f78cb3cbd830f0ea9b9d9e27e3ac4f221bca1fa5de70d"
)
BASELINE_KNOWN_FAILURE = (
    "02_code/01_panel_build/tests/test_outcome_components.py::"
    "test_supplementary_table_s3_reflects_v2_boundary"
)

KEY_COLUMNS = ("cohort", "person_id", "candidate_shock_wave")
EXPOSURE_PANEL_COLUMNS = (
    "cohort",
    "person_id",
    "candidate_shock_wave",
    "tminus1_wave",
    "tplus1_wave",
    "wealth_tminus1",
    "wealth_t",
    "wealth_tplus1",
    "wealth_loss_fraction",
    "major_shock",
    "is_index_shock",
    "recovery_fraction_raw",
    "recovery_status",
)
EXPOSURE_PANEL_DTYPES = {
    "cohort": "string",
    "person_id": "string",
    "candidate_shock_wave": "Int16",
    "tminus1_wave": "Int16",
    "tplus1_wave": "Int16",
    "wealth_tminus1": "float64",
    "wealth_t": "float64",
    "wealth_tplus1": "float64",
    "wealth_loss_fraction": "float64",
    "major_shock": "boolean",
    "is_index_shock": "boolean",
    "recovery_fraction_raw": "float64",
    "recovery_status": "string",
}
EXPOSURE_PANEL_NULLABLE = {
    "cohort": False,
    "person_id": False,
    "candidate_shock_wave": False,
    "tminus1_wave": False,
    "tplus1_wave": True,
    "wealth_tminus1": True,
    "wealth_t": True,
    "wealth_tplus1": True,
    "wealth_loss_fraction": True,
    "major_shock": True,
    "is_index_shock": False,
    "recovery_fraction_raw": True,
    "recovery_status": True,
}
RECOVERY_STATUS_VALUES = {
    "substantial_recovery",
    "persistent_or_limited_recovery",
}
EXPECTED_RELEASE_COUNTS = {
    "HRS": {
        "row_count": 678510,
        "index_shocks": 8204,
        "classifiable": 6547,
        "primary_recovery": 2088,
        "primary_comparator": 4459,
        "recovery_ge_0.25_n": 2819,
        "recovery_ge_0.75_n": 1581,
        "recovery_ge_1.00_n": 1124,
    },
    "ELSA": {
        "row_count": 195111,
        "index_shocks": 1284,
        "classifiable": 968,
        "primary_recovery": 362,
        "primary_comparator": 606,
        "recovery_ge_0.25_n": 474,
        "recovery_ge_0.75_n": 280,
        "recovery_ge_1.00_n": 204,
    },
    "KLoSA": {
        "row_count": 89392,
        "index_shocks": 3125,
        "classifiable": 2755,
        "primary_recovery": 1024,
        "primary_comparator": 1731,
        "recovery_ge_0.25_n": 1191,
        "recovery_ge_0.75_n": 896,
        "recovery_ge_1.00_n": 693,
    },
}
EXPECTED_CHRONOLOGY_COMPARISON = {
    "HRS": {
        "same_person_same_wave": 8147,
        "shifted_to_later_in_window_shock": 57,
        "dropped_no_window_index": 1006,
        "new_person": 0,
    },
    "ELSA": {
        "same_person_same_wave": 1284,
        "shifted_to_later_in_window_shock": 0,
        "dropped_no_window_index": 244,
        "new_person": 0,
    },
    "KLoSA": {
        "same_person_same_wave": 3125,
        "shifted_to_later_in_window_shock": 0,
        "dropped_no_window_index": 548,
        "new_person": 0,
    },
}


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    """以 UTF-8 读取 CSV。"""
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def _json_write(path: Path, payload: dict[str, Any]) -> None:
    """以 UTF-8 原子写入 JSON。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _stable_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def specification() -> dict[str, Any]:
    """返回锁定的 Exposure v1 机器可读定义。"""
    return {
        "spec_version": SPEC_VERSION,
        "grain": "one row per cohort-person-candidate shock interval",
        "shock_formula": "(wealth_tminus1 - wealth_t) / wealth_tminus1",
        "shock_evaluable_rule": (
            "structural_eligibility_status == pass and both wealth values are valid "
            "and wealth_tminus1 > 0"
        ),
        "shock_threshold": SHOCK_THRESHOLD,
        "shock_search_windows": {
            cohort: {"first_wave": bounds[0], "last_wave": bounds[1]}
            for cohort, bounds in SHOCK_SEARCH_WINDOWS.items()
        },
        "major_shock_scope": (
            "all evaluable candidate intervals, regardless of analytic shock-search window"
        ),
        "index_rule": (
            "chronological first qualifying major shock within the cohort-specific "
            "analytic shock-search window by candidate_interval_order, locked before "
            "t+1 wealth is read"
        ),
        "recovery_formula": (
            "(wealth_tplus1 - wealth_t) / (wealth_tminus1 - wealth_t)"
        ),
        "primary_recovery_threshold": PRIMARY_RECOVERY_THRESHOLD,
        "primary_labels": sorted(RECOVERY_STATUS_VALUES),
        "threshold_sensitivities": list(SENSITIVITY_THRESHOLDS),
        "audit_only_thresholds": list(AUDIT_ONLY_THRESHOLDS),
        "raw_fraction_policy": "preserve without clipping or winsorisation",
        "missing_tplus1_policy": "recovery fraction and status remain NA",
        "continuous_linear_sensitivity": False,
        "divisor_change_role": "exposure-only QC; no automatic inferential model",
        "outcome_or_fi_used": False,
    }


def specification_hash() -> str:
    return _stable_hash(specification())


def verify_locked_design() -> dict[str, dict[str, str]]:
    """确认 Exposure v1 的 shock、landmark、threshold 与窗口已锁定。"""
    rows = {row["decision_id"]: row for row in _read_csv_rows(DESIGN_REGISTER_PATH)}
    expected = {
        "shock_definition": "shock75_v0",
        "recovery_landmark_wave": "t+1",
        "recovery_threshold": "0.50",
        "shock_search_window_HRS": "W3-W14",
        "shock_search_window_ELSA": "W2-W8",
        "shock_search_window_KLoSA": "W2-W7",
    }
    locked: dict[str, dict[str, str]] = {}
    for decision_id, selected in expected.items():
        row = rows.get(decision_id)
        if row is None:
            raise RuntimeError(f"design register 缺少 {decision_id}")
        if row["status"] != "locked" or row["selected_value"] != selected:
            raise RuntimeError(
                f"design register 的 {decision_id} 必须 locked 且 selected_value={selected}"
            )
        locked[decision_id] = row
    note = locked["recovery_threshold"]["resolution_note"]
    for token in ("0.25", "1.00", "0.75", "without clipping"):
        if token not in note:
            raise RuntimeError(f"recovery_threshold 未记录必要边界：{token}")
    return locked


def _verify_interval_freeze(
    *,
    panel_name: str,
    freeze_path: Path,
    cohort: str,
    data: pd.DataFrame,
    path: Path,
    manifest: dict[str, Any],
) -> dict[str, Any]:
    """核对 Candidate 或 Eligibility 的活动文件与 freeze。"""
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    if freeze.get("freeze_status") != "FROZEN":
        raise RuntimeError(f"{panel_name}: freeze_status 不是 FROZEN")
    record = freeze["cohorts"][cohort]
    frozen_path = _repo_path(record["path"])
    if path.resolve() != frozen_path.resolve():
        raise RuntimeError(f"{cohort}: {panel_name} 活动路径与 freeze 不一致")
    actual_sha = sha256_file(path)
    if actual_sha != record["sha256"]:
        raise RuntimeError(f"{cohort}: {panel_name} 文件哈希与 freeze 不一致")
    if len(data) != int(record["row_count"]):
        raise RuntimeError(f"{cohort}: {panel_name} 行数与 freeze 不一致")
    manifest_path = _repo_path(record["active_manifest_path"])
    manifest_sha = sha256_file(manifest_path)
    if manifest_sha != record["active_manifest_sha256"]:
        raise RuntimeError(f"{cohort}: {panel_name} manifest 哈希与 freeze 不一致")
    if json.loads(manifest_path.read_text(encoding="utf-8")) != manifest:
        raise RuntimeError(f"{cohort}: {panel_name} loader manifest 与活动文件不一致")
    return {
        "path": _relative(path),
        "sha256": actual_sha,
        "manifest_path": _relative(manifest_path),
        "manifest_sha256": manifest_sha,
        "freeze_path": _relative(freeze_path),
        "freeze_sha256": sha256_file(freeze_path),
        "row_count": len(data),
        "schema": {column: str(data[column].dtype) for column in data.columns},
    }


def _load_wealth_panel(cohort: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    """从 Wealth Panel v1 freeze 读取并验证活动财富面板。"""
    freeze = json.loads(WEALTH_FREEZE_PATH.read_text(encoding="utf-8"))
    if freeze.get("freeze_status") != "FROZEN":
        raise RuntimeError("Wealth Panel v1 freeze_status 不是 FROZEN")
    record = freeze["cohorts"][cohort]
    path = _repo_path(record["parquet_path"])
    manifest_path = _repo_path(record["manifest_path"])
    if not path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError(f"{cohort}: Wealth Panel v1 资产不完整")
    actual_sha = sha256_file(path)
    manifest_sha = sha256_file(manifest_path)
    if actual_sha != record["parquet_sha256"]:
        raise RuntimeError(f"{cohort}: Wealth Panel v1 parquet 哈希不一致")
    if manifest_sha != record["manifest_sha256"]:
        raise RuntimeError(f"{cohort}: Wealth Panel v1 manifest 哈希不一致")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["output"]["sha256"] != actual_sha:
        raise RuntimeError(f"{cohort}: Wealth manifest 输出哈希不一致")
    panel = pd.read_parquet(path)
    if list(panel.columns) != list(WEALTH_PANEL_COLUMNS):
        raise RuntimeError(f"{cohort}: Wealth Panel v1 列结构不一致")
    actual_dtypes = {column: str(panel[column].dtype) for column in panel.columns}
    if actual_dtypes != WEALTH_PANEL_DTYPES:
        raise RuntimeError(f"{cohort}: Wealth Panel v1 dtype 不一致：{actual_dtypes}")
    if len(panel) != int(record["row_count"]):
        raise RuntimeError(f"{cohort}: Wealth Panel v1 行数与 freeze 不一致")
    if panel.duplicated(["cohort", "person_id", "wave"]).any():
        raise RuntimeError(f"{cohort}: Wealth Panel v1 主键不唯一")
    return panel, {
        "path": _relative(path),
        "sha256": actual_sha,
        "manifest_path": _relative(manifest_path),
        "manifest_sha256": manifest_sha,
        "freeze_path": _relative(WEALTH_FREEZE_PATH),
        "freeze_sha256": sha256_file(WEALTH_FREEZE_PATH),
        "row_count": len(panel),
        "schema": actual_dtypes,
    }


def load_frozen_inputs(
    *,
    load_candidate_intervals: Callable[
        [str], tuple[pd.DataFrame, Path, dict[str, Any]]
    ],
    load_eligibility_base: Callable[
        [str], tuple[pd.DataFrame, Path, dict[str, Any]]
    ],
) -> tuple[dict[str, dict[str, pd.DataFrame]], dict[str, Any]]:
    """读取三个正式 build input，并返回可比较的冻结快照。"""
    frames: dict[str, dict[str, pd.DataFrame]] = {}
    snapshot: dict[str, Any] = {
        "design_register": {
            "path": _relative(DESIGN_REGISTER_PATH),
            "sha256": sha256_file(DESIGN_REGISTER_PATH),
        },
        "cohorts": {},
    }
    for cohort in COHORTS:
        candidate, candidate_path, candidate_manifest = load_candidate_intervals(cohort)
        eligibility, eligibility_path, eligibility_manifest = load_eligibility_base(cohort)
        pd.testing.assert_frame_equal(
            candidate.loc[:, list(KEY_COLUMNS)],
            eligibility.loc[:, list(KEY_COLUMNS)],
            check_dtype=True,
            check_exact=True,
        )
        candidate_snapshot = _verify_interval_freeze(
            panel_name="candidate_intervals_v3",
            freeze_path=CANDIDATE_FREEZE_PATH,
            cohort=cohort,
            data=candidate,
            path=candidate_path,
            manifest=candidate_manifest,
        )
        eligibility_snapshot = _verify_interval_freeze(
            panel_name="eligibility_base_v2",
            freeze_path=ELIGIBILITY_FREEZE_PATH,
            cohort=cohort,
            data=eligibility,
            path=eligibility_path,
            manifest=eligibility_manifest,
        )
        wealth, wealth_snapshot = _load_wealth_panel(cohort)
        frames[cohort] = {
            "candidate": candidate,
            "eligibility": eligibility,
            "wealth": wealth,
        }
        snapshot["cohorts"][cohort] = {
            "candidate_intervals_v3": candidate_snapshot,
            "eligibility_base_v2": eligibility_snapshot,
            "wealth_panel_v1": wealth_snapshot,
        }
    return frames, snapshot


def _attach_wealth_role(
    ledger: pd.DataFrame,
    wealth: pd.DataFrame,
    *,
    wave_column: str,
    suffix: str,
) -> pd.DataFrame:
    """按指定时间角色连接财富、除数和来源 flag。"""
    lookup = wealth.loc[
        :, ["cohort", "person_id", "wave", "wealth_total_real_equiv", "wealth_equiv_divisor", "wealth_source_flag"]
    ].rename(
        columns={
            "wave": wave_column,
            "wealth_total_real_equiv": f"wealth_{suffix}",
            "wealth_equiv_divisor": f"divisor_{suffix}",
            "wealth_source_flag": f"flag_{suffix}",
        }
    )
    return ledger.merge(
        lookup,
        on=["cohort", "person_id", wave_column],
        how="left",
        validate="many_to_one",
        sort=False,
    )


def build_shock_pass(
    candidate: pd.DataFrame,
    eligibility: pd.DataFrame,
    wealth: pd.DataFrame,
    cohort: str,
) -> pd.DataFrame:
    """Pass A：只读 t-1/t 财富，计算 shock 并锁定首次 qualifying shock。"""
    required_candidate = {
        *KEY_COLUMNS,
        "candidate_interval_order",
        "tminus1_wave",
        "tplus1_wave",
        "tminus1_nominal_year",
        "shock_wave_nominal_year",
        "tplus1_nominal_year",
    }
    missing = required_candidate.difference(candidate.columns)
    if missing:
        raise KeyError(f"candidate intervals 缺少字段：{sorted(missing)}")
    if "structural_eligibility_status" not in eligibility.columns:
        raise KeyError("eligibility base 缺少 structural_eligibility_status")
    if set(candidate["cohort"].dropna().astype(str)) != {cohort}:
        raise RuntimeError(f"{cohort}: Candidate cohort 不一致")
    pd.testing.assert_frame_equal(
        candidate.loc[:, list(KEY_COLUMNS)],
        eligibility.loc[:, list(KEY_COLUMNS)],
        check_dtype=True,
        check_exact=True,
    )

    columns = [
        "cohort",
        "person_id",
        "candidate_interval_order",
        "candidate_shock_wave",
        "tminus1_wave",
        "tplus1_wave",
        "tminus1_nominal_year",
        "shock_wave_nominal_year",
        "tplus1_nominal_year",
    ]
    ledger = candidate.loc[:, columns].copy()
    ledger["structural_eligibility_status"] = eligibility[
        "structural_eligibility_status"
    ].reset_index(drop=True)
    ledger = _attach_wealth_role(
        ledger, wealth, wave_column="tminus1_wave", suffix="tminus1"
    )
    ledger = _attach_wealth_role(
        ledger, wealth, wave_column="candidate_shock_wave", suffix="t"
    )

    structural_pass = ledger["structural_eligibility_status"].eq("pass")
    evaluable = (
        structural_pass
        & ledger["wealth_tminus1"].notna()
        & ledger["wealth_t"].notna()
        & ledger["wealth_tminus1"].gt(0)
    )
    ledger["wealth_loss_fraction"] = np.nan
    ledger.loc[evaluable, "wealth_loss_fraction"] = (
        ledger.loc[evaluable, "wealth_tminus1"]
        - ledger.loc[evaluable, "wealth_t"]
    ) / ledger.loc[evaluable, "wealth_tminus1"]
    ledger["wealth_loss_fraction"] = ledger["wealth_loss_fraction"].astype("float64")

    major = pd.Series(pd.NA, index=ledger.index, dtype="boolean")
    major.loc[evaluable] = ledger.loc[evaluable, "wealth_loss_fraction"].ge(
        SHOCK_THRESHOLD
    )
    ledger["major_shock"] = major

    ledger["is_index_shock"] = pd.Series(False, index=ledger.index, dtype="boolean")
    first_wave, last_wave = SHOCK_SEARCH_WINDOWS[cohort]
    within_window = ledger["candidate_shock_wave"].between(first_wave, last_wave)
    qualifying = ledger.loc[
        ledger["major_shock"].eq(True) & within_window
    ].sort_values(
        ["cohort", "person_id", "candidate_interval_order"], kind="stable"
    )
    first_indices = qualifying.drop_duplicates(
        ["cohort", "person_id"], keep="first"
    ).index
    ledger.loc[first_indices, "is_index_shock"] = True
    return ledger


def attach_recovery_pass(shock_pass: pd.DataFrame, wealth: pd.DataFrame) -> pd.DataFrame:
    """Pass B：首次 shock 已锁定后，仅给 index 行读取 t+1 财富。"""
    if "wealth_tplus1" in shock_pass.columns:
        raise RuntimeError("Pass A 不得预先包含 t+1 财富")
    result = shock_pass.copy()
    index_keys_before = result.loc[
        result["is_index_shock"].eq(True), list(KEY_COLUMNS)
    ].reset_index(drop=True)

    result["wealth_tplus1"] = np.nan
    result["divisor_tplus1"] = pd.Series(pd.NA, index=result.index, dtype="Float64")
    result["flag_tplus1"] = pd.Series(pd.NA, index=result.index, dtype="string")
    requested = result["is_index_shock"].eq(True) & result["tplus1_wave"].notna()
    request = result.loc[
        requested, ["cohort", "person_id", "tplus1_wave"]
    ].copy()
    request["_row_id"] = request.index
    lookup = wealth.loc[
        :, ["cohort", "person_id", "wave", "wealth_total_real_equiv", "wealth_equiv_divisor", "wealth_source_flag"]
    ].rename(
        columns={
            "wave": "tplus1_wave",
            "wealth_total_real_equiv": "_wealth_tplus1",
            "wealth_equiv_divisor": "_divisor_tplus1",
            "wealth_source_flag": "_flag_tplus1",
        }
    )
    joined = request.merge(
        lookup,
        on=["cohort", "person_id", "tplus1_wave"],
        how="left",
        validate="many_to_one",
        sort=False,
    ).set_index("_row_id")
    if len(joined):
        row_ids = joined.index
        result.loc[row_ids, "wealth_tplus1"] = joined["_wealth_tplus1"].to_numpy()
        result.loc[row_ids, "divisor_tplus1"] = joined["_divisor_tplus1"].to_numpy()
        result.loc[row_ids, "flag_tplus1"] = joined["_flag_tplus1"].to_numpy()

    result["wealth_tplus1"] = result["wealth_tplus1"].astype("float64")
    result["recovery_fraction_raw"] = np.nan
    classifiable = result["is_index_shock"].eq(True) & result[
        "wealth_tplus1"
    ].notna()
    denominator = result["wealth_tminus1"] - result["wealth_t"]
    if (denominator.loc[classifiable] <= 0).any():
        raise RuntimeError("index shock 的 recovery denominator 必须大于零")
    result.loc[classifiable, "recovery_fraction_raw"] = (
        result.loc[classifiable, "wealth_tplus1"]
        - result.loc[classifiable, "wealth_t"]
    ) / denominator.loc[classifiable]
    result["recovery_fraction_raw"] = result["recovery_fraction_raw"].astype(
        "float64"
    )
    result["recovery_status"] = pd.Series(pd.NA, index=result.index, dtype="string")
    recovered = classifiable & result["recovery_fraction_raw"].ge(
        PRIMARY_RECOVERY_THRESHOLD
    )
    limited = classifiable & ~recovered
    result.loc[recovered, "recovery_status"] = "substantial_recovery"
    result.loc[limited, "recovery_status"] = "persistent_or_limited_recovery"

    index_keys_after = result.loc[
        result["is_index_shock"].eq(True), list(KEY_COLUMNS)
    ].reset_index(drop=True)
    pd.testing.assert_frame_equal(
        index_keys_before, index_keys_after, check_dtype=True, check_exact=True
    )
    return result


def build_exposure_panel(
    candidate: pd.DataFrame,
    eligibility: pd.DataFrame,
    wealth: pd.DataFrame,
    cohort: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """构建正式13列 Exposure Panel，并返回含 QC lineage 的内部表。"""
    shock_pass = build_shock_pass(candidate, eligibility, wealth, cohort)
    internal = attach_recovery_pass(shock_pass, wealth)
    panel = internal.loc[:, list(EXPOSURE_PANEL_COLUMNS)].copy()
    for column, dtype in EXPOSURE_PANEL_DTYPES.items():
        panel[column] = panel[column].astype(dtype)
    validate_exposure_panel(panel, candidate, eligibility, cohort)
    return panel, internal


def derive_recovery_indicator(
    panel: pd.DataFrame, threshold: float
) -> pd.Series:
    """由 raw recovery fraction 动态生成允许的主或敏感性阈值 indicator。"""
    allowed = {PRIMARY_RECOVERY_THRESHOLD, *SENSITIVITY_THRESHOLDS}
    if float(threshold) not in allowed:
        raise ValueError(f"正式 recovery threshold 仅允许 {sorted(allowed)}")
    result = pd.Series(pd.NA, index=panel.index, dtype="boolean")
    classifiable = panel["is_index_shock"].eq(True) & panel[
        "recovery_fraction_raw"
    ].notna()
    result.loc[classifiable] = panel.loc[classifiable, "recovery_fraction_raw"].ge(
        float(threshold)
    )
    return result


def select_index_exposure(panel: pd.DataFrame) -> pd.DataFrame:
    """返回一人一行的 index exposure view，并保留 recovery unavailable。"""
    missing = set(EXPOSURE_PANEL_COLUMNS).difference(panel.columns)
    if missing:
        raise KeyError(f"Exposure Panel 缺少字段：{sorted(missing)}")
    result = panel.loc[panel["is_index_shock"].eq(True)].copy()
    if result.duplicated(["cohort", "person_id"]).any():
        raise RuntimeError("index exposure view 不是一人一行")
    return result.reset_index(drop=True)


def validate_exposure_panel(
    panel: pd.DataFrame,
    candidate: pd.DataFrame,
    eligibility: pd.DataFrame,
    cohort: str,
) -> None:
    """验证13列契约、首次冲击、NA 与恢复分类边界。"""
    if list(panel.columns) != list(EXPOSURE_PANEL_COLUMNS):
        raise RuntimeError(f"{cohort}: Exposure Panel 列结构不一致")
    actual_dtypes = {column: str(panel[column].dtype) for column in panel.columns}
    if actual_dtypes != EXPOSURE_PANEL_DTYPES:
        raise RuntimeError(f"{cohort}: Exposure Panel dtype 不一致：{actual_dtypes}")
    if len(panel) != len(candidate):
        raise RuntimeError(f"{cohort}: Exposure Panel 行数未与 Candidate 守恒")
    pd.testing.assert_frame_equal(
        panel.loc[:, list(KEY_COLUMNS)],
        candidate.loc[:, list(KEY_COLUMNS)],
        check_dtype=True,
        check_exact=True,
    )
    pd.testing.assert_series_equal(
        panel["tminus1_wave"],
        candidate["tminus1_wave"],
        check_names=False,
        check_dtype=True,
        check_exact=True,
    )
    pd.testing.assert_series_equal(
        panel["tplus1_wave"],
        candidate["tplus1_wave"],
        check_names=False,
        check_dtype=True,
        check_exact=True,
    )
    if panel.duplicated(list(KEY_COLUMNS)).any():
        raise RuntimeError(f"{cohort}: Exposure Panel 主键不唯一")
    forbidden_nulls = {
        column: int(panel[column].isna().sum())
        for column, nullable in EXPOSURE_PANEL_NULLABLE.items()
        if not nullable and panel[column].isna().any()
    }
    if forbidden_nulls:
        raise RuntimeError(f"{cohort}: Exposure Panel 非空字段出现缺失：{forbidden_nulls}")
    statuses = set(panel["recovery_status"].dropna().astype(str))
    if not statuses.issubset(RECOVERY_STATUS_VALUES):
        raise RuntimeError(f"{cohort}: recovery_status 出现非法值：{statuses}")

    structural = eligibility["structural_eligibility_status"].eq("pass")
    evaluable = (
        structural
        & panel["wealth_tminus1"].notna()
        & panel["wealth_t"].notna()
        & panel["wealth_tminus1"].gt(0)
    )
    if not panel.loc[~evaluable, "wealth_loss_fraction"].isna().all():
        raise RuntimeError(f"{cohort}: 不可评估 interval 出现 wealth loss fraction")
    if not panel.loc[~evaluable, "major_shock"].isna().all():
        raise RuntimeError(f"{cohort}: 不可评估 interval 出现 major_shock 分类")
    expected_loss = (
        panel.loc[evaluable, "wealth_tminus1"] - panel.loc[evaluable, "wealth_t"]
    ) / panel.loc[evaluable, "wealth_tminus1"]
    if not np.allclose(
        panel.loc[evaluable, "wealth_loss_fraction"],
        expected_loss,
        rtol=1e-12,
        atol=1e-12,
    ):
        raise RuntimeError(f"{cohort}: wealth_loss_fraction 公式不一致")
    expected_major = expected_loss.ge(SHOCK_THRESHOLD).astype("boolean")
    pd.testing.assert_series_equal(
        panel.loc[evaluable, "major_shock"].reset_index(drop=True),
        expected_major.reset_index(drop=True),
        check_names=False,
        check_dtype=True,
        check_exact=True,
    )
    index_mask = panel["is_index_shock"].eq(True)
    if not panel.loc[index_mask, "major_shock"].eq(True).all():
        raise RuntimeError(f"{cohort}: index shock 不是 major shock")
    if panel.loc[index_mask].duplicated("person_id").any():
        raise RuntimeError(f"{cohort}: 每人存在多个 index shock")
    index_counts = panel.groupby("person_id", sort=False)["is_index_shock"].sum()
    if int(index_counts.max()) > 1:
        raise RuntimeError(f"{cohort}: 每人 index shock 数超过1")
    if int(index_mask.sum()) != int(panel.loc[index_mask, "person_id"].nunique()):
        raise RuntimeError(f"{cohort}: index shock 行数与 analytic index person 数不等")
    structural_pass = eligibility["structural_eligibility_status"].eq("pass")
    if not structural_pass.loc[index_mask].all():
        raise RuntimeError(f"{cohort}: index shock 未通过 structural eligibility")
    first_wave, last_wave = SHOCK_SEARCH_WINDOWS[cohort]
    within_window = panel["candidate_shock_wave"].between(first_wave, last_wave)
    if not within_window.loc[index_mask].all():
        raise RuntimeError(f"{cohort}: index shock 超出锁定 analytic window")

    expected_index = pd.Series(False, index=panel.index, dtype="boolean")
    order = candidate.loc[:, ["person_id", "candidate_interval_order"]].copy()
    qualifying = order.loc[
        panel["major_shock"].eq(True) & within_window
    ].sort_values(
        ["person_id", "candidate_interval_order"], kind="stable"
    )
    expected_index.loc[qualifying.drop_duplicates("person_id", keep="first").index] = True
    pd.testing.assert_series_equal(
        panel["is_index_shock"],
        expected_index,
        check_names=False,
        check_dtype=True,
        check_exact=True,
    )

    non_index = ~panel["is_index_shock"].eq(True)
    if not panel.loc[
        non_index,
        ["wealth_tplus1", "recovery_fraction_raw", "recovery_status"],
    ].isna().all().all():
        raise RuntimeError(f"{cohort}: 非 index interval 读取了 t+1 或 recovery")
    classifiable = panel["is_index_shock"].eq(True) & panel[
        "wealth_tplus1"
    ].notna()
    unavailable = panel["is_index_shock"].eq(True) & panel["wealth_tplus1"].isna()
    if not panel.loc[
        unavailable, ["recovery_fraction_raw", "recovery_status"]
    ].isna().all().all():
        raise RuntimeError(f"{cohort}: t+1 unavailable 被错误分类")
    denominator = panel["wealth_tminus1"] - panel["wealth_t"]
    expected_recovery = (
        panel.loc[classifiable, "wealth_tplus1"] - panel.loc[classifiable, "wealth_t"]
    ) / denominator.loc[classifiable]
    if not np.allclose(
        panel.loc[classifiable, "recovery_fraction_raw"],
        expected_recovery,
        rtol=1e-12,
        atol=1e-12,
    ):
        raise RuntimeError(f"{cohort}: recovery_fraction_raw 公式不一致")
    if not np.isfinite(
        panel.loc[classifiable, "recovery_fraction_raw"].to_numpy(dtype=float)
    ).all():
        raise RuntimeError(f"{cohort}: recovery_fraction_raw 含无穷值")
    expected_status = pd.Series(pd.NA, index=panel.index, dtype="string")
    expected_status.loc[classifiable & panel["recovery_fraction_raw"].ge(0.50)] = (
        "substantial_recovery"
    )
    expected_status.loc[classifiable & panel["recovery_fraction_raw"].lt(0.50)] = (
        "persistent_or_limited_recovery"
    )
    pd.testing.assert_series_equal(
        panel["recovery_status"],
        expected_status,
        check_names=False,
        check_dtype=True,
        check_exact=True,
    )


def _rate_text(numerator: int, denominator: int) -> str:
    if denominator == 0:
        return f"{numerator:,} (NA)"
    return f"{numerator:,} ({100 * numerator / denominator:.1f}%)"


def _finite(value: Any) -> float | None:
    if value is None or pd.isna(value) or not np.isfinite(float(value)):
        return None
    return float(value)


def _timing_summary(values: pd.Series) -> dict[str, float | None]:
    clean = pd.to_numeric(values, errors="coerce").dropna().astype(float)
    if clean.empty:
        return {key: None for key in ("median", "q1", "q3", "minimum", "maximum")}
    return {
        "median": _finite(clean.median()),
        "q1": _finite(clean.quantile(0.25)),
        "q3": _finite(clean.quantile(0.75)),
        "minimum": _finite(clean.min()),
        "maximum": _finite(clean.max()),
    }


def build_qc(
    cohort: str, panel: pd.DataFrame, internal: pd.DataFrame
) -> dict[str, Any]:
    """生成不含 person ID 的 shock、recovery、极端值与 divisor QC。"""
    shock_visible: list[dict[str, Any]] = []
    recovery_visible: list[dict[str, Any]] = []
    shock_machine: list[dict[str, Any]] = []
    recovery_machine: list[dict[str, Any]] = []
    first_wave, last_wave = SHOCK_SEARCH_WINDOWS[cohort]

    for wave, group in internal.groupby("candidate_shock_wave", sort=True):
        structural = group["structural_eligibility_status"].eq("pass")
        valid_pair = structural & group["wealth_tminus1"].notna() & group[
            "wealth_t"
        ].notna()
        positive = valid_pair & group["wealth_tminus1"].gt(0)
        major = group["major_shock"].eq(True)
        index = group["is_index_shock"].eq(True)
        record = {
            "candidate_shock_wave": int(wave),
            "within_analytic_shock_search_window": bool(
                first_wave <= int(wave) <= last_wave
            ),
            "structural_pass_n": int(structural.sum()),
            "valid_wealth_pair_n": int(valid_pair.sum()),
            "positive_tminus1_n": int(positive.sum()),
            "major_shock_n": int(major.sum()),
            "first_shock_n": int(index.sum()),
        }
        shock_machine.append(record)
        if record["within_analytic_shock_search_window"]:
            shock_visible.append(
                {
                    "Cohort": cohort,
                    "Shock wave": int(wave),
                    "Structural pass N": record["structural_pass_n"],
                    "Valid wealth pair N": record["valid_wealth_pair_n"],
                    "Positive t-1 wealth N": record["positive_tminus1_n"],
                    "Major shocks N": record["major_shock_n"],
                    "First shocks N": record["first_shock_n"],
                }
            )

        index_group = group.loc[index].copy()
        classifiable = index_group["recovery_fraction_raw"].notna()
        classified = index_group.loc[classifiable].copy()
        if classified[
            ["divisor_tminus1", "divisor_t", "divisor_tplus1"]
        ].isna().any(axis=None):
            raise RuntimeError(f"{cohort} W{int(wave)}: classifiable recovery 缺少 divisor")
        divisor_changed = (
            classified["divisor_tminus1"].ne(classified["divisor_t"])
            | classified["divisor_t"].ne(classified["divisor_tplus1"])
        )
        threshold_counts = {
            f"recovery_ge_{threshold:.2f}_n": int(
                classified["recovery_fraction_raw"].ge(threshold).sum()
            )
            for threshold in (0.25, 0.50, 0.75, 1.00)
        }
        stable = classified.loc[~divisor_changed]
        changed = classified.loc[divisor_changed]
        record_recovery = {
            "candidate_shock_wave": int(wave),
            "index_shock_n": len(index_group),
            "tplus1_wealth_available_n": int(classifiable.sum()),
            "recovery_classified_n": int(classifiable.sum()),
            "median_recovery_fraction_raw": _finite(
                classified["recovery_fraction_raw"].median()
            ),
            "q1_recovery_fraction_raw": _finite(
                classified["recovery_fraction_raw"].quantile(0.25)
            ),
            "q3_recovery_fraction_raw": _finite(
                classified["recovery_fraction_raw"].quantile(0.75)
            ),
            **threshold_counts,
            "divisor_changed_any_n": int(divisor_changed.sum()),
            "primary_recovery_stable_divisor_n": int(
                stable["recovery_fraction_raw"].ge(0.50).sum()
            ),
            "primary_comparator_stable_divisor_n": int(
                stable["recovery_fraction_raw"].lt(0.50).sum()
            ),
            "stable_divisor_n": len(stable),
            "primary_recovery_changed_divisor_n": int(
                changed["recovery_fraction_raw"].ge(0.50).sum()
            ),
            "primary_comparator_changed_divisor_n": int(
                changed["recovery_fraction_raw"].lt(0.50).sum()
            ),
            "changed_divisor_n": len(changed),
        }
        recovery_machine.append(record_recovery)
        no_next = bool(index_group["tplus1_wave"].isna().all()) and len(index_group) > 0
        if first_wave <= int(wave) <= last_wave:
            median = record_recovery["median_recovery_fraction_raw"]
            q1 = record_recovery["q1_recovery_fraction_raw"]
            q3 = record_recovery["q3_recovery_fraction_raw"]
            recovery_visible.append(
                {
                    "Cohort": cohort,
                    "Shock wave": int(wave),
                    "Index shocks N": len(index_group),
                    "t+1 wealth available, n (%)": (
                        "Not available by design"
                        if no_next
                        else _rate_text(int(classifiable.sum()), len(index_group))
                    ),
                    "Median R (IQR)": (
                        "NA"
                        if median is None
                        else f"{median:.3f} [{q1:.3f}, {q3:.3f}]"
                    ),
                    "R >= 0.25, n (%)": _rate_text(
                        threshold_counts["recovery_ge_0.25_n"], len(classified)
                    ),
                    "Primary R >= 0.50, n (%)": _rate_text(
                        threshold_counts["recovery_ge_0.50_n"], len(classified)
                    ),
                    "Audit R >= 0.75, n (%)": _rate_text(
                        threshold_counts["recovery_ge_0.75_n"], len(classified)
                    ),
                    "Full R >= 1.00, n (%)": _rate_text(
                        threshold_counts["recovery_ge_1.00_n"], len(classified)
                    ),
                }
            )

    index = internal.loc[internal["is_index_shock"].eq(True)].copy()
    classified = index.loc[index["recovery_fraction_raw"].notna()].copy()
    primary = classified["recovery_fraction_raw"].ge(0.50)
    counts = {
        "row_count": len(panel),
        "index_shocks": len(index),
        "classifiable": len(classified),
        "recovery_unavailable": len(index) - len(classified),
        "primary_recovery": int(primary.sum()),
        "primary_comparator": int((~primary).sum()),
        "no_cohort_global_tplus1": int(index["tplus1_wave"].isna().sum()),
        "tplus1_exists_but_wealth_missing": int(
            (index["tplus1_wave"].notna() & index["wealth_tplus1"].isna()).sum()
        ),
    }
    if counts["index_shocks"] == 0 or counts["classifiable"] == 0:
        raise RuntimeError(f"{cohort}: 没有可用于 exposure audit 的 index shock")

    threshold_support = []
    for threshold in (0.25, 0.50, 0.75, 1.00):
        recovered_n = int(classified["recovery_fraction_raw"].ge(threshold).sum())
        comparator_n = len(classified) - recovered_n
        record = {
            "threshold": threshold,
            "role": (
                "primary"
                if threshold == 0.50
                else "sensitivity"
                if threshold in SENSITIVITY_THRESHOLDS
                else "audit_only"
            ),
            "recovery_n": recovered_n,
            "recovery_rate": recovered_n / len(classified),
            "comparator_n": comparator_n,
            "comparator_rate": comparator_n / len(classified),
            "support_flag": (
                recovered_n < 100
                or comparator_n < 100
                or recovered_n / len(classified) < 0.10
                or comparator_n / len(classified) < 0.10
            ),
        }
        threshold_support.append(record)
        counts[f"recovery_ge_{threshold:.2f}_n"] = recovered_n
    if any(record["support_flag"] for record in threshold_support):
        raise RuntimeError(f"{cohort}: 至少一个预设 threshold 出现 support/positivity flag")
    nested_counts = [
        counts["recovery_ge_1.00_n"],
        counts["recovery_ge_0.75_n"],
        counts["recovery_ge_0.50_n"],
        counts["recovery_ge_0.25_n"],
        counts["classifiable"],
    ]
    if nested_counts != sorted(nested_counts):
        raise RuntimeError(f"{cohort}: recovery threshold 人数未满足嵌套关系")
    if counts["classifiable"] != (
        counts["primary_recovery"] + counts["primary_comparator"]
    ):
        raise RuntimeError(f"{cohort}: classifiable 与主分类人数不守恒")

    recovery = classified["recovery_fraction_raw"].astype(float)
    distribution = {
        "mean": _finite(recovery.mean()),
        "sd": _finite(recovery.std()),
        "minimum": _finite(recovery.min()),
        "p05": _finite(recovery.quantile(0.05)),
        "p10": _finite(recovery.quantile(0.10)),
        "p25": _finite(recovery.quantile(0.25)),
        "median": _finite(recovery.median()),
        "p75": _finite(recovery.quantile(0.75)),
        "p90": _finite(recovery.quantile(0.90)),
        "p95": _finite(recovery.quantile(0.95)),
        "maximum": _finite(recovery.max()),
    }
    categories = {
        "R_lt_0": int(recovery.lt(0).sum()),
        "R_0_to_lt_0_25": int(recovery.ge(0).mul(recovery.lt(0.25)).sum()),
        "R_0_25_to_lt_0_50": int(recovery.ge(0.25).mul(recovery.lt(0.50)).sum()),
        "R_0_50_to_lt_0_75": int(recovery.ge(0.50).mul(recovery.lt(0.75)).sum()),
        "R_0_75_to_lt_1": int(recovery.ge(0.75).mul(recovery.lt(1.00)).sum()),
        "R_ge_1": int(recovery.ge(1.00).sum()),
    }
    if sum(categories.values()) != len(classified):
        raise RuntimeError(f"{cohort}: recovery category 计数不守恒")

    extreme_masks = {
        "R_lt_minus1": recovery.lt(-1),
        "R_gt_2": recovery.gt(2),
        "R_gt_5": recovery.gt(5),
    }
    extremes = {
        name: {
            "n": int(mask.sum()),
            "median_wealth_tminus1": _finite(
                classified.loc[mask, "wealth_tminus1"].median()
            ),
            "source_flag_tplus1_available_n": int(
                classified.loc[mask, "flag_tplus1"].notna().sum()
            ),
        }
        for name, mask in extreme_masks.items()
    }

    divisor_pre = (
        index["divisor_tminus1"].notna()
        & index["divisor_t"].notna()
        & index["divisor_tminus1"].ne(index["divisor_t"])
    )
    divisor_recovery = (
        classified["divisor_t"].notna()
        & classified["divisor_tplus1"].notna()
        & classified["divisor_t"].ne(classified["divisor_tplus1"])
    )
    classified_pre_change = divisor_pre.loc[classified.index]
    divisor_any = classified_pre_change | divisor_recovery
    if classified[
        ["divisor_tminus1", "divisor_t", "divisor_tplus1"]
    ].isna().any(axis=None):
        raise RuntimeError(f"{cohort}: classifiable recovery 缺少 divisor")
    divisor_groups: dict[str, Any] = {}
    for label, mask in (("stable", ~divisor_any), ("changed_any", divisor_any)):
        group = classified.loc[mask]
        recovered_n = int(group["recovery_fraction_raw"].ge(0.50).sum())
        divisor_groups[label] = {
            "n": len(group),
            "primary_recovery_n": recovered_n,
            "primary_comparator_n": len(group) - recovered_n,
            "primary_recovery_rate": None if len(group) == 0 else recovered_n / len(group),
        }
    divisor_qc = {
        "definition": "any divisor change across t-1, t, or t+1 among classifiable index shocks",
        "classifiable_divisor_changed_any_n": int(divisor_any.sum()),
        "classifiable_divisor_changed_any_rate": int(divisor_any.sum()) / len(classified),
        "primary_cross_tab": divisor_groups,
        "recovery_rate_difference_changed_minus_stable": (
            None
            if divisor_groups["changed_any"]["primary_recovery_rate"] is None
            else divisor_groups["changed_any"]["primary_recovery_rate"]
            - divisor_groups["stable"]["primary_recovery_rate"]
        ),
        "inferential_sensitivity_automatically_triggered": False,
        "action": (
            "Record exposure-only QC. Stable-divisor inferential restriction requires "
            "separate scientific review and approval."
        ),
    }
    stable_rate = divisor_groups["stable"]["primary_recovery_rate"]
    changed_rate = divisor_groups["changed_any"]["primary_recovery_rate"]
    divisor_visible = {
        "Cohort": cohort,
        "Classifiable N": len(classified),
        "Divisor changed N": int(divisor_any.sum()),
        "Divisor changed %": int(divisor_any.sum()) / len(classified),
        "Recovery % stable": stable_rate,
        "Recovery % changed": changed_rate,
        "QC decision": "No automatic restricted model",
    }

    timing = {
        "tminus1_to_t_nominal_years": _timing_summary(
            index["shock_wave_nominal_year"] - index["tminus1_nominal_year"]
        ),
        "t_to_tplus1_nominal_years": _timing_summary(
            index["tplus1_nominal_year"] - index["shock_wave_nominal_year"]
        ),
    }
    return {
        "counts": counts,
        "threshold_support": threshold_support,
        "distribution": distribution,
        "categories": categories,
        "extremes": extremes,
        "divisor_qc": divisor_qc,
        "timing": timing,
        "shock_flow_by_wave": shock_machine,
        "recovery_by_wave": recovery_machine,
        "shock_visible": shock_visible,
        "recovery_visible": recovery_visible,
        "divisor_visible": divisor_visible,
    }


DEFINITION_ROWS = [
    {
        "Measure": "Wealth loss fraction",
        "Formula or rule": "(W[t-1] - W[t]) / W[t-1]",
        "Timing": "t-1 to t",
        "Role": "Continuous shock component",
        "Boundary rule": "Calculate only after structural pass, valid wealth pair, and W[t-1] > 0; do not cap",
        "Source": "Approved project exposure contract (2026-09-13)",
    },
    {
        "Measure": "Major wealth shock",
        "Formula or rule": "Loss fraction >= 0.75",
        "Timing": "t",
        "Role": "Primary shock definition",
        "Boundary rule": "major_shock is evaluated for every evaluable interval; index selection is restricted to the cohort-specific analytic window",
        "Source": "https://pubmed.ncbi.nlm.nih.gov/37544315/",
    },
    {
        "Measure": "Analytic index shock",
        "Formula or rule": "First major shock within HRS W3-W14, ELSA W2-W8, or KLoSA W2-W7",
        "Timing": "t",
        "Role": "Index event",
        "Boundary rule": "Select by candidate_interval_order before reading t+1 wealth, FI, follow-up, or health outcomes",
        "Source": "Approved analytic shock-search window contract (2026-09-13)",
    },
    {
        "Measure": "Recovery fraction",
        "Formula or rule": "(W[t+1] - W[t]) / (W[t-1] - W[t])",
        "Timing": "t+1",
        "Role": "Raw continuous component",
        "Boundary rule": "Preserve negative and above-one values; no clipping or winsorisation",
        "Source": "Approved project exposure contract (2026-09-13)",
    },
    {
        "Measure": "Substantial recovery",
        "Formula or rule": "R >= 0.50 -> substantial_recovery",
        "Timing": "t+1",
        "Role": "Primary exposure",
        "Boundary rule": "Represents recovery of at least half of the shock-related loss",
        "Source": "Approved project exposure contract (2026-09-13)",
    },
    {
        "Measure": "Primary comparator",
        "Formula or rule": "R < 0.50 -> persistent_or_limited_recovery",
        "Timing": "t+1",
        "Role": "Primary comparator",
        "Boundary rule": "Includes continued loss, no recovery, and recovery of less than half of the shock-related loss",
        "Source": "Approved project exposure contract (2026-09-13)",
    },
    {
        "Measure": "Partial recovery sensitivity",
        "Formula or rule": "Recovery fraction >= 0.25",
        "Timing": "t+1",
        "Role": "Threshold sensitivity",
        "Boundary rule": "Generated dynamically; not stored as a physical panel column",
        "Source": "Approved project exposure contract (2026-09-13)",
    },
    {
        "Measure": "Full recovery sensitivity",
        "Formula or rule": "Recovery fraction >= 1.00",
        "Timing": "t+1",
        "Role": "Threshold sensitivity",
        "Boundary rule": "Generated dynamically; not stored as a physical panel column",
        "Source": "Approved project exposure contract (2026-09-13)",
    },
    {
        "Measure": "75% recovery threshold",
        "Formula or rule": "Recovery fraction >= 0.75",
        "Timing": "t+1",
        "Role": "Exposure audit only",
        "Boundary rule": "Do not add to Methods, S6, or inferential models",
        "Source": "Post-Shock Wealth Recovery Exposure Audit",
    },
    {
        "Measure": "Unavailable recovery",
        "Formula or rule": "Missing valid W[t+1]",
        "Timing": "t+1",
        "Role": "Unclassified",
        "Boundary rule": "Recovery fraction and status remain NA; never substitute a later shock",
        "Source": "Approved project exposure contract (2026-09-13)",
    },
]


WORKBOOK_BUILDER = r'''
import fs from "node:fs/promises";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const [payloadPath, outputPath, previewDir] = process.argv.slice(2);
const payload = JSON.parse(await fs.readFile(payloadPath, "utf8"));
const workbook = Workbook.create();
const fontFamily = "Arial";

function addReaderSheet({name, title, context, headers, rows, tableName, widths, numericFormats, bodyHeight, contextHeight}) {
  const sheet = workbook.worksheets.add(name);
  sheet.showGridLines = false;
  sheet.getRange("A1").values = [[title]];
  sheet.getRange("A1").format.font = { name: fontFamily, size: 14, bold: true, color: "#1F2937" };
  const contextRange = sheet.getRangeByIndexes(1, 0, 1, headers.length);
  contextRange.merge();
  contextRange.values = [[context]];
  contextRange.format.font = { name: fontFamily, size: 10, italic: true, color: "#4B5563" };
  contextRange.format.wrapText = true;
  contextRange.format.rowHeightPx = contextHeight ?? 42;
  sheet.getRangeByIndexes(2, 0, 1, headers.length).format.borders = {
    bottom: { style: "thin", color: "#94A3B8" },
  };
  const matrix = [headers, ...rows.map((row) => headers.map((header) => row[header] ?? null))];
  const range = sheet.getRangeByIndexes(3, 0, matrix.length, headers.length);
  range.values = matrix;
  range.format.font = { name: fontFamily, size: 10, color: "#1F2937" };
  range.format.verticalAlignment = "top";
  range.format.wrapText = true;
  const header = sheet.getRangeByIndexes(3, 0, 1, headers.length);
  header.format = {
    fill: "#1F4E78",
    font: { name: fontFamily, size: 10, bold: true, color: "#FFFFFF" },
    horizontalAlignment: "center",
    verticalAlignment: "center",
    wrapText: true,
  };
  header.format.rowHeightPx = 46;
  for (let index = 0; index < widths.length; index += 1) {
    sheet.getRangeByIndexes(0, index, matrix.length + 3, 1).format.columnWidthPx = widths[index];
  }
  if (rows.length > 0) {
    sheet.getRangeByIndexes(4, 0, rows.length, headers.length).format.rowHeightPx = bodyHeight ?? 36;
  }
  for (const [headerName, formatCode] of Object.entries(numericFormats ?? {})) {
    const index = headers.indexOf(headerName);
    if (index >= 0 && rows.length > 0) {
      sheet.getRangeByIndexes(4, index, rows.length, 1).format.numberFormat = formatCode;
    }
  }
  const table = sheet.tables.add(range, true, tableName);
  table.style = "TableStyleMedium2";
  table.showFilterButton = true;
  table.showBandedRows = true;
  sheet.freezePanes.freezeRows(4);
  return sheet;
}

addReaderSheet({
  name: "1_Definition Contract",
  title: "Post-shock wealth recovery definition",
  context: "The 50% primary threshold is concept-driven: substantial recovery means regaining at least half of the shock-related wealth loss. Exposure distributions confirm support only; no frailty outcomes were used.",
  headers: payload.definition.headers,
  rows: payload.definition.rows,
  tableName: "ExposureDefinitionContract",
  widths: [155, 230, 90, 165, 300, 250],
  numericFormats: {},
  bodyHeight: 58,
  contextHeight: 46,
});

addReaderSheet({
  name: "2_Shock Flow",
  title: "First-shock identification within the prespecified analytic shock window",
  context: "Only the locked windows are shown: HRS W3-W14, ELSA W2-W8, and KLoSA W2-W7. First shocks are selected from structurally eligible intervals with valid t-1 and t wealth, positive t-1 wealth, and a loss fraction of at least 75%. t+1 and outcome information are not used.",
  headers: payload.shock_flow.headers,
  rows: payload.shock_flow.rows,
  tableName: "ExposureShockFlow",
  widths: [75, 80, 105, 115, 125, 100, 95],
  numericFormats: {
    "Shock wave": "0",
    "Structural pass N": "#,##0",
    "Valid wealth pair N": "#,##0",
    "Positive t-1 wealth N": "#,##0",
    "Major shocks N": "#,##0",
    "First shocks N": "#,##0",
  },
  bodyHeight: 38,
  contextHeight: 64,
});

const recoverySheet = addReaderSheet({
  name: "3_Recovery QC",
  title: "Recovery support and divisor quality control",
  context: "Threshold percentages use classifiable index shocks. R >= 0.50 is primary; R >= 0.25 and R >= 1.00 are sensitivities; R >= 0.75 is audit-only. Divisor cross-tabs are QC and do not automatically trigger an inferential model.",
  headers: payload.recovery_qc.headers,
  rows: payload.recovery_qc.rows,
  tableName: "ExposureRecoveryQC",
  widths: [75, 90, 105, 180, 185, 135, 160, 145, 145],
  numericFormats: {"Shock wave": "0", "Index shocks N": "#,##0"},
  bodyHeight: 42,
  contextHeight: 48,
});
const divisorTitleRow = 4 + payload.recovery_qc.rows.length + 1;
const divisorTitle = recoverySheet.getRangeByIndexes(divisorTitleRow, 0, 1, payload.divisor_qc.headers.length);
divisorTitle.merge();
divisorTitle.values = [["Cohort-level divisor-change QC among classifiable index shocks"]];
divisorTitle.format.font = { name: fontFamily, size: 11, bold: true, color: "#1F2937" };
const divisorMatrix = [
  payload.divisor_qc.headers,
  ...payload.divisor_qc.rows.map((row) => payload.divisor_qc.headers.map((header) => row[header] ?? null)),
];
const divisorRange = recoverySheet.getRangeByIndexes(divisorTitleRow + 1, 0, divisorMatrix.length, payload.divisor_qc.headers.length);
divisorRange.values = divisorMatrix;
divisorRange.format.font = { name: fontFamily, size: 10, color: "#1F2937" };
divisorRange.format.verticalAlignment = "top";
divisorRange.format.wrapText = true;
const divisorHeader = recoverySheet.getRangeByIndexes(divisorTitleRow + 1, 0, 1, payload.divisor_qc.headers.length);
divisorHeader.format = {
  fill: "#4F81BD",
  font: { name: fontFamily, size: 10, bold: true, color: "#FFFFFF" },
  horizontalAlignment: "center",
  verticalAlignment: "center",
  wrapText: true,
};
divisorHeader.format.rowHeightPx = 42;
recoverySheet.getRangeByIndexes(divisorTitleRow + 2, 0, payload.divisor_qc.rows.length, payload.divisor_qc.headers.length).format.rowHeightPx = 38;
for (const headerName of ["Divisor changed %", "Recovery % stable", "Recovery % changed"]) {
  const index = payload.divisor_qc.headers.indexOf(headerName);
  recoverySheet.getRangeByIndexes(divisorTitleRow + 2, index, payload.divisor_qc.rows.length, 1).format.numberFormat = "0.0%";
}
for (const headerName of ["Classifiable N", "Divisor changed N"]) {
  const index = payload.divisor_qc.headers.indexOf(headerName);
  recoverySheet.getRangeByIndexes(divisorTitleRow + 2, index, payload.divisor_qc.rows.length, 1).format.numberFormat = "#,##0";
}
const divisorTable = recoverySheet.tables.add(divisorRange, true, "ExposureDivisorQC");
divisorTable.style = "TableStyleMedium2";
divisorTable.showFilterButton = true;
divisorTable.showBandedRows = true;
const noteRow = divisorTitleRow + divisorMatrix.length + 2;
const note = recoverySheet.getRangeByIndexes(noteRow, 0, 1, payload.recovery_qc.headers.length);
note.merge();
note.values = [[payload.recovery_note]];
note.format.font = { name: fontFamily, size: 9, italic: true, color: "#4B5563" };
note.format.wrapText = true;
note.format.rowHeightPx = 48;

const overview = await workbook.inspect({
  kind: "workbook,sheet,table",
  maxChars: 6000,
  tableMaxRows: 5,
  tableMaxCols: 12,
});
const formulaErrors = await workbook.inspect({
  kind: "match",
  searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!|#SPILL!|#CALC!",
  options: { useRegex: true, maxResults: 100 },
  summary: "final formula error scan",
});
await fs.mkdir(previewDir, { recursive: true });
for (const sheetName of ["1_Definition Contract", "2_Shock Flow", "3_Recovery QC"]) {
  const preview = await workbook.render({ sheetName, autoCrop: "all", scale: 1.5, format: "png" });
  const safeName = sheetName.replaceAll(" ", "_");
  await fs.writeFile(`${previewDir}/${safeName}.png`, new Uint8Array(await preview.arrayBuffer()));
}
const output = await SpreadsheetFile.exportXlsx(workbook);
await output.save(outputPath);
console.log(JSON.stringify({overview: overview.ndjson, formulaErrors: formulaErrors.ndjson}));
'''


def _runtime_paths() -> tuple[Path, Path]:
    """取得 bundled Node 与 artifact-tool runtime。"""
    runtime = (
        Path.home()
        / ".cache"
        / "codex-runtimes"
        / "codex-primary-runtime"
        / "dependencies"
    )
    node = runtime / "node" / "bin" / "node.exe"
    node_modules = runtime / "node" / "node_modules"
    if not node.is_file() or not node_modules.is_dir():
        raise RuntimeError("bundled artifact-tool runtime 不可用")
    return node, node_modules


def _create_node_modules_link(link: Path, target: Path) -> None:
    completed = subprocess.run(
        ["cmd", "/c", "mklink", "/J", str(link), str(target)],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if completed.returncode != 0:
        raise RuntimeError("无法创建 artifact-tool node_modules junction")


def create_review_workbook(
    shock_rows: list[dict[str, Any]],
    recovery_rows: list[dict[str, Any]],
    qc_by_cohort: dict[str, dict[str, Any]],
    output_path: Path,
    preview_dir: Path,
) -> dict[str, Any]:
    """用 artifact-tool 创建、检查并渲染三表汇总 workbook。"""
    divisor_rows = [qc_by_cohort[cohort]["divisor_visible"] for cohort in COHORTS]
    payload = {
        "definition": {
            "headers": list(DEFINITION_ROWS[0]),
            "rows": DEFINITION_ROWS,
        },
        "shock_flow": {
            "headers": list(shock_rows[0]),
            "rows": shock_rows,
        },
        "recovery_qc": {
            "headers": list(recovery_rows[0]),
            "rows": recovery_rows,
        },
        "divisor_qc": {
            "headers": list(divisor_rows[0]),
            "rows": divisor_rows,
        },
        "recovery_note": (
            "Divisor QC is recorded without automatic inferential sensitivity. "
            "A stable-divisor restriction requires separate scientific review and approval."
        ),
    }
    node, node_modules = _runtime_paths()
    with tempfile.TemporaryDirectory(prefix="exposure_panel_xlsx_") as tmp_name:
        tmp = Path(tmp_name)
        payload_path = tmp / "payload.json"
        builder_path = tmp / "build_review.mjs"
        payload_path.write_text(
            json.dumps(payload, ensure_ascii=False), encoding="utf-8"
        )
        builder_path.write_text(WORKBOOK_BUILDER, encoding="utf-8")
        _create_node_modules_link(tmp / "node_modules", node_modules)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        preview_dir.mkdir(parents=True, exist_ok=True)
        completed = subprocess.run(
            [
                str(node),
                str(builder_path),
                str(payload_path),
                str(output_path),
                str(preview_dir),
            ],
            cwd=tmp,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        if completed.returncode != 0:
            raise RuntimeError(
                "artifact-tool Exposure review workbook 构建失败：\n"
                + (completed.stdout or "")
                + "\n"
                + (completed.stderr or "")
            )
        if not output_path.is_file():
            raise RuntimeError("artifact-tool 未生成 Exposure review workbook")
        stdout = completed.stdout.strip()
        if any(
            token in stdout
            for token in ("#REF!", "#DIV/0!", "#VALUE!", "#NAME?", "#NUM!")
        ):
            raise RuntimeError("Exposure review workbook 公式错误扫描未通过")
        return {
            "artifact_tool_stdout": stdout,
            "preview_files": sorted(str(path) for path in preview_dir.glob("*.png")),
        }


DICTIONARY_TEXT = """# Exposure Panel v1 数据字典

## 数据粒度与主键

每行表示一个队列内某人的一个 cohort-global 相邻候选 `t-1 -> t` 区间。主键固定为 `cohort + person_id + candidate_shock_wave`，行集合和顺序与冻结 Candidate Intervals v3 完全一致。正式分析通过 `select_index_exposure()` 取得一人一行的 index exposure view；不另存第二份 index parquet。

## 十三列接口

- `cohort`、`person_id`、`candidate_shock_wave`：候选区间主键。
- `tminus1_wave`、`tplus1_wave`：冲击前与下一 cohort-global 调查波次；终末波的 `tplus1_wave` 为 `NA`。
- `wealth_tminus1`、`wealth_t`：从冻结 Wealth Panel v1 连接的实际价格等值化财富。零和负值原样保留。
- `wealth_tplus1`：只在 `is_index_shock == TRUE` 后读取并连接；非 index 行恒为 `NA`。
- `wealth_loss_fraction`：`(wealth_tminus1 - wealth_t) / wealth_tminus1`。只在结构资格通过、两波财富有效且 `wealth_tminus1 > 0` 时计算，不截断。
- `major_shock`：nullable boolean。它表示任一可评估候选区间是否达到 `>= 0.75` 财富损失，不论该区间是否位于预设 analytic shock-search window 内。可评估且达到阈值为 `TRUE`，可评估但未达到为 `FALSE`，不可评估为 `NA`。
- `is_index_shock`：仅标识队列特异预设 analytic shock-search window 内，每人按 `candidate_interval_order` 选择的首次 qualifying major shock；非空 boolean。锁定窗口为 HRS W3–W14、ELSA W2–W8、KLoSA W2–W7。因此，窗口外可出现 `major_shock == TRUE` 且 `is_index_shock == FALSE`，这是预期设计而非构建错误。
- `recovery_fraction_raw`：`(wealth_tplus1 - wealth_t) / (wealth_tminus1 - wealth_t)`。只在 index shock 且 `t+1` 财富有效时计算；负值和大于1的值原样保留，不 clipping、不 winsorise。
- `recovery_status`：`R >= 0.50` 为 `substantial_recovery`；`R < 0.50` 为 `persistent_or_limited_recovery`；无法分类为 `NA`。

## 敏感性与 QC

正式 threshold sensitivities 为 `R >= 0.25` 和 `R >= 1.00`，均由 `recovery_fraction_raw` 动态生成，不保存物理列。`R >= 0.75` 只用于内部 Exposure Audit。Raw continuous R 不作为线性 exposure sensitivity。

等值化 divisor 变化只在 aggregate exposure QC 中报告 `recovery_status × divisor_changed`，不作为排除条件，也不自动启动 inferential sensitivity。如出现疑似构建问题或明显异常集中，应在冻结前停止并提交人工复核。Exposure Panel 不读取 FI、健康结局、协变量、权重或模型结果。
"""


def _schema(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return [
        {
            "name": column,
            "dtype": str(frame[column].dtype),
            "nullable": bool(frame[column].isna().any()),
        }
        for column in frame.columns
    ]


def _final_output_path(cohort: str) -> Path:
    return (
        DERIVED_ROOT
        / cohort
        / "07_exposure_panel"
        / f"{cohort.lower()}_exposure_panel_v1.parquet"
    )


def _manifest_path(cohort: str) -> Path:
    return EXPOSURE_MANIFEST_ROOT / f"{cohort.lower()}_exposure_panel_v1.json"


def _active_release_paths() -> list[Path]:
    """返回受控替换的旧 Exposure v1 活动资产。"""
    return [
        *(_final_output_path(cohort) for cohort in COHORTS),
        *(_manifest_path(cohort) for cohort in COHORTS),
        REVIEW_PATH,
        DICTIONARY_PATH,
        FREEZE_PATH,
    ]


def _downstream_consumer_check() -> dict[str, Any]:
    """确认未建立 Exposure v1 之后的真实数据产物或 manifest。"""
    stage_names = (
        "08_covariate_panel",
        "09_weights",
        "10_landmark_ledger",
        "11_eligibility_final",
        "12_analysis_panel",
        "13_validate",
        "14_cohort_model",
        "15_meta",
    )
    manifest_names = (
        "covariate",
        "weights",
        "landmark_ledger",
        "eligibility_final",
        "analysis_panel",
        "cohort_model",
        "meta",
    )
    found: list[str] = []
    for cohort in COHORTS:
        cohort_root = DERIVED_ROOT / cohort
        for stage in stage_names:
            directory = cohort_root / stage
            if directory.is_dir():
                found.extend(_relative(path) for path in directory.rglob("*") if path.is_file())
    for name in manifest_names:
        directory = MANIFEST_ROOT / name
        if directory.is_dir():
            found.extend(_relative(path) for path in directory.rglob("*") if path.is_file())
    return {
        "checked_stage_directories": list(stage_names),
        "checked_manifest_directories": list(manifest_names),
        "consumer_files": sorted(found),
        "no_downstream_consumer": not found,
    }


def _verify_withdrawable_release() -> tuple[dict[str, Any], dict[str, pd.DataFrame]]:
    """核对旧 freeze 及全部旧资产，为一次性受控撤销建立门禁。"""
    if ARCHIVED_FREEZE_PATH.exists() or WITHDRAWAL_PATH.exists():
        raise RuntimeError("旧 v1 已有撤销归档记录，拒绝再次同名替换")
    missing = [str(path) for path in _active_release_paths() if not path.is_file()]
    if missing:
        raise RuntimeError("旧 Exposure v1 资产不完整：" + "; ".join(missing))
    freeze_sha256 = sha256_file(FREEZE_PATH)
    if freeze_sha256 != EXPECTED_OLD_FREEZE_SHA256:
        raise RuntimeError(
            f"旧 freeze 已漂移：{freeze_sha256}，预期 {EXPECTED_OLD_FREEZE_SHA256}"
        )
    freeze = json.loads(FREEZE_PATH.read_text(encoding="utf-8"))
    if freeze.get("freeze_status") != "FROZEN":
        raise RuntimeError("旧 Exposure v1 freeze_status 不是 FROZEN")
    assets: list[dict[str, Any]] = [
        {
            "asset_role": "freeze_record",
            "path": _relative(FREEZE_PATH),
            "sha256": freeze_sha256,
            "size_bytes": FREEZE_PATH.stat().st_size,
        }
    ]
    old_panels: dict[str, pd.DataFrame] = {}
    for cohort in COHORTS:
        record = freeze["cohorts"][cohort]
        for role, path_key, hash_key in (
            ("parquet", "parquet_path", "parquet_sha256"),
            ("manifest", "manifest_path", "manifest_sha256"),
        ):
            path = _repo_path(record[path_key])
            actual = sha256_file(path)
            if actual != record[hash_key]:
                raise RuntimeError(f"{cohort} 旧 {role} 已漂移")
            assets.append(
                {
                    "asset_role": role,
                    "cohort": cohort,
                    "path": _relative(path),
                    "sha256": actual,
                    "size_bytes": path.stat().st_size,
                }
            )
        old_panels[cohort] = pd.read_parquet(_repo_path(record["parquet_path"]))
        if len(old_panels[cohort]) != int(record["row_count"]):
            raise RuntimeError(f"{cohort}: 旧 parquet 行数与 freeze 不一致")
    for role, section in (
        ("review_workbook", "review_workbook"),
        ("data_dictionary", "data_dictionary"),
    ):
        record = freeze[section]
        path = _repo_path(record["path"])
        actual = sha256_file(path)
        if actual != record["sha256"]:
            raise RuntimeError(f"旧 {role} 已漂移")
        assets.append(
            {
                "asset_role": role,
                "path": _relative(path),
                "sha256": actual,
                "size_bytes": path.stat().st_size,
            }
        )
    downstream = _downstream_consumer_check()
    if not downstream["no_downstream_consumer"]:
        raise RuntimeError(
            "发现 Exposure v1 下游 consumer，停止撤销："
            + "; ".join(downstream["consumer_files"])
        )
    gate = {
        "pre_window_freeze_sha256": freeze_sha256,
        "old_assets": assets,
        "old_counts": {
            cohort: {
                key: freeze["cohorts"][cohort][key]
                for key in ("row_count", "index_shocks", "classifiable", "primary_recovery")
            }
            for cohort in COHORTS
        },
        "downstream_consumer_check": downstream,
    }
    return gate, old_panels


def _compare_chronology(
    cohort: str, old_panel: pd.DataFrame, new_panel: pd.DataFrame
) -> dict[str, int]:
    """比较窗口修正前后的个人 index shock 变化。"""
    old_index = old_panel.loc[
        old_panel["is_index_shock"].eq(True), ["person_id", "candidate_shock_wave"]
    ].rename(columns={"candidate_shock_wave": "old_wave"})
    new_index = new_panel.loc[
        new_panel["is_index_shock"].eq(True), ["person_id", "candidate_shock_wave"]
    ].rename(columns={"candidate_shock_wave": "new_wave"})
    comparison = old_index.merge(new_index, on="person_id", how="outer", validate="one_to_one")
    both = comparison["old_wave"].notna() & comparison["new_wave"].notna()
    result = {
        "same_person_same_wave": int(
            (both & comparison["old_wave"].eq(comparison["new_wave"])).sum()
        ),
        "shifted_to_later_in_window_shock": int(
            (both & comparison["new_wave"].gt(comparison["old_wave"])).sum()
        ),
        "dropped_no_window_index": int(
            (comparison["old_wave"].notna() & comparison["new_wave"].isna()).sum()
        ),
        "new_person": int(
            (comparison["old_wave"].isna() & comparison["new_wave"].notna()).sum()
        ),
    }
    if result != EXPECTED_CHRONOLOGY_COMPARISON[cohort]:
        raise RuntimeError(
            f"{cohort}: 新旧 chronology 差异 {result}，预期 "
            f"{EXPECTED_CHRONOLOGY_COMPARISON[cohort]}"
        )
    return result


def _build_manifest(
    cohort: str,
    panel: pd.DataFrame,
    *,
    staged_output: Path,
    input_snapshot: dict[str, Any],
    qc: dict[str, Any],
    chronology_comparison: dict[str, int],
    provenance: dict[str, Any],
) -> dict[str, Any]:
    return {
        "manifest_schema_version": "exposure_panel_manifest_v1",
        "release_version": RELEASE_VERSION,
        "cohort": cohort,
        "generated_utc": datetime.now(UTC).isoformat(),
        "git_commit": provenance["git_commit"],
        "git_dirty": provenance["git_dirty"],
        "git_diff_sha256": provenance["git_diff_sha256"],
        "runner_version": RUNNER_VERSION,
        "builder_version": BUILDER_VERSION,
        "builder_sha256": sha256_file(Path(__file__)),
        "spec_version": SPEC_VERSION,
        "specification_sha256": specification_hash(),
        "design_register_sha256": sha256_file(DESIGN_REGISTER_PATH),
        "key": list(KEY_COLUMNS),
        "row_count": len(panel),
        "key_unique": not panel.duplicated(list(KEY_COLUMNS)).any(),
        "schema": _schema(panel),
        "build_inputs": {
            **input_snapshot,
            "design_register": {
                "path": _relative(DESIGN_REGISTER_PATH),
                "sha256": sha256_file(DESIGN_REGISTER_PATH),
            },
        },
        "exposure_contract": specification(),
        "pre_window_v1_chronology_comparison": chronology_comparison,
        "counts": qc["counts"],
        "threshold_support": qc["threshold_support"],
        "recovery_distribution": qc["distribution"],
        "recovery_categories": qc["categories"],
        "extreme_value_qc": qc["extremes"],
        "divisor_change_qc": qc["divisor_qc"],
        "timing_qc": qc["timing"],
        "shock_flow_by_wave": qc["shock_flow_by_wave"],
        "recovery_by_wave": qc["recovery_by_wave"],
        "invariants": {
            "candidate_interval_grain_preserved": True,
            "major_shock_evaluated_across_all_candidate_intervals": True,
            "index_shock_restricted_to_locked_analytic_window": True,
            "at_most_one_index_shock_per_person": True,
            "index_count_equals_person_count": True,
            "index_requires_major_structural_pass_and_window": True,
            "threshold_counts_nested": True,
            "classifiable_equals_primary_groups_sum": True,
            "two_pass_first_shock_before_tplus1": True,
            "later_shock_substitution_forbidden": True,
            "missing_tplus1_not_comparator": True,
            "raw_recovery_not_clipped_or_winsorised": True,
            "fi_outcome_covariate_weight_or_model_read": False,
            "divisor_change_is_qc_only": True,
            "continuous_linear_recovery_model_created": False,
        },
        "output": {
            "path": _relative(_final_output_path(cohort)),
            "sha256": sha256_file(staged_output),
        },
    }


def _build_freeze(
    *,
    staged_outputs: dict[str, Path],
    staged_manifests: dict[str, Path],
    staged_review: Path,
    staged_dictionary: Path,
    input_snapshot: dict[str, Any],
    qc_by_cohort: dict[str, dict[str, Any]],
    chronology_by_cohort: dict[str, dict[str, int]],
    staged_archived_freeze: Path,
    staged_withdrawal: Path,
    provenance: dict[str, Any],
) -> dict[str, Any]:
    return {
        "freeze_record_schema_version": "exposure_panel_freeze_record_v1",
        "freeze_status": "FROZEN",
        "release_version": RELEASE_VERSION,
        "generated_utc": datetime.now(UTC).isoformat(),
        "git_commit": provenance["git_commit"],
        "git_dirty": provenance["git_dirty"],
        "git_diff_sha256": provenance["git_diff_sha256"],
        "runner_version": RUNNER_VERSION,
        "builder_version": BUILDER_VERSION,
        "builder_sha256": sha256_file(Path(__file__)),
        "spec_version": SPEC_VERSION,
        "specification_sha256": specification_hash(),
        "design_register": {
            "path": _relative(DESIGN_REGISTER_PATH),
            "sha256": sha256_file(DESIGN_REGISTER_PATH),
            "locked_decisions": verify_locked_design(),
        },
        "common_key": list(KEY_COLUMNS),
        "physical_columns": list(EXPOSURE_PANEL_COLUMNS),
        "exposure_contract": specification(),
        "cohorts": {
            cohort: {
                "parquet_path": _relative(_final_output_path(cohort)),
                "parquet_sha256": sha256_file(staged_outputs[cohort]),
                "manifest_path": _relative(_manifest_path(cohort)),
                "manifest_sha256": sha256_file(staged_manifests[cohort]),
                "row_count": int(len(pd.read_parquet(staged_outputs[cohort]))),
                "index_shocks": qc_by_cohort[cohort]["counts"]["index_shocks"],
                "classifiable": qc_by_cohort[cohort]["counts"]["classifiable"],
                "primary_recovery": qc_by_cohort[cohort]["counts"]["primary_recovery"],
                "primary_comparator": qc_by_cohort[cohort]["counts"]["primary_comparator"],
                "recovery_ge_0.25_n": qc_by_cohort[cohort]["counts"]["recovery_ge_0.25_n"],
                "recovery_ge_0.75_n": qc_by_cohort[cohort]["counts"]["recovery_ge_0.75_n"],
                "recovery_ge_1.00_n": qc_by_cohort[cohort]["counts"]["recovery_ge_1.00_n"],
                "pre_window_v1_chronology_comparison": chronology_by_cohort[cohort],
            }
            for cohort in COHORTS
        },
        "controlled_pre_window_v1_withdrawal": {
            "archived_freeze_path": _relative(ARCHIVED_FREEZE_PATH),
            "archived_freeze_sha256": sha256_file(staged_archived_freeze),
            "withdrawal_record_path": _relative(WITHDRAWAL_PATH),
            "withdrawal_record_sha256": sha256_file(staged_withdrawal),
            "original_freeze_sha256": EXPECTED_OLD_FREEZE_SHA256,
            "no_downstream_consumer_confirmed": True,
        },
        "review_workbook": {
            "path": _relative(REVIEW_PATH),
            "sha256": sha256_file(staged_review),
            "sheets": [
                "1_Definition Contract",
                "2_Shock Flow",
                "3_Recovery QC",
            ],
            "formula_error_scan": "PASS",
            "three_sheet_render": "PASS",
        },
        "data_dictionary": {
            "path": _relative(DICTIONARY_PATH),
            "sha256": sha256_file(staged_dictionary),
        },
        "input_snapshot": input_snapshot,
        "protected_inputs_before_and_after_identical": True,
        "fi_or_health_outcome_is_not_a_build_input": True,
        "sensitivity_physical_columns_created": False,
        "overwrite_policy": (
            "This corrected replacement v1 is immutable; do not overwrite it or this "
            "freeze record, and create a new version for any later structural change."
        ),
        "partial_publish_policy": (
            "Build and validate all cohorts in system temporary staging; publish only "
            "after all checks pass; write freeze last."
        ),
        "next_stage_status": "COVARIATE_NOT_STARTED",
    }


def _publish(staged: Path, final: Path) -> None:
    if final.exists():
        raise RuntimeError(f"拒绝覆盖已有文件：{final}")
    final.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(staged), str(final))


def _publish_controlled_replacement(
    *,
    staged_by_final: list[tuple[Path, Path]],
    rollback_root: Path,
) -> None:
    """将旧活动资产移入临时 rollback，并以 freeze 最后发布。"""
    rollback_map: list[tuple[Path, Path]] = []
    published: list[Path] = []
    try:
        for final in _active_release_paths():
            if not final.is_file():
                raise RuntimeError(f"受控替换前旧资产消失：{final}")
            relative = final.resolve().relative_to(ROOT.resolve())
            rollback = rollback_root / relative
            rollback.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(final), str(rollback))
            rollback_map.append((rollback, final))
        for staged, final in staged_by_final:
            _publish(staged, final)
            published.append(final)
    except Exception:
        for final in reversed(published):
            resolved = final.resolve()
            if ROOT.resolve() not in resolved.parents:
                raise RuntimeError(f"拒绝回滚删除仓库外文件：{final}")
            if final.is_file():
                final.unlink()
        for rollback, final in reversed(rollback_map):
            final.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(rollback), str(final))
        raise


def run_exposure_panel_release(
    cohorts: list[str],
    *,
    load_candidate_intervals: Callable[
        [str], tuple[pd.DataFrame, Path, dict[str, Any]]
    ],
    load_eligibility_base: Callable[
        [str], tuple[pd.DataFrame, Path, dict[str, Any]]
    ],
) -> dict[str, Any]:
    """受控撤销旧 v1，按锁定窗口重建、发布并冻结替换 v1。"""
    if tuple(cohorts) != COHORTS:
        raise RuntimeError("exposure v1 必须使用 --cohort all 一次性生成共同 review/freeze")
    withdrawal_gate, old_panels = _verify_withdrawable_release()
    verify_locked_design()
    frames, inputs_before = load_frozen_inputs(
        load_candidate_intervals=load_candidate_intervals,
        load_eligibility_base=load_eligibility_base,
    )

    panels: dict[str, pd.DataFrame] = {}
    qc_by_cohort: dict[str, dict[str, Any]] = {}
    chronology_by_cohort: dict[str, dict[str, int]] = {}
    for cohort in COHORTS:
        panel, internal = build_exposure_panel(
            frames[cohort]["candidate"],
            frames[cohort]["eligibility"],
            frames[cohort]["wealth"],
            cohort,
        )
        qc = build_qc(cohort, panel, internal)
        for field, expected in EXPECTED_RELEASE_COUNTS[cohort].items():
            if qc["counts"][field] != expected:
                raise RuntimeError(
                    f"{cohort}: release count {field}={qc['counts'][field]}，预期 {expected}"
                )
        chronology_by_cohort[cohort] = _compare_chronology(
            cohort, old_panels[cohort], panel
        )
        panels[cohort] = panel
        qc_by_cohort[cohort] = qc

    if sum(len(qc_by_cohort[cohort]["shock_visible"]) for cohort in COHORTS) != 25:
        raise RuntimeError("review workbook 的 analytic shock-window 波次必须恰为25行")

    provenance = git_provenance()
    preview_dir = Path(tempfile.gettempdir()) / "exposure_panel_review_v1_final_previews"
    if preview_dir.exists():
        shutil.rmtree(preview_dir)

    with tempfile.TemporaryDirectory(prefix="exposure_panel_v1_") as tmp_name:
        staging = Path(tmp_name)
        rollback_root = staging / "rollback"
        staged_outputs: dict[str, Path] = {}
        for cohort in COHORTS:
            staged = staging / _final_output_path(cohort).name
            panels[cohort].to_parquet(staged, index=False, compression="zstd")
            reloaded = pd.read_parquet(staged)
            validate_exposure_panel(
                reloaded,
                frames[cohort]["candidate"],
                frames[cohort]["eligibility"],
                cohort,
            )
            staged_outputs[cohort] = staged

        staged_review = staging / REVIEW_PATH.name
        workbook_checks = create_review_workbook(
            [row for cohort in COHORTS for row in qc_by_cohort[cohort]["shock_visible"]],
            [row for cohort in COHORTS for row in qc_by_cohort[cohort]["recovery_visible"]],
            qc_by_cohort,
            staged_review,
            preview_dir,
        )
        if len(workbook_checks["preview_files"]) != 3:
            raise RuntimeError("Exposure review workbook 未完成三 Sheet 渲染")

        staged_dictionary = staging / DICTIONARY_PATH.name
        staged_dictionary.write_text(DICTIONARY_TEXT, encoding="utf-8")
        staged_archived_freeze = staging / ARCHIVED_FREEZE_PATH.name
        shutil.copyfile(FREEZE_PATH, staged_archived_freeze)
        if sha256_file(staged_archived_freeze) != EXPECTED_OLD_FREEZE_SHA256:
            raise RuntimeError("旧 freeze 归档副本未逐字节保真")
        withdrawn_utc = datetime.now(UTC).isoformat()
        withdrawal_payload = {
            "withdrawal_record_schema_version": "exposure_panel_withdrawal_record_v1",
            "withdrawal_status": "WITHDRAWN_AND_REPLACED",
            "withdrawn_utc": withdrawn_utc,
            "withdrawal_reason": (
                "The pre-window Exposure Panel v1 selected the first major shock across "
                "all cohort-global candidate waves. The approved analytic estimand requires "
                "the first qualifying major shock within the prespecified cohort-specific "
                "analytic shock-search window."
            ),
            "replacement_release": "Exposure Panel v1 corrected window-constrained replacement",
            "original_freeze": {
                "path": _relative(ARCHIVED_FREEZE_PATH),
                "sha256": EXPECTED_OLD_FREEZE_SHA256,
                "archived_byte_for_byte": True,
            },
            "old_assets": withdrawal_gate["old_assets"],
            "old_counts": withdrawal_gate["old_counts"],
            "chronology_comparison_to_replacement": chronology_by_cohort,
            "baseline_repository_test": {
                "command": "uv run --locked pytest -q",
                "passed": 44,
                "failed": 1,
                "known_failure": BASELINE_KNOWN_FAILURE,
                "failure_is_unrelated_to_exposure": True,
                "supplementary_file_change_authorized": False,
            },
            "downstream_consumer_check": withdrawal_gate[
                "downstream_consumer_check"
            ],
            "old_parquet_retention": (
                "Not retained after successful transactional replacement; hashes, sizes, "
                "counts, manifests, and the exact old freeze remain permanently recorded."
            ),
            "rollback_policy": (
                "Old active assets are retained in system-temporary rollback storage until "
                "all replacement assets are published with the new freeze written last."
            ),
        }
        staged_withdrawal = staging / WITHDRAWAL_PATH.name
        _json_write(staged_withdrawal, withdrawal_payload)
        staged_manifests: dict[str, Path] = {}
        for cohort in COHORTS:
            manifest = _build_manifest(
                cohort,
                panels[cohort],
                staged_output=staged_outputs[cohort],
                input_snapshot=inputs_before["cohorts"][cohort],
                qc=qc_by_cohort[cohort],
                chronology_comparison=chronology_by_cohort[cohort],
                provenance=provenance,
            )
            staged_manifest = staging / _manifest_path(cohort).name
            _json_write(staged_manifest, manifest)
            staged_manifests[cohort] = staged_manifest

        _frames_after, inputs_after = load_frozen_inputs(
            load_candidate_intervals=load_candidate_intervals,
            load_eligibility_base=load_eligibility_base,
        )
        if inputs_before != inputs_after:
            raise RuntimeError("Exposure 构建期间 Candidate/Eligibility/Wealth/design register 发生变化")

        staged_freeze = staging / FREEZE_PATH.name
        _json_write(
            staged_freeze,
            _build_freeze(
                staged_outputs=staged_outputs,
                staged_manifests=staged_manifests,
                staged_review=staged_review,
                staged_dictionary=staged_dictionary,
                input_snapshot=inputs_before,
                qc_by_cohort=qc_by_cohort,
                chronology_by_cohort=chronology_by_cohort,
                staged_archived_freeze=staged_archived_freeze,
                staged_withdrawal=staged_withdrawal,
                provenance=provenance,
            ),
        )
        staged_by_final = [
            (staged_archived_freeze, ARCHIVED_FREEZE_PATH),
            (staged_withdrawal, WITHDRAWAL_PATH),
            *((staged_outputs[cohort], _final_output_path(cohort)) for cohort in COHORTS),
            (staged_review, REVIEW_PATH),
            (staged_dictionary, DICTIONARY_PATH),
            *((staged_manifests[cohort], _manifest_path(cohort)) for cohort in COHORTS),
            (staged_freeze, FREEZE_PATH),
        ]
        _publish_controlled_replacement(
            staged_by_final=staged_by_final,
            rollback_root=rollback_root,
        )

    if sha256_file(ARCHIVED_FREEZE_PATH) != EXPECTED_OLD_FREEZE_SHA256:
        raise RuntimeError("发布后旧 freeze 归档哈希不一致")

    return {
        "freeze_status": "FROZEN",
        "freeze_path": str(FREEZE_PATH),
        "preview_dir": str(preview_dir),
        "archived_pre_window_freeze": str(ARCHIVED_FREEZE_PATH),
        "withdrawal_record": str(WITHDRAWAL_PATH),
        "chronology_comparison": chronology_by_cohort,
        "cohorts": {
            cohort: {
                "rows": len(panels[cohort]),
                **qc_by_cohort[cohort]["counts"],
                "output": str(_final_output_path(cohort)),
            }
            for cohort in COHORTS
        },
        "review_workbook": str(REVIEW_PATH),
    }


__all__ = [
    "AUDIT_ONLY_THRESHOLDS",
    "ARCHIVED_FREEZE_PATH",
    "BUILDER_VERSION",
    "EXPECTED_RELEASE_COUNTS",
    "EXPOSURE_PANEL_COLUMNS",
    "EXPOSURE_PANEL_DTYPES",
    "FREEZE_PATH",
    "PRIMARY_RECOVERY_THRESHOLD",
    "REVIEW_PATH",
    "SENSITIVITY_THRESHOLDS",
    "SHOCK_THRESHOLD",
    "SHOCK_SEARCH_WINDOWS",
    "WITHDRAWAL_PATH",
    "attach_recovery_pass",
    "build_exposure_panel",
    "build_qc",
    "build_shock_pass",
    "derive_recovery_indicator",
    "run_exposure_panel_release",
    "select_index_exposure",
    "specification",
    "specification_hash",
    "validate_exposure_panel",
    "verify_locked_design",
]
