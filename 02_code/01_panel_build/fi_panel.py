"""正式 FI specification、七列 person-wave panel、QC 与冻结。"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from outcome_components import (
    COMMON_22,
    COMPONENT_SPECS,
    KEY_COLUMNS,
    PRIMARY_ITEMS,
    build_components_from_raw,
    map_version,
    read_raw_source,
    score_series,
    validate_outcome_frame,
    verify_prepanels,
)
from panel_core import (
    RUNNER_VERSION,
    normalise_person_id,
    resolve_cohort_dataset_path,
    sha256_file,
)


ROOT = Path(__file__).resolve().parents[2]
ADMIN = ROOT / "00_admin"
MANIFEST_ROOT = ADMIN / "panel_manifests"
FI_MANIFEST_ROOT = MANIFEST_ROOT / "fi_panel"
DERIVED_ROOT = ROOT / "01_data" / "derived" / "panel_components"
DESIGN_REGISTER_PATH = ADMIN / "panel_design_register.csv"
REGISTRY_PATH = ADMIN / "cohort_registry.csv"
ELIGIBILITY_MAP_PATH = ADMIN / "variable_maps" / "eligibility_base_variable_map.csv"
COMPONENT_MAP_PATH = ADMIN / "variable_maps" / "fi_component_map.csv"
OUTCOME_FREEZE_PATH = (
    MANIFEST_ROOT / "outcome_components" / "outcome_components_v2_freeze.json"
)
V1_FREEZE_PATH = FI_MANIFEST_ROOT / "fi_panel_v1_freeze.json"

COHORTS = ("HRS", "ELSA", "KLoSA")


@dataclass(frozen=True)
class FIReleaseSpec:
    """把历史与当前 FI release 的门槛和版本显式绑定。"""

    release_version: str
    builder_version: str
    spec_version: str
    primary_threshold: int
    common22_threshold: int
    threshold_proportion: float


V1_SPEC = FIReleaseSpec(
    release_version="v1",
    builder_version="fi_panel_builder_v1.0.0",
    spec_version="fi_spec_v1",
    primary_threshold=24,
    common22_threshold=18,
    threshold_proportion=0.80,
)
V2_SPEC = FIReleaseSpec(
    release_version="v2",
    builder_version="fi_panel_builder_v2.0.0",
    spec_version="fi_spec_v2",
    primary_threshold=27,
    common22_threshold=20,
    threshold_proportion=0.90,
)
ACTIVE_SPEC = V2_SPEC
RELEASE_VERSION = ACTIVE_SPEC.release_version
BUILDER_VERSION = ACTIVE_SPEC.builder_version
SPEC_VERSION = ACTIVE_SPEC.spec_version
REVIEW_PATH = ROOT / "04_collaboration" / f"fi_finalisation_review_{RELEASE_VERSION}.xlsx"
DICTIONARY_PATH = FI_MANIFEST_ROOT / f"fi_panel_data_dictionary_{RELEASE_VERSION}.md"
FREEZE_PATH = FI_MANIFEST_ROOT / f"fi_panel_{RELEASE_VERSION}_freeze.json"
FI_PANEL_COLUMNS = (
    "cohort",
    "person_id",
    "wave",
    "fi_primary",
    "fi_primary_n_valid",
    "fi_common22",
    "fi_common22_n_valid",
)
FORMAL_WINDOWS = {
    "HRS": (3, 16),
    "ELSA": (1, 10),
    "KLoSA": (1, 9),
}
PRIMARY_QC80_THRESHOLD = V1_SPEC.primary_threshold
COMMON22_QC80_THRESHOLD = V1_SPEC.common22_threshold
PRIMARY_QC90_THRESHOLD = 27
COMMON22_QC90_THRESHOLD = 20
PRIMARY_THRESHOLD = ACTIVE_SPEC.primary_threshold
COMMON22_THRESHOLD = ACTIVE_SPEC.common22_threshold

LOCKED_DESIGN = {
    "fi_primary_definition": "cohort_specific_primary30",
    "fi_common22_definition": "common22_sensitivity",
    "fi_completeness_rule": "at_least_90_percent",
    "fi_window_HRS": "W3-W16",
    "fi_window_ELSA": "W1-W10",
    "fi_window_KLoSA": "W1-W9",
}


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    """以 UTF-8 读取登记表。"""
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _repo_path(value: str | Path) -> Path:
    """把 manifest 中的相对路径解析到仓库根。"""
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _relative(path: Path) -> str:
    """输出使用正斜线的仓库相对路径。"""
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


def _finite_float(value: Any) -> float | None:
    """把统计量转换为可序列化有限浮点数。"""
    if value is None or pd.isna(value) or not np.isfinite(float(value)):
        return None
    return float(value)


def fi_specification(release_spec: FIReleaseSpec = ACTIVE_SPEC) -> dict[str, Any]:
    """返回唯一、可哈希的正式 FI specification。"""
    specification = {
        "spec_version": release_spec.spec_version,
        "primary_definition": "cohort_specific_primary30",
        "common22_definition": "common22_sensitivity",
        "formal_windows": {
            cohort: {"wave_start": bounds[0], "wave_end": bounds[1]}
            for cohort, bounds in FORMAL_WINDOWS.items()
        },
        "primary_threshold": {
            "item_count": 30,
            "minimum_valid": release_spec.primary_threshold,
            "proportion": release_spec.threshold_proportion,
        },
        "common22_threshold": {
            "item_count": 22,
            "minimum_valid": release_spec.common22_threshold,
            "proportion": release_spec.threshold_proportion,
        },
        "primary_items": {
            cohort: list(PRIMARY_ITEMS[cohort]) for cohort in COHORTS
        },
        "common22_items": list(COMMON_22),
        "missing_policy": "Missing components are excluded from numerator and denominator and are never scored as zero.",
    }
    if release_spec == V1_SPEC:
        specification["qc90_reference"] = {
            "primary_minimum_valid": V2_SPEC.primary_threshold,
            "common22_minimum_valid": V2_SPEC.common22_threshold,
            "creates_alternative_fi": False,
        }
    else:
        specification["qc80_reference"] = {
            "primary_minimum_valid": V1_SPEC.primary_threshold,
            "common22_minimum_valid": V1_SPEC.common22_threshold,
            "creates_alternative_fi": False,
        }
    return specification


def specification_hash(release_spec: FIReleaseSpec = ACTIVE_SPEC) -> str:
    """返回正式 FI specification 的内容哈希。"""
    encoded = json.dumps(
        fi_specification(release_spec),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def verify_locked_design(
    release_spec: FIReleaseSpec = ACTIVE_SPEC,
) -> dict[str, dict[str, str]]:
    """确认所有 FI 决定已在 design register 精确锁定。"""
    rows = _read_csv_rows(DESIGN_REGISTER_PATH)
    by_id = {row["decision_id"]: row for row in rows}
    resolved: dict[str, dict[str, str]] = {}
    expected_design = dict(LOCKED_DESIGN)
    expected_design["fi_completeness_rule"] = (
        "at_least_80_percent" if release_spec == V1_SPEC else "at_least_90_percent"
    )
    for decision_id, expected in expected_design.items():
        row = by_id.get(decision_id)
        if row is None:
            raise RuntimeError(f"design register 缺少 {decision_id}")
        if row["status"] != "locked" or row["selected_value"] != expected:
            raise RuntimeError(
                f"{decision_id} 未 locked 为 {expected}: "
                f"status={row['status']}, selected={row['selected_value']}"
            )
        resolved[decision_id] = row
    return resolved


def _final_output_path(
    cohort: str,
    release_spec: FIReleaseSpec = ACTIVE_SPEC,
) -> Path:
    return (
        DERIVED_ROOT
        / cohort
        / "05_fi_panel"
        / f"{cohort.lower()}_fi_panel_{release_spec.release_version}.parquet"
    )


def _manifest_path(
    cohort: str,
    release_spec: FIReleaseSpec = ACTIVE_SPEC,
) -> Path:
    return (
        FI_MANIFEST_ROOT
        / f"{cohort.lower()}_fi_panel_{release_spec.release_version}.json"
    )


def _verify_release_absent(release_spec: FIReleaseSpec = ACTIVE_SPEC) -> None:
    """拒绝覆盖任何同名正式资产或半成品。"""
    paths = [
        *(_final_output_path(cohort, release_spec) for cohort in COHORTS),
        *(_manifest_path(cohort, release_spec) for cohort in COHORTS),
        REVIEW_PATH,
        DICTIONARY_PATH,
        FREEZE_PATH,
    ]
    existing = [str(path) for path in paths if path.exists()]
    if existing:
        raise RuntimeError(
            f"发现同名 FI panel {release_spec.release_version} 资产，拒绝覆盖：{existing}"
        )


def _v1_release_snapshot() -> dict[str, str]:
    """核对并快照全部历史 FI Panel v1 冻结资产。"""
    if not V1_FREEZE_PATH.is_file():
        raise RuntimeError("FI Panel v1 freeze 不存在，不能发布 v2")
    freeze = json.loads(V1_FREEZE_PATH.read_text(encoding="utf-8"))
    if freeze.get("freeze_status") != "FROZEN":
        raise RuntimeError("FI Panel v1 未处于 FROZEN 状态")
    expected: dict[str, str] = {
        _relative(V1_FREEZE_PATH): sha256_file(V1_FREEZE_PATH),
        freeze["review_workbook"]["path"]: freeze["review_workbook"]["sha256"],
        freeze["data_dictionary"]["path"]: freeze["data_dictionary"]["sha256"],
    }
    for record in freeze["cohorts"].values():
        expected[record["parquet_path"]] = record["parquet_sha256"]
        expected[record["manifest_path"]] = record["manifest_sha256"]
    for relative, expected_hash in expected.items():
        path = _repo_path(relative)
        if not path.is_file() or sha256_file(path) != expected_hash:
            raise RuntimeError(f"FI Panel v1 冻结资产缺失或哈希不一致：{relative}")
    return expected


def _load_frozen_outcomes(
    load_spine: Callable[[str], tuple[pd.DataFrame, Path, dict[str, Any]]],
) -> tuple[
    dict[str, pd.DataFrame],
    dict[str, pd.DataFrame],
    dict[str, dict[str, Any]],
    dict[str, Any],
]:
    """只从 Outcome v2 freeze 解析正式构建输入。"""
    freeze = json.loads(OUTCOME_FREEZE_PATH.read_text(encoding="utf-8"))
    if freeze.get("freeze_status") != "FROZEN":
        raise RuntimeError("Outcome Components v2 未冻结")
    if freeze.get("map_version") != map_version():
        raise RuntimeError("Outcome freeze map_version 与当前可执行 mapping 不一致")
    registered = freeze.get("registered_variable_map", {})
    if sha256_file(COMPONENT_MAP_PATH) != registered.get("sha256"):
        raise RuntimeError("fi_component_map.csv 与 Outcome freeze 不一致")

    outcomes: dict[str, pd.DataFrame] = {}
    spines: dict[str, pd.DataFrame] = {}
    inputs: dict[str, dict[str, Any]] = {}
    for cohort in COHORTS:
        spine, spine_path, spine_manifest = load_spine(cohort)
        record = freeze["cohorts"][cohort]
        outcome_path = _repo_path(record["parquet_path"])
        outcome_manifest_path = _repo_path(record["manifest_path"])
        if sha256_file(outcome_path) != record["parquet_sha256"]:
            raise RuntimeError(f"{cohort}: Outcome v2 Parquet 哈希不一致")
        if sha256_file(outcome_manifest_path) != record["manifest_sha256"]:
            raise RuntimeError(f"{cohort}: Outcome v2 manifest 哈希不一致")
        manifest = json.loads(outcome_manifest_path.read_text(encoding="utf-8"))
        if manifest.get("output_hash") != record["parquet_sha256"]:
            raise RuntimeError(f"{cohort}: Outcome v2 manifest 未指向冻结输出")
        outcome = pd.read_parquet(outcome_path)
        validate_outcome_frame(outcome, spine, cohort)
        if len(PRIMARY_ITEMS[cohort]) != 30 or len(set(PRIMARY_ITEMS[cohort])) != 30:
            raise RuntimeError(f"{cohort}: primary item set 不是精确 30 项")
        if len(COMMON_22) != 22 or not set(COMMON_22).issubset(PRIMARY_ITEMS[cohort]):
            raise RuntimeError(f"{cohort}: common-22 契约不正确")
        outcomes[cohort] = outcome
        spines[cohort] = spine
        inputs[cohort] = {
            "outcome_components": {
                "path": _relative(outcome_path),
                "sha256": record["parquet_sha256"],
                "manifest_path": _relative(outcome_manifest_path),
                "manifest_sha256": record["manifest_sha256"],
            },
            "spine": {
                "path": _relative(spine_path),
                "sha256": spine_manifest["output"]["sha256"],
            },
        }
    return outcomes, spines, inputs, freeze


def build_fi_panel(
    outcome: pd.DataFrame,
    spine: pd.DataFrame,
    cohort: str,
    release_spec: FIReleaseSpec = ACTIVE_SPEC,
) -> pd.DataFrame:
    """只由 Outcome Components、Spine 和已锁定 specification 计算七列 FI。"""
    primary_items = list(PRIMARY_ITEMS[cohort])
    common_items = list(COMMON_22)
    start, end = FORMAL_WINDOWS[cohort]
    if not outcome.loc[:, list(KEY_COLUMNS)].equals(
        spine.loc[:, list(KEY_COLUMNS)].astype(
            {"cohort": "string", "person_id": "string", "wave": "int16"}
        )
    ):
        raise ValueError(f"{cohort}: Outcome 与 Spine 键或行序不一致")

    opportunity = (
        spine["observed_in_wave"].eq(True).fillna(False)
        & spine["wave"].between(start, end)
    )
    primary_raw = outcome[primary_items].notna().sum(axis=1)
    common_raw = outcome[common_items].notna().sum(axis=1)
    primary_n = pd.Series(pd.NA, index=outcome.index, dtype="Int16")
    common_n = pd.Series(pd.NA, index=outcome.index, dtype="Int16")
    primary_n.loc[opportunity] = primary_raw.loc[opportunity].astype("int16")
    common_n.loc[opportunity] = common_raw.loc[opportunity].astype("int16")

    primary_score = pd.Series(pd.NA, index=outcome.index, dtype="Float64")
    common_score = pd.Series(pd.NA, index=outcome.index, dtype="Float64")
    primary_complete = opportunity & primary_raw.ge(release_spec.primary_threshold)
    common_complete = opportunity & common_raw.ge(release_spec.common22_threshold)
    primary_score.loc[primary_complete] = (
        outcome.loc[primary_complete, primary_items]
        .sum(axis=1, skipna=True)
        .div(primary_raw.loc[primary_complete])
        .astype(float)
    )
    common_score.loc[common_complete] = (
        outcome.loc[common_complete, common_items]
        .sum(axis=1, skipna=True)
        .div(common_raw.loc[common_complete])
        .astype(float)
    )

    panel = spine.loc[:, list(KEY_COLUMNS)].copy()
    panel = panel.astype({"cohort": "string", "person_id": "string", "wave": "int16"})
    panel["fi_primary"] = primary_score
    panel["fi_primary_n_valid"] = primary_n
    panel["fi_common22"] = common_score
    panel["fi_common22_n_valid"] = common_n
    return panel.loc[:, list(FI_PANEL_COLUMNS)]


def with_fi_completeness_flags(
    frame: pd.DataFrame,
    release_spec: FIReleaseSpec = ACTIVE_SPEC,
) -> pd.DataFrame:
    """按需动态派生完整性 boolean，不改变物理 panel。"""
    output = frame.copy()
    output["fi_primary_complete"] = (
        output["fi_primary_n_valid"].ge(release_spec.primary_threshold)
        .where(output["fi_primary_n_valid"].notna())
        .astype("boolean")
    )
    output["fi_common22_complete"] = (
        output["fi_common22_n_valid"].ge(release_spec.common22_threshold)
        .where(output["fi_common22_n_valid"].notna())
        .astype("boolean")
    )
    return output


def validate_fi_panel(
    frame: pd.DataFrame,
    spine: pd.DataFrame,
    outcome: pd.DataFrame,
    cohort: str,
    release_spec: FIReleaseSpec = ACTIVE_SPEC,
) -> None:
    """验证七列 schema、掩蔽、分母、公式和取值范围。"""
    if list(frame.columns) != list(FI_PANEL_COLUMNS):
        raise ValueError(f"{cohort}: FI panel 列顺序不一致")
    expected_dtypes = {
        "cohort": "string",
        "person_id": "string",
        "wave": "int16",
        "fi_primary": "Float64",
        "fi_primary_n_valid": "Int16",
        "fi_common22": "Float64",
        "fi_common22_n_valid": "Int16",
    }
    actual_dtypes = {column: str(frame[column].dtype) for column in frame.columns}
    if actual_dtypes != expected_dtypes:
        raise TypeError(
            f"{cohort}: FI panel dtype 不一致：actual={actual_dtypes}, expected={expected_dtypes}"
        )
    if len(frame) != len(spine) or frame.duplicated(list(KEY_COLUMNS)).any():
        raise ValueError(f"{cohort}: FI panel 行数或主键不正确")
    pd.testing.assert_frame_equal(
        frame.loc[:, list(KEY_COLUMNS)],
        spine.loc[:, list(KEY_COLUMNS)].astype(
            {"cohort": "string", "person_id": "string", "wave": "int16"}
        ),
        check_dtype=True,
        check_exact=True,
    )

    start, end = FORMAL_WINDOWS[cohort]
    opportunity = (
        spine["observed_in_wave"].eq(True).fillna(False)
        & spine["wave"].between(start, end)
    )
    value_columns = list(FI_PANEL_COLUMNS[3:])
    if frame.loc[~opportunity, value_columns].notna().any(axis=None):
        raise ValueError(f"{cohort}: 未访谈或正式窗口外出现 FI/n_valid")
    if frame.loc[opportunity, ["fi_primary_n_valid", "fi_common22_n_valid"]].isna().any(axis=None):
        raise ValueError(f"{cohort}: 正式测量机会缺少 n_valid")

    for score, count, threshold, items in (
        (
            "fi_primary",
            "fi_primary_n_valid",
            release_spec.primary_threshold,
            list(PRIMARY_ITEMS[cohort]),
        ),
        (
            "fi_common22",
            "fi_common22_n_valid",
            release_spec.common22_threshold,
            list(COMMON_22),
        ),
    ):
        complete = frame[count].ge(threshold).fillna(False)
        if not frame[score].notna().equals(complete.astype(bool)):
            raise ValueError(f"{cohort}: {score} 与 {count} 门槛关系不一致")
        expected = outcome[items].sum(axis=1, skipna=True).div(frame[count].astype("Float64"))
        if not np.allclose(
            frame.loc[complete, score].astype(float),
            expected.loc[complete].astype(float),
            rtol=0,
            atol=1e-12,
        ):
            raise ValueError(f"{cohort}: {score} 公式不一致")
        values = frame[score].dropna()
        if not values.between(0, 1, inclusive="both").all():
            raise ValueError(f"{cohort}: {score} 超出 [0,1]")


def _demographic_long(cohort: str) -> pd.DataFrame:
    """仅为聚合 QC 读取 person ID、性别和逐波年龄，不写入 panel。"""
    registry = next(
        row for row in _read_csv_rows(REGISTRY_PATH) if row["cohort"] == cohort
    )
    registry["dataset_path"] = str(resolve_cohort_dataset_path(cohort))
    age_row = next(
        row
        for row in _read_csv_rows(ELIGIBILITY_MAP_PATH)
        if row["cohort"] == cohort and row["canonical_field"] == "age_in_years"
    )
    age_sources = {
        wave: age_row["source_variable_template"].format(wave=wave)
        for wave in range(int(age_row["wave_min"]), int(age_row["wave_max"]) + 1)
    }
    id_source = registry["person_id_source"]
    columns = [id_source, "ragender", *age_sources.values()]
    raw = pd.read_stata(
        registry["dataset_path"],
        columns=columns,
        convert_categoricals=False,
        preserve_dtypes=False,
    )
    person_id = normalise_person_id(raw[id_source], cohort)
    gender = pd.to_numeric(raw["ragender"], errors="coerce")
    female = pd.Series(pd.NA, index=raw.index, dtype="boolean")
    female.loc[gender.eq(1)] = False
    female.loc[gender.eq(2)] = True
    frames: list[pd.DataFrame] = []
    for wave, source in age_sources.items():
        age = pd.to_numeric(raw[source], errors="coerce")
        age = age.where(age.between(0, 120))
        frames.append(
            pd.DataFrame(
                {
                    "person_id": person_id,
                    "wave": pd.Series(wave, index=raw.index, dtype="int16"),
                    "age": age.astype("Float64"),
                    "female": female,
                }
            )
        )
    output = pd.concat(frames, ignore_index=True)
    if output.duplicated(["person_id", "wave"]).any():
        raise RuntimeError(f"{cohort}: demographic QC 键不唯一")
    return output


def _underweight_source_checks() -> dict[str, Any]:
    """核对 KLoSA W1-W9 BMI category 的标签、类别和 mapping。"""
    registry = next(
        row for row in _read_csv_rows(REGISTRY_PATH) if row["cohort"] == "KLoSA"
    )
    registry["dataset_path"] = str(resolve_cohort_dataset_path("KLoSA"))
    variables = [f"r{wave}bmicat" for wave in range(1, 10)]
    with pd.io.stata.StataReader(registry["dataset_path"]) as reader:
        labels = reader.variable_labels()
        value_labels = reader.value_labels()
        dataset_label = str(reader.data_label)
    expected_categories = {
        1: "1.underweight, less than 18.5",
        2: "2.normal weight, 18.5 to 22.9",
        3: "3.pre-obesity, 23 to 24.9",
        4: "4.obesity class 1, 25 to 29.9",
        5: "5.obesity class 2, 30 to 34.9",
        6: "6.obesity class 3, 35 or greater",
    }
    bmicat_labels = {
        int(code): str(label)
        for code, label in value_labels.get("bmicats", {}).items()
        if int(code) in expected_categories
    }
    variable_labels = {variable: str(labels.get(variable, "")) for variable in variables}
    label_stable = all(
        variable_labels[variable] == f"{variable}:w{wave} R BMI categorization"
        for wave, variable in enumerate(variables, start=1)
    )
    raw = pd.read_stata(
        registry["dataset_path"],
        columns=variables,
        convert_categoricals=False,
        preserve_dtypes=False,
    )
    unexpected_codes = sorted(
        {
            float(value)
            for variable in variables
            for value in pd.to_numeric(raw[variable], errors="coerce").dropna().unique()
            if float(value) not in {1.0, 2.0, 3.0, 4.0, 5.0, 6.0}
        }
    )
    mapping_rows = [
        row
        for row in _read_csv_rows(COMPONENT_MAP_PATH)
        if row["cohort"] == "KLoSA" and row["deficit_id"] == "underweight"
    ]
    mapping_ok = (
        len(mapping_rows) == 1
        and mapping_rows[0]["source_variable_pattern"] == "r{wave}bmicat"
        and mapping_rows[0]["scoring_rule_id"] == "underweight_bmicat"
        and mapping_rows[0]["semantic_verification_status"] == "verified"
        and mapping_rows[0]["longitudinal_support_status"] == "verified"
    )
    passed = (
        dataset_label == "Gateway Harmonized KLoSA Ver.F"
        and label_stable
        and bmicat_labels == expected_categories
        and not unexpected_codes
        and mapping_ok
    )
    return {
        "passed": passed,
        "dataset_label": dataset_label,
        "source_variables": variables,
        "variable_labels": variable_labels,
        "category_labels": {str(key): value for key, value in bmicat_labels.items()},
        "category_1_definition": bmicat_labels.get(1),
        "generation_logic": "Official rWbmicat category derived from self-reported height and weight; a stable variable label and common category set are present in W1-W9.",
        "unexpected_numeric_codes": unexpected_codes,
        "mapping_verified": mapping_ok,
    }


def underweight_qc(
    outcome: pd.DataFrame,
    spine: pd.DataFrame,
    demographics: pd.DataFrame,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """执行 KLoSA underweight 的逐波和总体 fail-closed QC。"""
    data = outcome.loc[:, [*KEY_COLUMNS, "underweight"]].merge(
        spine.loc[:, [*KEY_COLUMNS, "observed_in_wave"]],
        on=list(KEY_COLUMNS),
        validate="one_to_one",
    )
    data = data.merge(
        demographics,
        on=["person_id", "wave"],
        how="left",
        validate="many_to_one",
    )
    data = data.loc[data["observed_in_wave"].eq(True) & data["wave"].between(1, 9)].copy()
    source_checks = _underweight_source_checks()
    rows: list[dict[str, Any]] = []
    prevalence_by_wave: list[float] = []
    for wave, group in data.groupby("wave", sort=True):
        valid = group["underweight"].notna()
        valid_rate = float(valid.mean())
        prevalence = float(group.loc[valid, "underweight"].mean())
        age_data = group.loc[valid & group["age"].notna(), ["age", "underweight"]]
        rho = (
            float(age_data.corr(method="spearman").iloc[0, 1])
            if len(age_data) > 1 and age_data["underweight"].nunique() > 1
            else np.nan
        )
        prevalence_by_wave.append(prevalence)
        rows.append(
            {
                "Wave": int(wave),
                "Valid %": valid_rate,
                "Underweight %": prevalence,
                "Age pattern": f"rho={rho:.3f} (positive)" if rho > 0 else f"rho={rho:.3f}",
                "Decision": "Pass",
                "valid_n": int(valid.sum()),
                "observed_n": int(len(group)),
                "age_spearman": _finite_float(rho),
            }
        )

    valid_all = data["underweight"].notna() & data["age"].notna()
    age_bands = pd.cut(
        data.loc[valid_all, "age"],
        [-np.inf, 59, 69, 79, np.inf],
        labels=["<60", "60-69", "70-79", "80+"],
    )
    age_band_rates = (
        data.loc[valid_all]
        .assign(age_band=age_bands)
        .groupby("age_band", observed=True)["underweight"]
        .mean()
    )
    age_pattern_ok = bool(
        len(age_band_rates) == 4
        and np.all(np.diff(age_band_rates.to_numpy(dtype=float)) >= 0)
    )
    other_items = [item for item in PRIMARY_ITEMS["KLoSA"] if item != "underweight"]
    correlations = outcome.loc[:, ["underweight", *other_items]].corr(
        method="spearman", min_periods=100
    )["underweight"].drop("underweight").dropna()
    max_item = str(correlations.abs().idxmax())
    max_correlation = float(correlations.loc[max_item])
    max_jump = float(pd.Series(prevalence_by_wave).diff().abs().max())
    criteria = {
        "source_definition_stable": bool(source_checks["passed"]),
        "all_wave_valid_rate_ge_95pct": all(row["Valid %"] >= 0.95 for row in rows),
        "all_wave_prevalence_ge_1pct": all(row["Underweight %"] >= 0.01 for row in rows),
        "max_adjacent_prevalence_jump_le_2pp": max_jump <= 0.02,
        "pooled_age_band_pattern_nondecreasing": age_pattern_ok,
        "no_high_item_redundancy": abs(max_correlation) < 0.80,
    }
    passed = all(criteria.values())
    if not passed:
        for row in rows:
            row["Decision"] = "Stop"
    summary = {
        "decision": "PASS" if passed else "STOP",
        "criteria": criteria,
        "source_checks": source_checks,
        "age_band_prevalence": {
            str(index): float(value) for index, value in age_band_rates.items()
        },
        "maximum_adjacent_wave_prevalence_change": max_jump,
        "maximum_absolute_item_correlation": {
            "item": max_item,
            "spearman": max_correlation,
        },
        "by_wave": [
            {
                "wave": row["Wave"],
                "observed_n": row["observed_n"],
                "valid_n": row["valid_n"],
                "valid_rate": row["Valid %"],
                "underweight_prevalence": row["Underweight %"],
                "age_spearman": row["age_spearman"],
            }
            for row in rows
        ],
    }
    human_rows = [
        {key: row[key] for key in ("Wave", "Valid %", "Underweight %", "Age pattern", "Decision")}
        for row in rows
    ]
    return human_rows, summary


def completeness_qc(
    panels: dict[str, pd.DataFrame],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """比较 80% 与 90% 的逐波 person-wave 和参与者支持。"""
    human: list[dict[str, Any]] = []
    machine: dict[str, Any] = {}
    for cohort in COHORTS:
        panel = panels[cohort]
        start, end = FORMAL_WINDOWS[cohort]
        cohort_rows: list[dict[str, Any]] = []
        for wave in range(start, end + 1):
            group = panel.loc[panel["wave"].eq(wave)]
            observed = group["fi_primary_n_valid"].notna()
            denominator = int(observed.sum())
            measures = {
                "primary_80": int(group.loc[observed, "fi_primary_n_valid"].ge(PRIMARY_QC80_THRESHOLD).sum()),
                "primary_90": int(group.loc[observed, "fi_primary_n_valid"].ge(PRIMARY_QC90_THRESHOLD).sum()),
                "common22_80": int(group.loc[observed, "fi_common22_n_valid"].ge(COMMON22_QC80_THRESHOLD).sum()),
                "common22_90": int(group.loc[observed, "fi_common22_n_valid"].ge(COMMON22_QC90_THRESHOLD).sum()),
            }
            rates = {
                key: (value / denominator if denominator else None)
                for key, value in measures.items()
            }
            human.append(
                {
                    "Cohort": cohort,
                    "Wave": wave,
                    "Observed N": denominator,
                    "Primary >=80%, n (%) [QC reference]": f"{measures['primary_80']:,} ({rates['primary_80']:.2%})",
                    "Primary >=90%, n (%) [formal]": f"{measures['primary_90']:,} ({rates['primary_90']:.2%})",
                    "Common-22 >=80%, n (%) [QC reference]": f"{measures['common22_80']:,} ({rates['common22_80']:.2%})",
                    "Common-22 >=90%, n (%) [formal]": f"{measures['common22_90']:,} ({rates['common22_90']:.2%})",
                }
            )
            cohort_rows.append(
                {
                    "wave": wave,
                    "observed_person_waves": denominator,
                    **{
                        key: {"n": measures[key], "rate": rates[key]}
                        for key in measures
                    },
                }
            )
        participants: dict[str, int] = {}
        for name, column, threshold in (
            ("primary_80", "fi_primary_n_valid", PRIMARY_QC80_THRESHOLD),
            ("primary_90", "fi_primary_n_valid", PRIMARY_QC90_THRESHOLD),
            ("common22_80", "fi_common22_n_valid", COMMON22_QC80_THRESHOLD),
            ("common22_90", "fi_common22_n_valid", COMMON22_QC90_THRESHOLD),
        ):
            participants[name] = int(
                panel.loc[panel[column].ge(threshold).fillna(False), "person_id"].nunique()
            )
        machine[cohort] = {"by_wave": cohort_rows, "retained_participants": participants}
    return human, machine


def _cohort_construct_diagnostics(
    panel: pd.DataFrame,
    demographics: pd.DataFrame,
) -> dict[str, Any]:
    """计算年龄、性别和整体 primary/common-22 一致性诊断。"""
    data = panel.merge(
        demographics,
        on=["person_id", "wave"],
        how="left",
        validate="many_to_one",
    )
    valid_age = data.dropna(subset=["fi_primary", "age"])
    age_spearman = valid_age[["age", "fi_primary"]].corr(method="spearman").iloc[0, 1]
    age_bands = pd.cut(
        valid_age["age"],
        [-np.inf, 59, 69, 79, np.inf],
        labels=["<60", "60-69", "70-79", "80+"],
    )
    age_means = (
        valid_age.assign(age_band=age_bands)
        .groupby("age_band", observed=True)["fi_primary"]
        .mean()
    )
    valid_sex = data.dropna(subset=["fi_primary", "female"])
    female_mean = valid_sex.loc[valid_sex["female"].eq(True), "fi_primary"].mean()
    male_mean = valid_sex.loc[valid_sex["female"].eq(False), "fi_primary"].mean()
    paired = data.dropna(subset=["fi_primary", "fi_common22"])
    return {
        "age_spearman": _finite_float(age_spearman),
        "age_band_primary_means": {
            str(index): float(value) for index, value in age_means.items()
        },
        "female_primary_mean": _finite_float(female_mean),
        "male_primary_mean": _finite_float(male_mean),
        "female_minus_male": _finite_float(female_mean - male_mean),
        "overall_primary_common22_pearson": _finite_float(
            paired[["fi_primary", "fi_common22"]].corr(method="pearson").iloc[0, 1]
        ),
        "overall_primary_common22_spearman": _finite_float(
            paired[["fi_primary", "fi_common22"]].corr(method="spearman").iloc[0, 1]
        ),
        "overall_primary_minus_common22_mean": _finite_float(
            (paired["fi_primary"] - paired["fi_common22"]).mean()
        ),
    }


def fi_distribution_qc(
    panels: dict[str, pd.DataFrame],
    demographics: dict[str, pd.DataFrame],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """生成精简人类表和详细机器级 FI measurement QC。"""
    human: list[dict[str, Any]] = []
    machine: dict[str, Any] = {}
    for cohort in COHORTS:
        records: list[dict[str, Any]] = []
        panel = panels[cohort]
        start, end = FORMAL_WINDOWS[cohort]
        for wave in range(start, end + 1):
            group = panel.loc[panel["wave"].eq(wave)]
            primary = group["fi_primary"].dropna()
            paired = group.dropna(subset=["fi_primary", "fi_common22"])
            pearson = paired[["fi_primary", "fi_common22"]].corr(method="pearson").iloc[0, 1]
            spearman = paired[["fi_primary", "fi_common22"]].corr(method="spearman").iloc[0, 1]
            p1 = primary.quantile(0.01)
            p99 = primary.quantile(0.99)
            record = {
                "wave": wave,
                "primary_n": int(len(primary)),
                "mean": _finite_float(primary.mean()),
                "sd": _finite_float(primary.std()),
                "median": _finite_float(primary.median()),
                "iqr": _finite_float(primary.quantile(0.75) - primary.quantile(0.25)),
                "p1": _finite_float(p1),
                "p99": _finite_float(p99),
                "min": _finite_float(primary.min()),
                "max": _finite_float(primary.max()),
                "skewness": _finite_float(primary.skew()),
                "paired_n": int(len(paired)),
                "primary_common22_pearson": _finite_float(pearson),
                "primary_common22_spearman": _finite_float(spearman),
                "primary_minus_common22_mean": _finite_float(
                    (paired["fi_primary"] - paired["fi_common22"]).mean()
                ),
                "review_flags": [],
            }
            if record["p99"] is not None and record["p99"] > 0.70:
                record["review_flags"].append("P99 > 0.70")
            if record["primary_common22_pearson"] is not None and record["primary_common22_pearson"] < 0.90:
                record["review_flags"].append("Pearson correlation < 0.90")
            if record["skewness"] is not None and record["skewness"] <= 0:
                record["review_flags"].append("Distribution not right-skewed")
            records.append(record)
        for previous, current in zip(records, records[1:]):
            if (
                previous["mean"] is not None
                and current["mean"] is not None
                and abs(current["mean"] - previous["mean"]) > 0.03
            ):
                current["review_flags"].append("Adjacent-wave mean change > 0.03")
        for record in records:
            human.append(
                {
                    "Cohort": cohort,
                    "Wave": record["wave"],
                    "Primary N": record["primary_n"],
                    "Mean": record["mean"],
                    "SD": record["sd"],
                    "Median": record["median"],
                    "P99": record["p99"],
                    "Correlation with common-22": record["primary_common22_pearson"],
                    "Review flag": "; ".join(record["review_flags"]),
                }
            )
        machine[cohort] = {
            "by_wave": records,
            "construct_diagnostics": _cohort_construct_diagnostics(
                panel, demographics[cohort]
            ),
        }
    return human, machine


CODED_MISSING_VALUES = {
    "ELSA": {
        -22: "spouse_nonresponse",
        -21: "unmarried",
        -18: "refuse",
        -16: "proxy",
        -13: "missing",
        -4: "dont_know",
        -3: "no_self_completion",
    }
}


def _top_tail_qc(
    panels: dict[str, pd.DataFrame],
    outcomes: dict[str, pd.DataFrame],
    spines: dict[str, pd.DataFrame],
    build_inputs: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """全量复核按历史 80% 规则可计算且 FI >=0.70 的 person-waves。"""
    registry = {row["cohort"]: row for row in _read_csv_rows(REGISTRY_PATH)}
    output: dict[str, Any] = {
        "reference_definition": {
            "primary_minimum_valid": V1_SPEC.primary_threshold,
            "fi_cutoff": 0.70,
            "reason": "Audit the full historical v1 upper tail, including rows excluded by the formal v2 threshold.",
        },
        "person_level_records_persisted": False,
        "cohorts": {},
    }

    for cohort in COHORTS:
        panel = panels[cohort]
        outcome = outcomes[cohort]
        spine = spines[cohort]
        primary_items = list(PRIMARY_ITEMS[cohort])
        if len(primary_items) != 30 or len(set(primary_items)) != 30:
            raise RuntimeError(f"{cohort}: top-tail audit 发现 primary 条目重复或数量错误")
        if len(COMMON_22) != 22 or len(set(COMMON_22)) != 22:
            raise RuntimeError("top-tail audit 发现 common-22 条目重复或数量错误")

        start, end = FORMAL_WINDOWS[cohort]
        opportunity = (
            spine["observed_in_wave"].eq(True).fillna(False)
            & spine["wave"].between(start, end)
        )
        reference_n = outcome[primary_items].notna().sum(axis=1)
        reference_fi = outcome[primary_items].sum(axis=1, skipna=True).div(reference_n)
        tail = opportunity & reference_n.ge(V1_SPEC.primary_threshold) & reference_fi.ge(0.70)
        tail_indices = panel.index[tail]
        tail_rows = panel.loc[tail_indices]
        if tail_rows.empty:
            raise RuntimeError(f"{cohort}: top-tail audit 没有可复核记录")

        raw_path = resolve_cohort_dataset_path(cohort)
        outcome_manifest = json.loads(
            _repo_path(
                build_inputs[cohort]["outcome_components"]["manifest_path"]
            ).read_text(encoding="utf-8")
        )
        current_source_hash = sha256_file(raw_path)
        if current_source_hash != outcome_manifest["source_hash"]:
            raise RuntimeError(f"{cohort}: top-tail audit raw source 哈希与 Outcome v2 不一致")

        raw = read_raw_source(raw_path, cohort)
        rebuilt, _diagnostics = build_components_from_raw(raw, spine, cohort)
        rebuilt_differences = 0
        for item in primary_items:
            rebuilt_values = rebuilt.loc[tail_indices, item].to_numpy(
                dtype=float, na_value=np.nan
            )
            frozen_values = outcome.loc[tail_indices, item].to_numpy(
                dtype=float, na_value=np.nan
            )
            rebuilt_differences += int(
                (~np.isclose(
                    rebuilt_values,
                    frozen_values,
                    rtol=0,
                    atol=1e-12,
                    equal_nan=True,
                )).sum()
            )

        id_source = registry[cohort]["person_id_source"]
        raw_work = raw.copy()
        raw_work["_person_id"] = normalise_person_id(raw_work[id_source], cohort)
        raw_work = raw_work.set_index("_person_id")
        specs = {spec.item_id: spec for spec in COMPONENT_SPECS[cohort]}
        tag_counts: Counter[str] = Counter()
        coded_missing_counts: Counter[str] = Counter()
        unrecognised_numeric: Counter[str] = Counter()
        ordinal_pairs: dict[str, dict[float, set[float]]] = defaultdict(
            lambda: defaultdict(set)
        )
        tagged_scored_nonmissing = 0
        direct_raw_differences = 0

        for wave, wave_indices in tail_rows.groupby("wave").groups.items():
            person_ids = tail_rows.loc[wave_indices, "person_id"].astype("string")
            for item in primary_items:
                spec = specs[item]
                source = spec.source_for_wave(int(wave))
                if source is None:
                    raise RuntimeError(
                        f"{cohort}: top-tail audit 缺少 {item} W{wave} 来源"
                    )
                raw_values = raw_work[source].reindex(person_ids.to_numpy()).reset_index(
                    drop=True
                )
                scored, tags, invalid_numeric, numeric = score_series(
                    raw_values, spec.rule
                )
                stored = outcome.loc[wave_indices, item].reset_index(drop=True)
                direct_raw_differences += int(
                    (~np.isclose(
                        scored.to_numpy(dtype=float, na_value=np.nan),
                        stored.to_numpy(dtype=float, na_value=np.nan),
                        rtol=0,
                        atol=1e-12,
                        equal_nan=True,
                    )).sum()
                )
                tagged_scored_nonmissing += int((tags.notna() & scored.notna()).sum())
                for tag, count in tags.dropna().astype(str).value_counts().items():
                    tag_counts[str(tag)] += int(count)

                for value, count in numeric.loc[invalid_numeric].value_counts().items():
                    integer_value = int(value)
                    category = CODED_MISSING_VALUES.get(cohort, {}).get(integer_value)
                    if category is None:
                        unrecognised_numeric[f"{item}|W{int(wave)}|{value}"] += int(count)
                    else:
                        coded_missing_counts[category] += int(count)

                if spec.rule.startswith("ordinal_"):
                    valid = numeric.notna() & scored.notna()
                    key = f"{item}|{spec.rule}"
                    for raw_value, score in zip(
                        numeric.loc[valid], scored.loc[valid], strict=True
                    ):
                        ordinal_pairs[key][float(raw_value)].add(float(score))

        ordinal_direction_ok = True
        ordinal_mappings: dict[str, dict[str, list[float]]] = {}
        for key, pairs in ordinal_pairs.items():
            ordinal_mappings[key] = {
                str(raw_value): sorted(scores)
                for raw_value, scores in sorted(pairs.items())
            }
            ordered_scores = [next(iter(pairs[value])) for value in sorted(pairs)]
            rule = key.split("|", maxsplit=1)[1]
            if rule == "ordinal_poor_to_excellent":
                ordinal_direction_ok &= all(
                    left >= right
                    for left, right in zip(
                        ordered_scores, ordered_scores[1:], strict=False
                    )
                )
            else:
                ordinal_direction_ok &= all(
                    left <= right
                    for left, right in zip(
                        ordered_scores, ordered_scores[1:], strict=False
                    )
                )

        formula_difference = float(
            (
                reference_fi.loc[tail_indices].astype(float)
                - outcome.loc[tail_indices, primary_items]
                .sum(axis=1, skipna=True)
                .div(reference_n.loc[tail_indices])
                .astype(float)
            )
            .abs()
            .max()
        )
        v2_retained = reference_n.loc[tail_indices].ge(V2_SPEC.primary_threshold)
        retained_score_state_ok = panel.loc[tail_indices, "fi_primary"].notna().equals(
            v2_retained.astype(bool)
        )
        n_valid = reference_n.loc[tail_indices].astype(int)
        observed_by_wave = spine.loc[opportunity].groupby("wave").size()
        tail_by_wave = tail_rows.groupby("wave").size()
        retained_by_wave = tail_rows.loc[v2_retained.to_numpy()].groupby("wave").size()

        hard_failures = {
            "raw_rebuild_component_cell_differences": rebuilt_differences,
            "direct_raw_score_cell_differences": direct_raw_differences,
            "tagged_cells_scored_nonmissing": tagged_scored_nonmissing,
            "unrecognised_numeric_code_count": int(sum(unrecognised_numeric.values())),
            "formula_max_abs_difference": formula_difference,
            "ordinal_direction_ok": bool(ordinal_direction_ok),
            "v2_retained_score_state_ok": bool(retained_score_state_ok),
        }
        if (
            rebuilt_differences
            or direct_raw_differences
            or tagged_scored_nonmissing
            or unrecognised_numeric
            or formula_difference > 1e-12
            or not ordinal_direction_ok
            or not retained_score_state_ok
        ):
            raise RuntimeError(
                f"{cohort}: top-tail audit 未通过：{json.dumps(hard_failures, ensure_ascii=False)}"
            )

        output["cohorts"][cohort] = {
            "decision": "PASS",
            "raw_source_qc_input": {
                "sha256": current_source_hash,
                "matches_frozen_outcome_source": True,
            },
            "tail_n": int(len(tail_rows)),
            "tail_rate_among_formal_interviews": float(
                len(tail_rows) / int(opportunity.sum())
            ),
            "v2_retained_tail_n": int(v2_retained.sum()),
            "n_valid_summary": {
                "min": int(n_valid.min()),
                "p1": float(n_valid.quantile(0.01)),
                "median": float(n_valid.median()),
                "max": int(n_valid.max()),
            },
            "by_wave": [
                {
                    "wave": int(wave),
                    "formal_interview_n": int(observed_by_wave.loc[wave]),
                    "tail_n": int(tail_by_wave.get(wave, 0)),
                    "tail_rate": float(
                        tail_by_wave.get(wave, 0) / observed_by_wave.loc[wave]
                    ),
                    "v2_retained_tail_n": int(retained_by_wave.get(wave, 0)),
                }
                for wave in observed_by_wave.index
            ],
            "tagged_missing_counts": dict(sorted(tag_counts.items())),
            "coded_missing_counts": dict(sorted(coded_missing_counts.items())),
            "unrecognised_numeric_codes": dict(sorted(unrecognised_numeric.items())),
            "ordinal_mappings_observed": ordinal_mappings,
            "item_contract": {
                "primary_count": len(primary_items),
                "primary_unique_count": len(set(primary_items)),
                "common22_count": len(COMMON_22),
                "common22_unique_count": len(set(COMMON_22)),
            },
            **hard_failures,
        }
    return output


def _validate_construct_qc(fi_qc: dict[str, Any]) -> None:
    """冻结前确认预设 construct 方向完整且未反转。"""
    for cohort, payload in fi_qc.items():
        construct = payload["construct_diagnostics"]
        if (
            construct["age_spearman"] is None
            or construct["age_spearman"] <= 0
            or construct["female_minus_male"] is None
            or construct["female_minus_male"] <= 0
        ):
            raise RuntimeError(f"{cohort}: age/sex construct diagnostic 未通过")
        if any(
            row["skewness"] is None or row["skewness"] <= 0
            for row in payload["by_wave"]
        ):
            raise RuntimeError(f"{cohort}: FI 逐波分布未保持右偏")


def _lookup_role_available(
    ledger: pd.DataFrame,
    available: pd.DataFrame,
    wave_column: str,
) -> pd.Series:
    """把某一候选时间角色映射到 FI measurement availability。"""
    left = ledger.loc[:, ["person_id", wave_column]].rename(columns={wave_column: "wave"})
    merged = left.merge(
        available,
        on=["person_id", "wave"],
        how="left",
        validate="many_to_one",
    )
    return merged["available"].fillna(False).astype(bool)


def potential_role_support_qc(
    panels: dict[str, pd.DataFrame],
    *,
    load_candidate_intervals: Callable[[str], tuple[pd.DataFrame, Path, dict[str, Any]]],
    load_eligibility_base: Callable[[str], tuple[pd.DataFrame, Path, dict[str, Any]]],
) -> dict[str, Any]:
    """独立评估潜在 t/t+1/t+2+ 支持，不构成 FI build input。"""
    output: dict[str, Any] = {
        "interpretation": "Potential measurement-role support among structurally eligible candidate intervals; no shock, landmark, recovery group, or final sample is identified.",
        "cohorts": {},
    }
    for cohort in COHORTS:
        candidate, candidate_path, candidate_manifest = load_candidate_intervals(cohort)
        eligibility, eligibility_path, eligibility_manifest = load_eligibility_base(cohort)
        ledger = candidate.merge(
            eligibility.loc[:, ["cohort", "person_id", "candidate_shock_wave", "structural_eligibility_status"]],
            on=["cohort", "person_id", "candidate_shock_wave"],
            validate="one_to_one",
        )
        ledger = ledger.loc[ledger["structural_eligibility_status"].eq("pass")].reset_index(drop=True)
        denominator = int(len(ledger))
        cohort_result: dict[str, Any] = {
            "candidate_intervals_qc_input": {
                "path": _relative(candidate_path),
                "sha256": candidate_manifest["output"]["sha256"],
            },
            "eligibility_base_qc_input": {
                "path": _relative(eligibility_path),
                "sha256": eligibility_manifest["output"]["sha256"],
            },
            "denominator": denominator,
            "denominator_definition": "Eligibility Base v2 intervals with structural_eligibility_status=pass.",
            "definitions": {},
        }
        for definition, column, thresholds in (
            (
                "primary",
                "fi_primary_n_valid",
                {
                    "80pct": PRIMARY_QC80_THRESHOLD,
                    "90pct": PRIMARY_QC90_THRESHOLD,
                },
            ),
            (
                "common22",
                "fi_common22_n_valid",
                {
                    "80pct": COMMON22_QC80_THRESHOLD,
                    "90pct": COMMON22_QC90_THRESHOLD,
                },
            ),
        ):
            cohort_result["definitions"][definition] = {}
            for label, threshold in thresholds.items():
                available = panels[cohort].loc[:, ["person_id", "wave", column]].copy()
                available["wave"] = available["wave"].astype("Int16")
                available["available"] = available[column].ge(threshold).fillna(False)
                available = available.loc[:, ["person_id", "wave", "available"]]
                role_t = _lookup_role_available(ledger, available, "candidate_shock_wave")
                role_t1 = _lookup_role_available(ledger, available, "tplus1_wave")
                role_t2 = _lookup_role_available(ledger, available, "tplus2_wave")
                max_complete_wave = (
                    available.loc[available["available"]]
                    .groupby("person_id", sort=False)["wave"]
                    .max()
                )
                person_max = ledger["person_id"].map(max_complete_wave)
                any_t2_later = (
                    person_max.notna()
                    & person_max.ge(ledger["candidate_shock_wave"].astype(float) + 2)
                )
                measures = {
                    "potential_t": role_t,
                    "potential_tplus1": role_t1,
                    "exact_tplus2": role_t2,
                    "tplus2_or_later": any_t2_later,
                    "tplus1_and_exact_tplus2": role_t1 & role_t2,
                    "tplus1_and_any_tplus2_or_later": role_t1 & any_t2_later,
                }
                cohort_result["definitions"][definition][label] = {
                    name: {
                        "n": int(mask.sum()),
                        "rate": None if denominator == 0 else float(mask.sum() / denominator),
                    }
                    for name, mask in measures.items()
                }
        output["cohorts"][cohort] = cohort_result
    return output


def _schema(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return [
        {
            "name": column,
            "dtype": str(frame[column].dtype),
            "nullable": bool(frame[column].isna().any()),
        }
        for column in frame.columns
    ]


def _git_output(arguments: list[str], *, binary: bool = False) -> bytes | str:
    completed = subprocess.run(
        ["git", *arguments], cwd=ROOT, check=True, capture_output=True
    )
    if binary:
        return completed.stdout
    return completed.stdout.decode("utf-8", errors="replace").strip()


def git_provenance() -> dict[str, Any]:
    """记录 dirty worktree 下与构建有关的代码指纹。"""
    commit = str(_git_output(["rev-parse", "HEAD"]))
    status = str(_git_output(["status", "--porcelain=v1"]))
    diff = bytes(_git_output(["diff", "--binary", "HEAD"], binary=True))
    untracked = str(
        _git_output(
            [
                "ls-files",
                "--others",
                "--exclude-standard",
                "--",
                "AGENTS.md",
                "README.md",
                "00_admin",
                "02_code/01_panel_build",
            ]
        )
    )
    inventory: list[str] = []
    for relative in sorted(line for line in untracked.splitlines() if line):
        if relative.replace("\\", "/").startswith("00_admin/panel_manifests/"):
            continue
        path = ROOT / relative
        if path.is_file() and path.suffix.lower() != ".pdf":
            inventory.append(f"{relative}\t{sha256_file(path)}")
    digest = hashlib.sha256()
    digest.update(diff)
    digest.update("\n".join(inventory).encode("utf-8"))
    return {
        "git_commit": commit,
        "git_dirty": bool(status),
        "git_diff_sha256": digest.hexdigest(),
        "untracked_build_file_inventory": inventory,
    }


def _workbook_payload(
    underweight_rows: list[dict[str, Any]],
    completeness_rows: list[dict[str, Any]],
    fi_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "underweight": underweight_rows,
        "completeness": completeness_rows,
        "fi_qc": fi_rows,
    }


WORKBOOK_BUILDER = r'''
import fs from "node:fs/promises";
import { FileBlob, SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const [payloadPath, outputPath, previewDir] = process.argv.slice(2);
const payload = JSON.parse(await fs.readFile(payloadPath, "utf8"));
const workbook = Workbook.create();
const fontFamily = "Arial";

function addTableSheet(name, rows, headers, tableName, widths, notes) {
  const sheet = workbook.worksheets.add(name);
  const matrix = [headers, ...rows.map((row) => headers.map((header) => row[header] ?? null))];
  const range = sheet.getRangeByIndexes(0, 0, matrix.length, headers.length);
  range.values = matrix;
  range.format.font = { name: fontFamily, size: 10, color: "#1F2937" };
  range.format.verticalAlignment = "top";
  range.format.wrapText = true;
  const header = sheet.getRangeByIndexes(0, 0, 1, headers.length);
  header.format = {
    fill: "#1F4E78",
    font: { name: fontFamily, size: 10, bold: true, color: "#FFFFFF" },
    horizontalAlignment: "center",
    verticalAlignment: "center",
    wrapText: true,
  };
  header.format.rowHeightPx = 38;
  for (let index = 0; index < widths.length; index += 1) {
    sheet.getRangeByIndexes(0, index, matrix.length, 1).format.columnWidthPx = widths[index];
  }
  const table = sheet.tables.add(range, true, tableName);
  table.style = "TableStyleMedium2";
  table.showFilterButton = true;
  table.showBandedRows = true;
  sheet.freezePanes.freezeRows(1);
  sheet.showGridLines = false;
  if (matrix.length > 1) {
    sheet.getRangeByIndexes(1, 0, matrix.length - 1, headers.length).format.autofitRows();
  }
  let noteRow = matrix.length + 1;
  for (const note of notes) {
    const noteRange = sheet.getRangeByIndexes(noteRow, 0, 1, headers.length);
    noteRange.merge();
    noteRange.values = [[note]];
    noteRange.format = {
      font: { name: fontFamily, size: 9, italic: true, color: "#4B5563" },
      verticalAlignment: "top",
      wrapText: true,
    };
    noteRange.format.rowHeightPx = 34;
    noteRow += 1;
  }
  return { sheet, rows: matrix.length, columns: headers.length, renderRows: noteRow };
}

const underweightHeaders = ["Wave", "Valid %", "Underweight %", "Age pattern", "Decision"];
const completenessHeaders = ["Cohort", "Wave", "Observed N", "Primary >=80%, n (%) [QC reference]", "Primary >=90%, n (%) [formal]", "Common-22 >=80%, n (%) [QC reference]", "Common-22 >=90%, n (%) [formal]"];
const fiHeaders = ["Cohort", "Wave", "Primary N", "Mean", "SD", "Median", "P99", "Correlation with common-22", "Review flag"];

const underweight = addTableSheet(
  "1_Underweight_QC",
  payload.underweight,
  underweightHeaders,
  "FIUnderweightQC",
  [72, 100, 120, 190, 90],
  ["Note: KLoSA only. Percentages use interviewed KLoSA person-waves in W1-W9. Age pattern is the within-wave Spearman correlation between age and underweight."],
);
const completeness = addTableSheet(
  "2_Completeness",
  payload.completeness,
  completenessHeaders,
  "FICompleteness",
  [90, 65, 105, 200, 185, 220, 205],
  ["Note: Percentages use interviewed person-waves in each cohort's formal FI window. The 90% columns are the formal v2 rule; 80% is retained only as a QC reference."],
);
const fiQc = addTableSheet(
  "3_FI_QC",
  payload.fi_qc,
  fiHeaders,
  "FIMeasurementQC",
  [90, 65, 105, 90, 90, 90, 90, 190, 210],
  [
    "Note: FI summaries use person-waves meeting the formal 90% completeness rule. The full FI >=0.70 coding audit passed; P99 >0.70 remains a diagnostic flag.",
    "Caution: Absolute FI levels are not direct country-health comparisons because measurement severity and cohort-specific items differ.",
  ],
);

completeness.sheet.getRangeByIndexes(0, 4, 1, 1).format.fill = "#2F855A";
completeness.sheet.getRangeByIndexes(0, 6, 1, 1).format.fill = "#2F855A";

underweight.sheet.getRangeByIndexes(1, 1, underweight.rows - 1, 2).format.numberFormat = "0.00%";
completeness.sheet.getRangeByIndexes(1, 2, completeness.rows - 1, 1).format.numberFormat = "#,##0";
fiQc.sheet.getRangeByIndexes(1, 2, fiQc.rows - 1, 1).format.numberFormat = "#,##0";
fiQc.sheet.getRangeByIndexes(1, 3, fiQc.rows - 1, 5).format.numberFormat = "0.000";

const xlsx = await SpreadsheetFile.exportXlsx(workbook);
await xlsx.save(outputPath);
const saved = await SpreadsheetFile.importXlsx(await FileBlob.load(outputPath));

for (const target of [underweight, completeness, fiQc]) {
  const savedSheet = saved.worksheets.getItem(target.sheet.name);
  const lastColumn = String.fromCharCode(64 + target.columns);
  const inspected = await saved.inspect({
    kind: "table",
    range: `${target.sheet.name}!A1:${lastColumn}${target.rows}`,
    include: "values,formulas",
    tableMaxRows: 8,
    tableMaxCols: target.columns,
    maxChars: 6000,
  });
  console.log(inspected.ndjson);
  if (!savedSheet) throw new Error(`Missing saved sheet: ${target.sheet.name}`);
}

const errors = await saved.inspect({
  kind: "match",
  searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!|#SPILL!|#CALC!",
  options: { useRegex: true, maxResults: 100 },
  summary: "final formula error scan",
});
console.log(errors.ndjson);

await fs.mkdir(previewDir, { recursive: true });
for (const target of [underweight, completeness, fiQc]) {
  const lastColumn = String.fromCharCode(64 + target.columns);
  const preview = await saved.render({
    sheetName: target.sheet.name,
    range: `A1:${lastColumn}${target.renderRows}`,
    scale: 1,
    format: "png",
  });
  await fs.writeFile(
    `${previewDir}/${target.sheet.name}.png`,
    new Uint8Array(await preview.arrayBuffer()),
  );
}
'''


def _runtime_paths() -> tuple[Path, Path]:
    """定位 load_workspace_dependencies 提供的 Node runtime。"""
    dependency_root = (
        Path(os.environ.get("USERPROFILE", str(Path.home())))
        / ".cache"
        / "codex-runtimes"
        / "codex-primary-runtime"
        / "dependencies"
    )
    node = dependency_root / "node" / "bin" / "node.exe"
    node_modules = dependency_root / "node" / "node_modules"
    if not node.is_file() or not node_modules.is_dir():
        raise FileNotFoundError("找不到 load_workspace_dependencies 提供的 Node runtime")
    return node, node_modules


def _create_node_modules_link(link: Path, target: Path) -> None:
    """在系统临时目录建立 artifact-tool 模块链接。"""
    try:
        os.symlink(target, link, target_is_directory=True)
        return
    except OSError:
        pass
    completed = subprocess.run(
        ["cmd.exe", "/d", "/c", "mklink", "/J", str(link), str(target)],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "创建临时 node_modules junction 失败："
            + (completed.stdout or "")
            + (completed.stderr or "")
        )


def create_review_workbook(
    underweight_rows: list[dict[str, Any]],
    completeness_rows: list[dict[str, Any]],
    fi_rows: list[dict[str, Any]],
    output_path: Path,
    preview_dir: Path,
) -> None:
    """用 artifact-tool 生成并重新载入检查三表 QC workbook。"""
    node, node_modules = _runtime_paths()
    with tempfile.TemporaryDirectory(prefix="fi_panel_xlsx_") as temporary_name:
        temporary = Path(temporary_name)
        payload_path = temporary / "payload.json"
        builder_path = temporary / "build_fi_review.mjs"
        payload_path.write_text(
            json.dumps(
                _workbook_payload(underweight_rows, completeness_rows, fi_rows),
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        builder_path.write_text(WORKBOOK_BUILDER, encoding="utf-8")
        _create_node_modules_link(temporary / "node_modules", node_modules)
        preview_dir.mkdir(parents=True, exist_ok=True)
        completed = subprocess.run(
            [
                str(node),
                str(builder_path),
                str(payload_path),
                str(output_path),
                str(preview_dir),
            ],
            cwd=temporary,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        if completed.returncode != 0:
            raise RuntimeError(
                "artifact-tool FI workbook 构建失败：\n"
                + (completed.stdout or "")
                + "\n"
                + (completed.stderr or "")
            )
        if completed.stdout:
            print(
                completed.stdout.encode("ascii", errors="backslashreplace").decode("ascii")
            )
        if not output_path.is_file():
            raise RuntimeError("artifact-tool 未生成 FI QC workbook")


DICTIONARY_TEXT = """# FI Panel v2 数据字典

## 数据粒度与主键

每行表示一个队列中的一个 person-wave。主键为 `cohort + person_id + wave`，行集合及顺序与冻结 Full Spine 完全一致。

## 字段

- `fi_primary`：队列特异 30 项 deficit-accumulation FI。仅在正式窗口、实际访谈且至少 27 项有效时计算。
- `fi_primary_n_valid`：primary 30 项中实际进入该 person-wave 分母的有效条目数。
- `fi_common22`：原 common-22 measurement-sensitivity FI。仅在同一正式窗口、实际访谈且至少 20 项有效时计算。
- `fi_common22_n_valid`：common-22 中实际进入分母的有效条目数。

两个 FI 均等于有效缺陷分值之和除以有效条目数，范围为 0 到 1，数值越高表示缺陷累积越多。组件缺失不会编码为零。

## NA 与完整性

正式窗口为 HRS W3-W16、ELSA W1-W10、KLoSA W1-W9。窗口外或 `observed_in_wave` 不为 TRUE 时，四个 FI/n_valid 字段均为 NA，表示没有正式测量机会。窗口内实际访谈但条目不足时保留 `n_valid`，FI 为 NA。

物理表不保存 `*_complete`。下游可确定性派生：`fi_primary_n_valid >= 27` 和 `fi_common22_n_valid >= 20`。80%（24/30 与 18/22）只保留为 measurement-support QC reference，不生成替代 FI panel。

三队列使用不同严重度和队列特异条目，因此 FI 的绝对水平不得直接解释为国家间健康差异。正式分析在各队列内独立进行，common-22 用于检查条目组成敏感性。本层不是最终资格、landmark 样本或模型结果。
"""


def _compact_snapshot(snapshot: dict[str, Any]) -> dict[str, Any]:
    """把 pre-panel 快照压缩为路径、哈希、维度和 manifest 哈希。"""
    output: dict[str, Any] = {}
    for panel_name, cohorts in snapshot.items():
        output[panel_name] = {}
        for cohort, record in cohorts.items():
            path = Path(record["path"])
            manifest_path = Path(record["manifest_path"])
            output[panel_name][cohort] = {
                "path": _relative(path),
                "sha256": record["sha256"],
                "manifest_path": _relative(manifest_path),
                "manifest_sha256": record["manifest_sha256"],
                "row_count": record["row_count"],
                "column_count": record["column_count"],
            }
    return output


def _build_manifest(
    cohort: str,
    frame: pd.DataFrame,
    output_path: Path,
    build_inputs: dict[str, Any],
    completeness: dict[str, Any],
    fi_qc: dict[str, Any],
    top_tail_qc: dict[str, Any],
    underweight: dict[str, Any] | None,
    release_spec: FIReleaseSpec = ACTIVE_SPEC,
) -> dict[str, Any]:
    """生成每队列正式 FI panel manifest。"""
    return {
        "manifest_schema_version": "fi_panel_manifest_v2",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "stage": "fi-panel",
        "cohort": cohort,
        "runner_version": RUNNER_VERSION,
        "builder_version": release_spec.builder_version,
        "spec_version": release_spec.spec_version,
        "specification_sha256": specification_hash(release_spec),
        "mapping_sha256": sha256_file(COMPONENT_MAP_PATH),
        "design_register_sha256": sha256_file(DESIGN_REGISTER_PATH),
        **git_provenance(),
        "build_inputs": build_inputs,
        "formal_window": {
            "wave_start": FORMAL_WINDOWS[cohort][0],
            "wave_end": FORMAL_WINDOWS[cohort][1],
        },
        "item_sets": {
            "primary": list(PRIMARY_ITEMS[cohort]),
            "primary_count": len(PRIMARY_ITEMS[cohort]),
            "common22": list(COMMON_22),
            "common22_count": len(COMMON_22),
        },
        "thresholds": {
            "primary_minimum_valid": release_spec.primary_threshold,
            "common22_minimum_valid": release_spec.common22_threshold,
            "primary_80pct_qc_reference": PRIMARY_QC80_THRESHOLD,
            "common22_80pct_qc_reference": COMMON22_QC80_THRESHOLD,
        },
        "schema": _schema(frame),
        "row_count": int(len(frame)),
        "key": list(KEY_COLUMNS),
        "unique_key_check": not frame.duplicated(list(KEY_COLUMNS)).any(),
        "output": {
            "path": _relative(_final_output_path(cohort)),
            "sha256": sha256_file(output_path),
        },
        "completeness_qc": completeness,
        "fi_measurement_qc": fi_qc,
        "top_tail_qc": top_tail_qc,
        "underweight_qc": underweight,
        "invariants": {
            "full_spine_rows_and_order_preserved": True,
            "candidate_or_eligibility_filter_applied": False,
            "missing_components_scored_as_zero": False,
            "physical_complete_flags_stored": False,
            "wealth_or_recovery_read": False,
            "association_model_run": False,
        },
    }


def _build_freeze(
    staged_outputs: dict[str, Path],
    staged_manifests: dict[str, Path],
    staged_review: Path,
    staged_dictionary: Path,
    build_inputs: dict[str, dict[str, Any]],
    pre_snapshot: dict[str, Any],
    role_qc: dict[str, Any],
    underweight: dict[str, Any],
    top_tail_qc: dict[str, Any],
    release_spec: FIReleaseSpec = ACTIVE_SPEC,
) -> dict[str, Any]:
    """生成唯一 FI panel v2 freeze record。"""
    return {
        "freeze_record_schema_version": "fi_panel_freeze_record_v2",
        "freeze_status": "FROZEN",
        "frozen_at_utc": datetime.now(UTC).isoformat(),
        "runner_version": RUNNER_VERSION,
        "builder_version": release_spec.builder_version,
        "specification": fi_specification(release_spec),
        "specification_sha256": specification_hash(release_spec),
        "supersedes": {
            "scope": "Completeness threshold only; FI items, scoring, windows, and seven-column schema are unchanged.",
            "fi_panel_v1_freeze_path": _relative(V1_FREEZE_PATH),
            "fi_panel_v1_freeze_sha256": sha256_file(V1_FREEZE_PATH),
        },
        "design_register": {
            "path": _relative(DESIGN_REGISTER_PATH),
            "sha256": sha256_file(DESIGN_REGISTER_PATH),
        },
        "outcome_components_freeze": {
            "path": _relative(OUTCOME_FREEZE_PATH),
            "sha256": sha256_file(OUTCOME_FREEZE_PATH),
            "map_version": map_version(),
        },
        "build_inputs": build_inputs,
        "prepanel_snapshot": _compact_snapshot(pre_snapshot),
        "underweight_qc": underweight,
        "qc_context": {
            "candidate_and_eligibility_are_not_build_inputs": True,
            "potential_measurement_role_support": role_qc,
            "top_tail_audit_is_not_a_build_input": True,
            "top_tail_audit": top_tail_qc,
        },
        "cohorts": {
            cohort: {
                "parquet_path": _relative(_final_output_path(cohort)),
                "parquet_sha256": sha256_file(staged_outputs[cohort]),
                "manifest_path": _relative(_manifest_path(cohort)),
                "manifest_sha256": sha256_file(staged_manifests[cohort]),
                "row_count": int(pd.read_parquet(staged_outputs[cohort], columns=["wave"]).shape[0]),
            }
            for cohort in COHORTS
        },
        "review_workbook": {
            "path": _relative(REVIEW_PATH),
            "sha256": sha256_file(staged_review),
            "sheets": ["1_Underweight_QC", "2_Completeness", "3_FI_QC"],
        },
        "data_dictionary": {
            "path": _relative(DICTIONARY_PATH),
            "sha256": sha256_file(staged_dictionary),
        },
        "overwrite_policy": "Do not overwrite FI Panel v1, FI Panel v2, or either freeze record; create a new version.",
        "publication_sync_status": "NOT_STARTED_REQUIRES_USER_APPROVAL",
    }


def _publish(staged: Path, final: Path) -> None:
    """验证完成后发布单个资产，拒绝覆盖。"""
    if final.exists():
        raise RuntimeError(f"拒绝覆盖已有文件：{final}")
    final.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(staged), str(final))


def run_fi_panel_release(
    cohorts: list[str],
    *,
    load_spine: Callable[[str], tuple[pd.DataFrame, Path, dict[str, Any]]],
    load_candidate_intervals: Callable[[str], tuple[pd.DataFrame, Path, dict[str, Any]]],
    load_eligibility_base: Callable[[str], tuple[pd.DataFrame, Path, dict[str, Any]]],
) -> dict[str, Any]:
    """构建、QC、原子发布并冻结三队列 FI Panel v2。"""
    if tuple(cohorts) != COHORTS:
        raise RuntimeError("fi-panel v2 必须使用 --cohort all 一次性生成")
    verify_locked_design(ACTIVE_SPEC)
    _verify_release_absent(ACTIVE_SPEC)
    v1_snapshot = _v1_release_snapshot()
    pre_snapshot = verify_prepanels(
        load_spine, load_candidate_intervals, load_eligibility_base
    )
    outcomes, spines, build_inputs, _outcome_freeze = _load_frozen_outcomes(load_spine)
    demographics = {cohort: _demographic_long(cohort) for cohort in COHORTS}
    underweight_rows, underweight_summary = underweight_qc(
        outcomes["KLoSA"], spines["KLoSA"], demographics["KLoSA"]
    )
    if underweight_summary["decision"] != "PASS":
        raise RuntimeError("KLoSA underweight QC 未通过；停止且不自动替换条目")

    preview_override = os.environ.get("FI_PANEL_PREVIEW_DIR")
    staging_context = tempfile.TemporaryDirectory(prefix="fi_panel_v2_")
    try:
        staging = Path(staging_context.name)
        preview_dir = Path(preview_override) if preview_override else staging / "workbook_previews"
        panels: dict[str, pd.DataFrame] = {}
        staged_outputs: dict[str, Path] = {}
        for cohort in COHORTS:
            panel = build_fi_panel(
                outcomes[cohort], spines[cohort], cohort, ACTIVE_SPEC
            )
            validate_fi_panel(
                panel, spines[cohort], outcomes[cohort], cohort, ACTIVE_SPEC
            )
            staged_output = staging / f"{cohort.lower()}_fi_panel_v2.parquet"
            panel.to_parquet(staged_output, index=False, compression="zstd")
            persisted = pd.read_parquet(staged_output)
            validate_fi_panel(
                persisted, spines[cohort], outcomes[cohort], cohort, ACTIVE_SPEC
            )
            pd.testing.assert_frame_equal(
                persisted, panel, check_dtype=True, check_exact=True
            )
            panels[cohort] = persisted
            staged_outputs[cohort] = staged_output
            print(f"{cohort}: FI panel staged | rows={len(panel):,}")

        completeness_rows, completeness_machine = completeness_qc(panels)
        for cohort in COHORTS:
            if any(
                record[definition]["rate"] is not None
                and record[definition]["rate"] < 0.95
                for record in completeness_machine[cohort]["by_wave"]
                for definition in ("primary_90", "common22_90")
            ):
                raise RuntimeError(f"{cohort}: 90% 完整性逐波支持低于预设人工复核线 95%")
        fi_rows, fi_machine = fi_distribution_qc(panels, demographics)
        _validate_construct_qc(fi_machine)
        top_tail_qc = _top_tail_qc(
            panels, outcomes, spines, build_inputs
        )
        role_qc = potential_role_support_qc(
            panels,
            load_candidate_intervals=load_candidate_intervals,
            load_eligibility_base=load_eligibility_base,
        )

        staged_review = staging / REVIEW_PATH.name
        create_review_workbook(
            underweight_rows,
            completeness_rows,
            fi_rows,
            staged_review,
            preview_dir,
        )
        staged_dictionary = staging / DICTIONARY_PATH.name
        staged_dictionary.write_text(DICTIONARY_TEXT, encoding="utf-8")

        staged_manifests: dict[str, Path] = {}
        for cohort in COHORTS:
            manifest = _build_manifest(
                cohort,
                panels[cohort],
                staged_outputs[cohort],
                build_inputs[cohort],
                completeness_machine[cohort],
                fi_machine[cohort],
                top_tail_qc["cohorts"][cohort],
                underweight_summary if cohort == "KLoSA" else None,
                ACTIVE_SPEC,
            )
            staged_manifest = staging / _manifest_path(cohort).name
            _json_write(staged_manifest, manifest)
            staged_manifests[cohort] = staged_manifest

        staged_freeze = staging / FREEZE_PATH.name
        _json_write(
            staged_freeze,
            _build_freeze(
                staged_outputs,
                staged_manifests,
                staged_review,
                staged_dictionary,
                build_inputs,
                pre_snapshot,
                role_qc,
                underweight_summary,
                top_tail_qc,
                ACTIVE_SPEC,
            ),
        )

        post_snapshot = verify_prepanels(
            load_spine, load_candidate_intervals, load_eligibility_base
        )
        if post_snapshot != pre_snapshot:
            raise RuntimeError("FI 构建前后三张 pre-panel 状态发生变化")
        if _v1_release_snapshot() != v1_snapshot:
            raise RuntimeError("FI v2 构建期间历史 FI Panel v1 冻结资产发生变化")
        if sha256_file(OUTCOME_FREEZE_PATH) != _build_freeze(
            staged_outputs,
            staged_manifests,
            staged_review,
            staged_dictionary,
            build_inputs,
            pre_snapshot,
            role_qc,
            underweight_summary,
            top_tail_qc,
            ACTIVE_SPEC,
        )["outcome_components_freeze"]["sha256"]:
            raise RuntimeError("FI 构建期间 Outcome Components freeze 发生变化")

        for cohort in COHORTS:
            _publish(staged_outputs[cohort], _final_output_path(cohort))
        _publish(staged_review, REVIEW_PATH)
        _publish(staged_dictionary, DICTIONARY_PATH)
        for cohort in COHORTS:
            _publish(staged_manifests[cohort], _manifest_path(cohort))
        _publish(staged_freeze, FREEZE_PATH)
    finally:
        staging_context.cleanup()

    return {
        "status": "FI PANEL V2 FROZEN",
        "freeze_path": str(FREEZE_PATH),
        "review_path": str(REVIEW_PATH),
        "preview_dir": preview_override,
    }


__all__ = [
    "ACTIVE_SPEC",
    "BUILDER_VERSION",
    "COMMON22_THRESHOLD",
    "FIReleaseSpec",
    "FI_PANEL_COLUMNS",
    "FORMAL_WINDOWS",
    "PRIMARY_THRESHOLD",
    "SPEC_VERSION",
    "V1_SPEC",
    "V2_SPEC",
    "build_fi_panel",
    "fi_specification",
    "run_fi_panel_release",
    "specification_hash",
    "validate_fi_panel",
    "verify_locked_design",
    "with_fi_completeness_flags",
]
