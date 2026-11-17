"""三队列 Weight and Survey-Design Panel v1：来源层、QC 与候选发布。"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from pandas.io.stata import StataMissingValue

from panel_core import (
    RUNNER_VERSION,
    normalise_person_id,
    resolve_cohort_dataset_path,
    sha256_file,
)


ROOT = Path(__file__).resolve().parents[2]
ADMIN = ROOT / "00_admin"
REGISTRY_PATH = ADMIN / "cohort_registry.csv"
CODEBOOK_MANIFEST_PATH = ADMIN / "codebooks" / "codebook_manifest.csv"
DESIGN_REGISTER_PATH = ADMIN / "panel_design_register.csv"
VARIABLE_MAP_PATH = ADMIN / "variable_maps" / "weight_design_variable_map.csv"
MANIFEST_ROOT = ADMIN / "panel_manifests"
WEIGHT_MANIFEST_ROOT = MANIFEST_ROOT / "weight_design_panel"
DERIVED_ROOT = ROOT / "01_data" / "derived" / "panel_components"
REVIEW_PATH = ROOT / "04_collaboration" / "weight_design_panel_review_v1.xlsx"
DICTIONARY_PATH = WEIGHT_MANIFEST_ROOT / "weight_design_panel_data_dictionary.md"
RELEASE_CANDIDATE_PATH = (
    WEIGHT_MANIFEST_ROOT / "weight_design_panel_v1_release_candidate.json"
)
FORMAL_FREEZE_PATH = WEIGHT_MANIFEST_ROOT / "weight_design_panel_v1_freeze.json"
WORKBOOK_BUILDER_PATH = Path(__file__).with_name("weight_design_review.mjs")
RUN_R_PATH = ADMIN / "run_r.ps1"
SYNTHETIC_SEMANTICS_PATH = (
    Path(__file__).parent / "r" / "validate_weight_semantics_synthetic.R"
)

COHORTS = ("HRS", "ELSA", "KLoSA")
RELEASE_VERSION = "v1"
BUILDER_VERSION = "weight_design_panel_builder_v1.0.1"
MAP_VERSION = "weight_design_variable_map_v1"
SPEC_VERSION = "weight_design_measurement_spec_v1"

KEY_COLUMNS = ("cohort", "person_id", "wave")
WEIGHT_PANEL_COLUMNS = (
    "cohort",
    "person_id",
    "wave",
    "person_weight_raw",
    "survey_stratum_id",
    "survey_cluster_id",
    "household_id",
)
WEIGHT_PANEL_DTYPES = {
    "cohort": "string",
    "person_id": "string",
    "wave": "int16",
    "person_weight_raw": "float64",
    "survey_stratum_id": "string",
    "survey_cluster_id": "string",
    "household_id": "string",
}
WEIGHT_PANEL_NULLABLE = {
    "cohort": False,
    "person_id": False,
    "wave": False,
    "person_weight_raw": True,
    "survey_stratum_id": True,
    "survey_cluster_id": True,
    "household_id": True,
}
EXPECTED_SPINE_ROWS = {"HRS": 723_744, "ELSA": 216_790, "KLoSA": 100_566}
EXPECTED_INDEX_COUNTS = {"HRS": 8_204, "ELSA": 1_284, "KLoSA": 3_125}
EXPECTED_MARGINAL = {
    "HRS": {
        "index_shocks": 8_204,
        "recovery_classifiable": 6_547,
        "fi_t_available": 7_898,
        "landmark_fi_available": 6_335,
        "landmark_and_later_fi_available": 5_222,
        "nine_covariate_complete": 8_144,
    },
    "ELSA": {
        "index_shocks": 1_284,
        "recovery_classifiable": 968,
        "fi_t_available": 1_282,
        "landmark_fi_available": 993,
        "landmark_and_later_fi_available": 788,
        "nine_covariate_complete": 1_169,
    },
    "KLoSA": {
        "index_shocks": 3_125,
        "recovery_classifiable": 2_755,
        "fi_t_available": 3_125,
        "landmark_fi_available": 2_768,
        "landmark_and_later_fi_available": 2_519,
        "nine_covariate_complete": 3_118,
    },
}
EXPECTED_JOINT = {
    "HRS": {
        "recovery_classifiable": (6_547, 6_484, 63, 0),
        "longitudinal_measurement_support": (5_150, 5_133, 17, 0),
        "complete_covariate_diagnostic_subset": (5_114, 5_097, 17, 0),
    },
    "ELSA": {
        "recovery_classifiable": (968, 928, 13, 27),
        "longitudinal_measurement_support": (772, 748, 9, 15),
        "complete_covariate_diagnostic_subset": (699, 676, 8, 15),
    },
    "KLoSA": {
        "recovery_classifiable": (2_755, 2_755, 0, 0),
        "longitudinal_measurement_support": (2_507, 2_507, 0, 0),
        "complete_covariate_diagnostic_subset": (2_502, 2_502, 0, 0),
    },
}

MAPPING_COLUMNS = (
    "cohort",
    "target_variable",
    "source_variable",
    "source_definition",
    "available_waves",
    "measurement_level",
    "value_provenance",
    "missing_code_policy",
    "id_uniqueness_scope",
    "cross_wave_stability",
    "permitted_downstream_use",
    "harmonisation_status",
    "selection_status",
    "alternative_source",
    "map_version",
)

SOURCE_DECISIONS = {
    "weight_definition": (
        "locked",
        "cohort_specific_raw_cross_sectional_person_weight_at_exact_tplus1_v1",
    ),
    "weight_panel_schema": ("locked", "person_wave_7_columns_v1"),
    "survey_design_id_semantics": (
        "locked",
        "cohort_specific_source_ids_with_klosa_cluster_na_v1",
    ),
}
PENDING_MODEL_DECISIONS = {
    "weight_scaling_rule": ("unlocked", ""),
    "weight_variance_method": ("unlocked", ""),
}


@dataclass(frozen=True)
class CohortWeightSpec:
    """单队列权重与设计字段规范。"""

    cohort: str
    wave_min: int
    wave_max: int
    person_id_source: str
    weight_template: str
    household_template: str
    stratum_source: str
    cluster_source: str
    alternative_templates: tuple[str, ...]
    fixed_design: bool


COHORT_SPECS = {
    "HRS": CohortWeightSpec(
        cohort="HRS",
        wave_min=1,
        wave_max=16,
        person_id_source="hhidpn",
        weight_template="r{wave}wtcrnh",
        household_template="h{wave}hhidc",
        stratum_source="raestrat",
        cluster_source="raestrat + raehsamp",
        alternative_templates=("r{wave}wtresp",),
        fixed_design=True,
    ),
    "ELSA": CohortWeightSpec(
        cohort="ELSA",
        wave_min=1,
        wave_max=10,
        person_id_source="idauniq",
        weight_template="r{wave}wtresp",
        household_template="hh{wave}hhidc",
        stratum_source="r{wave}strat",
        cluster_source="r{wave}clust",
        alternative_templates=("r{wave}lwtresp", "r{wave}lwt4resp"),
        fixed_design=False,
    ),
    "KLoSA": CohortWeightSpec(
        cohort="KLoSA",
        wave_min=1,
        wave_max=9,
        person_id_source="pid",
        weight_template="r{wave}wtresp",
        household_template="hh{wave}hhidc",
        stratum_source="raestrat",
        cluster_source="raeclust (rejected as PSU; output NA)",
        alternative_templates=("r{wave}lwtresp", "r{wave}lwt5resp"),
        fixed_design=True,
    ),
}

UPSTREAM_FREEZES = {
    "spine_v2": MANIFEST_ROOT / "spine" / "spine_v2_freeze.json",
    "candidate_intervals_v3": MANIFEST_ROOT
    / "candidate_intervals"
    / "candidate_intervals_v3_freeze.json",
    "eligibility_base_v2": MANIFEST_ROOT
    / "eligibility_base"
    / "eligibility_base_v2_freeze.json",
    "outcome_components_v2": MANIFEST_ROOT
    / "outcome_components"
    / "outcome_components_v2_freeze.json",
    "fi_panel_v2": MANIFEST_ROOT / "fi_panel" / "fi_panel_v2_freeze.json",
    "wealth_panel_v1": MANIFEST_ROOT / "wealth_panel" / "wealth_panel_v1_freeze.json",
    "exposure_panel_v1": MANIFEST_ROOT
    / "exposure_panel"
    / "exposure_panel_v1_freeze.json",
    "covariate_panel_v1": MANIFEST_ROOT
    / "covariate_panel"
    / "covariate_panel_v1_freeze.json",
}


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def _json_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _stable_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _is_stata_missing(value: Any) -> bool:
    return isinstance(value, StataMissingValue)


def _normalise_numeric(values: pd.Series, variable: str) -> pd.Series:
    """把 tagged/text missing 规范为 NA，并验证有限非负权重。"""
    clean = values.copy().astype("object")
    missing = clean.map(_is_stata_missing)
    clean.loc[missing] = np.nan
    strings = clean.map(lambda value: isinstance(value, str))
    if strings.any():
        stripped = clean.loc[strings].astype(str).str.strip()
        dot_missing = stripped.str.fullmatch(r"\.*", na=False)
        stripped.loc[dot_missing] = pd.NA
        clean.loc[strings] = stripped
    numeric = pd.to_numeric(clean, errors="coerce").astype("float64")
    invalid = clean.notna() & numeric.isna()
    if invalid.any():
        examples = clean.loc[invalid].astype(str).drop_duplicates().head(8).tolist()
        raise RuntimeError(f"{variable}: 出现无法解释的非数值：{examples}")
    infinite = numeric.notna() & ~np.isfinite(numeric)
    if infinite.any():
        raise RuntimeError(f"{variable}: 出现无限值")
    if numeric.lt(0).fillna(False).any():
        examples = numeric.loc[numeric.lt(0)].drop_duplicates().head(8).tolist()
        raise RuntimeError(f"{variable}: 出现负权重：{examples}")
    return numeric


def _normalise_source_id(values: pd.Series, variable: str) -> pd.Series:
    """保留字符串前导零；把 Stata missing、空值和全点字符串转为 NA。"""
    result: list[Any] = []
    for value in values.astype("object"):
        if value is None or _is_stata_missing(value):
            result.append(pd.NA)
            continue
        if isinstance(value, (float, np.floating)):
            if np.isnan(value):
                result.append(pd.NA)
                continue
            if not np.isfinite(value) or float(value) % 1 != 0:
                raise RuntimeError(f"{variable}: 出现非整数数值 ID：{value}")
            result.append(str(int(value)))
            continue
        if isinstance(value, (int, np.integer)):
            result.append(str(int(value)))
            continue
        text = str(value).strip()
        if not text or set(text) == {"."}:
            result.append(pd.NA)
        else:
            result.append(text)
    return pd.Series(result, index=values.index, dtype="string")


def _schema(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return [
        {
            "name": column,
            "dtype": str(frame[column].dtype),
            "nullable": bool(frame[column].isna().any()),
        }
        for column in frame.columns
    ]


def specification() -> dict[str, Any]:
    return {
        "spec_version": SPEC_VERSION,
        "grain": "one row per cohort + person_id + wave",
        "physical_columns": list(WEIGHT_PANEL_COLUMNS),
        "raw_weight_only": True,
        "landmark_rule": "exact t+1 wave after frozen index-shock chronology; no fallback",
        "zero_policy": "retain zero distinct from missing",
        "negative_weight_policy": "block release",
        "cross_wave_id_stability_assumed": False,
        "mean_one_scaling_status": "candidate pending software validation; not stored",
        "variance_status": "pending household and survey-design method validation",
        "cohorts": {
            cohort: {
                "wave_range": [spec.wave_min, spec.wave_max],
                "primary_weight": spec.weight_template,
                "stratum": spec.stratum_source,
                "cluster": spec.cluster_source,
                "household": spec.household_template,
                "alternative_weights": list(spec.alternative_templates),
            }
            for cohort, spec in COHORT_SPECS.items()
        },
    }


def specification_hash() -> str:
    return _stable_hash(specification())


def verify_variable_map_contract() -> list[dict[str, str]]:
    with VARIABLE_MAP_PATH.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != MAPPING_COLUMNS:
            raise RuntimeError("weight_design_variable_map.csv 列契约不一致")
        rows = list(reader)
    if len(rows) != 12:
        raise RuntimeError("weight_design_variable_map.csv 必须为 3 队列 × 4 来源变量")
    keys = [(row["cohort"], row["target_variable"]) for row in rows]
    if len(keys) != len(set(keys)):
        raise RuntimeError("weight_design_variable_map.csv 队列×变量键不唯一")
    targets = set(WEIGHT_PANEL_COLUMNS[3:])
    expected_sources = {
        "HRS": {
            "person_weight_raw": "r{wave}wtcrnh",
            "survey_stratum_id": "raestrat",
            "survey_cluster_id": "raestrat + raehsamp",
            "household_id": "h{wave}hhidc",
        },
        "ELSA": {
            "person_weight_raw": "r{wave}wtresp",
            "survey_stratum_id": "r{wave}strat",
            "survey_cluster_id": "r{wave}clust",
            "household_id": "hh{wave}hhidc",
        },
        "KLoSA": {
            "person_weight_raw": "r{wave}wtresp",
            "survey_stratum_id": "raestrat",
            "survey_cluster_id": "raeclust",
            "household_id": "hh{wave}hhidc",
        },
    }
    for cohort in COHORTS:
        actual = {row["target_variable"] for row in rows if row["cohort"] == cohort}
        if actual != targets:
            raise RuntimeError(f"{cohort}: weight/design mapping 目标变量不完整")
    for row in rows:
        cohort = row["cohort"]
        target = row["target_variable"]
        spec = COHORT_SPECS[cohort]
        if row["source_variable"] != expected_sources[cohort][target]:
            raise RuntimeError(f"{cohort} {target}: source_variable 与可执行规范不一致")
        if row["available_waves"] != f"W{spec.wave_min}-W{spec.wave_max}":
            raise RuntimeError(f"{cohort} {target}: available_waves 不一致")
        if row["map_version"] != MAP_VERSION:
            raise RuntimeError("weight_design_variable_map.csv map_version 不一致")
        for field in (
            "id_uniqueness_scope",
            "cross_wave_stability",
            "permitted_downstream_use",
        ):
            if not row[field].strip():
                raise RuntimeError(f"{cohort} {target}: 缺少 {field}")
    klosa_cluster = next(
        row
        for row in rows
        if row["cohort"] == "KLoSA"
        and row["target_variable"] == "survey_cluster_id"
    )
    if klosa_cluster["selection_status"] != "rejected_as_psu":
        raise RuntimeError("KLoSA raeclust 必须明确登记为 rejected_as_psu")
    return rows


def verify_candidate_design() -> dict[str, dict[str, str]]:
    rows = {row["decision_id"]: row for row in _read_csv_rows(DESIGN_REGISTER_PATH)}
    resolved: dict[str, dict[str, str]] = {}
    for decision_id, (status, selected) in {
        **SOURCE_DECISIONS,
        **PENDING_MODEL_DECISIONS,
    }.items():
        row = rows.get(decision_id)
        if row is None:
            raise RuntimeError(f"design register 缺少 {decision_id}")
        if row["status"] != status or row["selected_value"] != selected:
            raise RuntimeError(
                f"{decision_id}: 状态或 selected_value 与候选边界不一致"
            )
        resolved[decision_id] = row
    return resolved


def design_subset_hash() -> str:
    rows = verify_candidate_design()
    ordered = [
        *SOURCE_DECISIONS,
        *PENDING_MODEL_DECISIONS,
    ]
    return _stable_hash([rows[key] for key in ordered])


def _registry_record(cohort: str) -> dict[str, str]:
    rows = [row for row in _read_csv_rows(REGISTRY_PATH) if row["cohort"] == cohort]
    if len(rows) != 1 or rows[0]["version_pair_verified"].upper() != "TRUE":
        raise RuntimeError(f"{cohort}: dataset/codebook 版本配对未 verified")
    record = rows[0].copy()
    record["dataset_path"] = str(resolve_cohort_dataset_path(cohort))
    return record


def _codebook_record(cohort: str) -> dict[str, str]:
    rows = [
        row
        for row in _read_csv_rows(CODEBOOK_MANIFEST_PATH)
        if row["cohort"] == cohort and row["usage_status"] == "authoritative"
    ]
    if len(rows) != 1:
        raise RuntimeError(f"{cohort}: 缺少唯一 authoritative codebook")
    return rows[0]


def verify_source_context(cohort: str) -> dict[str, Any]:
    registry = _registry_record(cohort)
    codebook = _codebook_record(cohort)
    spec = COHORT_SPECS[cohort]
    if registry["person_id_source"] != spec.person_id_source:
        raise RuntimeError(f"{cohort}: person ID 来源不一致")
    raw_path = Path(registry["dataset_path"])
    codebook_path = _repo_path(codebook["local_path"])
    raw_sha = sha256_file(raw_path)
    codebook_sha = sha256_file(codebook_path)
    spine_freeze = json.loads(
        UPSTREAM_FREEZES["spine_v2"].read_text(encoding="utf-8")
    )
    frozen = spine_freeze["cohorts"][cohort]
    if raw_sha != frozen["source_dataset_sha256"]:
        raise RuntimeError(f"{cohort}: raw 哈希与冻结 Spine 来源不一致")
    if codebook_sha != codebook["sha256"] or codebook_sha != frozen[
        "source_codebook_sha256"
    ]:
        raise RuntimeError(f"{cohort}: authoritative codebook 哈希不一致")
    if cohort == "ELSA" and "Version H" not in codebook["codebook_version"]:
        raise RuntimeError("ELSA: 必须使用匹配的 Version H codebook")
    return {
        "raw_path": raw_path,
        "raw_sha256": raw_sha,
        "raw_bytes": raw_path.stat().st_size,
        "raw_mtime_ns": raw_path.stat().st_mtime_ns,
        "dataset_version": registry["dataset_version_expected"],
        "codebook_path": codebook_path,
        "codebook_sha256": codebook_sha,
        "codebook_version": codebook["codebook_version"],
    }


def _alternative_variable_names(cohort: str, raw_names: set[str]) -> list[str]:
    spec = COHORT_SPECS[cohort]
    result: list[str] = []
    for wave in range(spec.wave_min, spec.wave_max + 1):
        for template in spec.alternative_templates:
            name = template.format(wave=wave)
            if name in raw_names:
                result.append(name)
    return result


def required_source_variables(cohort: str, source_path: Path) -> list[str]:
    spec = COHORT_SPECS[cohort]
    reader = pd.io.stata.StataReader(source_path, convert_categoricals=False)
    raw_names = set(reader.variable_labels())
    variables = [spec.person_id_source]
    if cohort == "HRS":
        variables.extend(["raestrat", "raehsamp"])
    elif cohort == "KLoSA":
        variables.extend(["raestrat", "raeclust"])
    for wave in range(spec.wave_min, spec.wave_max + 1):
        variables.extend(
            [
                spec.weight_template.format(wave=wave),
                spec.household_template.format(wave=wave),
            ]
        )
        if cohort == "ELSA":
            variables.extend(
                [
                    spec.stratum_source.format(wave=wave),
                    spec.cluster_source.format(wave=wave),
                ]
            )
    variables.extend(_alternative_variable_names(cohort, raw_names))
    missing = sorted(set(variables).difference(raw_names))
    if missing:
        raise RuntimeError(f"{cohort}: 原始 DTA 缺少必需字段：{missing}")
    return list(dict.fromkeys(variables))


def read_raw_source(source_path: Path, cohort: str) -> tuple[pd.DataFrame, list[str]]:
    variables = required_source_variables(cohort, source_path)
    raw = pd.read_stata(
        source_path,
        columns=variables,
        convert_categoricals=False,
        convert_missing=True,
        preserve_dtypes=True,
    )
    return raw, variables


def _aligned(raw_index: pd.DataFrame, person_ids: pd.Series, variable: str) -> pd.Series:
    return raw_index.loc[person_ids.astype(str).tolist(), variable].reset_index(drop=True)


def build_weight_design_panel(
    raw: pd.DataFrame,
    spine: pd.DataFrame,
    cohort: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """只从原始 DTA 和冻结 Full Spine 构建七列来源层。"""
    spec = COHORT_SPECS[cohort]
    raw = raw.copy()
    raw["person_id"] = normalise_person_id(raw[spec.person_id_source], cohort)
    if raw["person_id"].duplicated().any():
        raise RuntimeError(f"{cohort}: 原始 person ID 不唯一")
    raw_index = raw.set_index("person_id", drop=False)
    missing_people = set(spine["person_id"].astype(str)).difference(raw_index.index)
    if missing_people:
        raise RuntimeError(f"{cohort}: Full Spine 人员无法回查原始 DTA")

    panel = spine.loc[:, list(KEY_COLUMNS)].copy()
    panel["cohort"] = panel["cohort"].astype("string")
    panel["person_id"] = panel["person_id"].astype("string")
    panel["wave"] = panel["wave"].astype("int16")
    panel["person_weight_raw"] = np.nan
    panel["survey_stratum_id"] = pd.Series(pd.NA, index=panel.index, dtype="string")
    panel["survey_cluster_id"] = pd.Series(pd.NA, index=panel.index, dtype="string")
    panel["household_id"] = pd.Series(pd.NA, index=panel.index, dtype="string")

    source_qc: dict[str, Any] = {
        "primary_weight": spec.weight_template,
        "alternative_weight_by_wave": [],
        "klosa_raeclust_rejected_distribution": None,
        "hrs_verified_99_99_person_n": 0,
    }

    for wave in range(spec.wave_min, spec.wave_max + 1):
        mask = panel["wave"].eq(wave)
        positions = panel.index[mask]
        ids = panel.loc[positions, "person_id"]
        observed = (
            spine.loc[positions, "observed_in_wave"].eq(True).fillna(False).to_numpy()
        )
        weight_name = spec.weight_template.format(wave=wave)
        weight = _normalise_numeric(
            _aligned(raw_index, ids, weight_name), weight_name
        )
        weight.loc[~observed] = np.nan
        panel.loc[positions, "person_weight_raw"] = weight.to_numpy()

        household_name = spec.household_template.format(wave=wave)
        household = _normalise_source_id(
            _aligned(raw_index, ids, household_name), household_name
        )
        household.loc[~observed] = pd.NA
        panel.loc[positions, "household_id"] = household.to_numpy()

        if cohort == "ELSA":
            stratum_name = spec.stratum_source.format(wave=wave)
            cluster_name = spec.cluster_source.format(wave=wave)
            stratum = _normalise_source_id(
                _aligned(raw_index, ids, stratum_name), stratum_name
            )
            cluster = _normalise_source_id(
                _aligned(raw_index, ids, cluster_name), cluster_name
            )
            stratum.loc[~observed] = pd.NA
            cluster.loc[~observed] = pd.NA
            panel.loc[positions, "survey_stratum_id"] = stratum.to_numpy()
            panel.loc[positions, "survey_cluster_id"] = cluster.to_numpy()

        for template in spec.alternative_templates:
            alternative_name = template.format(wave=wave)
            if alternative_name not in raw.columns:
                continue
            alternative = _normalise_numeric(
                _aligned(raw_index, ids, alternative_name), alternative_name
            )
            alternative.loc[~observed] = np.nan
            primary = weight
            both = primary.notna() & alternative.notna()
            source_qc["alternative_weight_by_wave"].append(
                {
                    "wave": wave,
                    "alternative_source": alternative_name,
                    "both_nonmissing_n": int(both.sum()),
                    "both_positive_n": int(
                        (both & primary.gt(0) & alternative.gt(0)).sum()
                    ),
                    "primary_positive_alternative_nonpositive_or_missing_n": int(
                        (primary.gt(0) & ~alternative.gt(0).fillna(False)).sum()
                    ),
                    "alternative_positive_primary_nonpositive_or_missing_n": int(
                        (alternative.gt(0) & ~primary.gt(0).fillna(False)).sum()
                    ),
                    "equal_when_both_nonmissing_n": int(
                        np.isclose(
                            primary.loc[both], alternative.loc[both], rtol=0, atol=0
                        ).sum()
                    ),
                }
            )

    if cohort in {"HRS", "KLoSA"}:
        all_ids = panel["person_id"]
        raw_rows = raw_index.loc[all_ids.astype(str).tolist()]
        stratum = _normalise_source_id(
            raw_rows["raestrat"].reset_index(drop=True), "raestrat"
        )
        if cohort == "HRS":
            half = _normalise_source_id(
                raw_rows["raehsamp"].reset_index(drop=True), "raehsamp"
            )
            pair_99 = stratum.eq("99") & half.eq("99")
            mixed_99 = stratum.eq("99") ^ half.eq("99")
            if mixed_99.any():
                raise RuntimeError("HRS: 出现未经核实的非 99/99 设计码组合")
            source_qc["hrs_verified_99_99_person_n"] = int(
                pd.DataFrame(
                    {"person_id": all_ids, "pair_99": pair_99.to_numpy()}
                )
                .drop_duplicates("person_id")["pair_99"]
                .sum()
            )
            stratum.loc[pair_99] = pd.NA
            half.loc[pair_99] = pd.NA
            valid_cluster = stratum.notna() & half.notna()
            cluster = pd.Series(pd.NA, index=panel.index, dtype="string")
            cluster.loc[valid_cluster] = (
                stratum.loc[valid_cluster] + ":" + half.loc[valid_cluster]
            )
            panel["survey_cluster_id"] = cluster
        else:
            rejected = _normalise_source_id(
                raw_index.loc[raw_index.index, "raeclust"].reset_index(drop=True),
                "raeclust",
            )
            source_qc["klosa_raeclust_rejected_distribution"] = {
                str(key): int(value)
                for key, value in rejected.value_counts(dropna=False).items()
            }
        panel["survey_stratum_id"] = stratum

    panel = panel.loc[:, list(WEIGHT_PANEL_COLUMNS)]
    for column, dtype in WEIGHT_PANEL_DTYPES.items():
        panel[column] = panel[column].astype(dtype)
    return panel, source_qc


def validate_weight_design_panel(
    panel: pd.DataFrame,
    spine: pd.DataFrame,
    cohort: str,
) -> None:
    if tuple(panel.columns) != WEIGHT_PANEL_COLUMNS:
        raise RuntimeError(f"{cohort}: Weight Panel 列顺序不一致")
    for column, dtype in WEIGHT_PANEL_DTYPES.items():
        if str(panel[column].dtype) != dtype:
            raise RuntimeError(
                f"{cohort}: {column} dtype={panel[column].dtype} expected={dtype}"
            )
    if len(panel) != EXPECTED_SPINE_ROWS[cohort] or len(panel) != len(spine):
        raise RuntimeError(f"{cohort}: Weight Panel 行数与 Full Spine 不一致")
    pd.testing.assert_frame_equal(
        panel.loc[:, list(KEY_COLUMNS)].reset_index(drop=True),
        spine.loc[:, list(KEY_COLUMNS)].astype(
            {"cohort": "string", "person_id": "string", "wave": "int16"}
        ).reset_index(drop=True),
        check_dtype=True,
        check_exact=True,
    )
    if panel.duplicated(list(KEY_COLUMNS)).any():
        raise RuntimeError(f"{cohort}: Weight Panel 主键不唯一")
    weight = panel["person_weight_raw"]
    if weight.lt(0).fillna(False).any() or (
        weight.notna() & ~np.isfinite(weight)
    ).any():
        raise RuntimeError(f"{cohort}: Weight Panel 存在负值或无限权重")
    observed = spine["observed_in_wave"].eq(True).fillna(False)
    for column in ("person_weight_raw", "household_id"):
        if panel.loc[~observed, column].notna().any():
            raise RuntimeError(f"{cohort}: 非访谈行错误填充 {column}")
    if cohort == "ELSA":
        for column in ("survey_stratum_id", "survey_cluster_id"):
            if panel.loc[~observed, column].notna().any():
                raise RuntimeError(f"ELSA: 非访谈行错误填充 {column}")
        landmark = panel["wave"].between(3, 9)
        both = (
            landmark
            & panel["survey_cluster_id"].notna()
            & panel["household_id"].notna()
        )
        if not panel.loc[both, "survey_cluster_id"].equals(
            panel.loc[both, "household_id"]
        ):
            raise RuntimeError("ELSA W3-W9: cluster 与同波 household ID 不一致")
    if cohort == "KLoSA" and panel["survey_cluster_id"].notna().any():
        raise RuntimeError("KLoSA: survey_cluster_id 必须全为 NA")
    if cohort == "HRS":
        invalid = panel["survey_cluster_id"].str.contains(
            r"NA|<NA>|None", na=False, regex=True
        )
        if invalid.any():
            raise RuntimeError("HRS: composite cluster 含伪缺失片段")


def _record_paths(record: dict[str, Any]) -> tuple[Path, str, Path, str]:
    if "parquet_path" in record:
        return (
            _repo_path(record["parquet_path"]),
            record["parquet_sha256"],
            _repo_path(record["manifest_path"]),
            record["manifest_sha256"],
        )
    data_path = _repo_path(record["path"])
    manifest_key = (
        "build_manifest_path"
        if "build_manifest_path" in record
        else "active_manifest_path"
    )
    manifest_sha_key = (
        "build_manifest_sha256"
        if "build_manifest_sha256" in record
        else "active_manifest_sha256"
    )
    return (
        data_path,
        record["sha256"],
        _repo_path(record[manifest_key]),
        record[manifest_sha_key],
    )


def _verify_frozen_component(name: str, path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(f"缺少上游 freeze：{path}")
    freeze = json.loads(path.read_text(encoding="utf-8"))
    if freeze.get("freeze_status") != "FROZEN":
        raise RuntimeError(f"{name}: freeze_status 不是 FROZEN")
    cohorts: dict[str, Any] = {}
    for cohort, record in freeze.get("cohorts", {}).items():
        data_path, data_sha, manifest_path, manifest_sha = _record_paths(record)
        actual_data = sha256_file(data_path)
        actual_manifest = sha256_file(manifest_path)
        if actual_data != data_sha or actual_manifest != manifest_sha:
            raise RuntimeError(f"{name} {cohort}: frozen asset 哈希漂移")
        cohorts[cohort] = {
            "data_path": _relative(data_path),
            "data_sha256": actual_data,
            "manifest_path": _relative(manifest_path),
            "manifest_sha256": actual_manifest,
        }
    extras: dict[str, Any] = {}
    for field in ("review_workbook", "data_dictionary"):
        record = freeze.get(field)
        if not isinstance(record, dict) or "path" not in record or "sha256" not in record:
            continue
        extra_path = _repo_path(record["path"])
        actual = sha256_file(extra_path)
        if actual != record["sha256"]:
            raise RuntimeError(f"{name}: {field} 哈希漂移")
        extras[field] = {"path": _relative(extra_path), "sha256": actual}
    return {
        "freeze_path": _relative(path),
        "freeze_sha256": sha256_file(path),
        "cohorts": cohorts,
        **extras,
    }


def protected_upstream_snapshot() -> dict[str, Any]:
    return {
        name: _verify_frozen_component(name, path)
        for name, path in UPSTREAM_FREEZES.items()
    }


def _load_snapshot_panel(
    snapshot: dict[str, Any], component: str, cohort: str
) -> pd.DataFrame:
    path = _repo_path(snapshot[component]["cohorts"][cohort]["data_path"])
    return pd.read_parquet(path)


def _rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def wave_qc_rows(
    panels: dict[str, pd.DataFrame],
    spines: dict[str, pd.DataFrame],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    visible: list[dict[str, Any]] = []
    machine: list[dict[str, Any]] = []
    for cohort in COHORTS:
        panel = panels[cohort]
        spine = spines[cohort]
        for wave in sorted(panel["wave"].unique()):
            wave_mask = panel["wave"].eq(wave)
            observed = wave_mask & spine["observed_in_wave"].eq(True).fillna(False)
            weights = panel.loc[observed, "person_weight_raw"]
            positive = weights.loc[weights.gt(0)]
            positive_n = int(weights.gt(0).sum())
            zero_n = int(weights.eq(0).sum())
            missing_n = int(weights.isna().sum())
            mean = float(positive.mean()) if positive_n else None
            cv = (
                float(positive.std(ddof=0) / mean)
                if positive_n and mean and mean > 0
                else None
            )
            kish = (
                float(positive.sum() ** 2 / positive.pow(2).sum())
                if positive_n
                else None
            )
            record = {
                "cohort": cohort,
                "wave": int(wave),
                "observed_n": int(observed.sum()),
                "positive_weight_n": positive_n,
                "zero_weight_n": zero_n,
                "missing_weight_n": missing_n,
                "positive_landmark_weight_percent": _rate(
                    positive_n, int(observed.sum())
                ),
                "positive_weight_median": (
                    float(positive.median()) if positive_n else None
                ),
                "positive_weight_p99": (
                    float(positive.quantile(0.99)) if positive_n else None
                ),
                "positive_weight_cv": cv,
                "kish_effective_n_weights_only": kish,
                "stratum_nonmissing_n": int(
                    panel.loc[observed, "survey_stratum_id"].notna().sum()
                ),
                "cluster_nonmissing_n": int(
                    panel.loc[observed, "survey_cluster_id"].notna().sum()
                ),
                "household_nonmissing_n": int(
                    panel.loc[observed, "household_id"].notna().sum()
                ),
            }
            machine.append(record)
            visible.append(
                {
                    "Cohort": cohort,
                    "Wave": int(wave),
                    "Observed N": record["observed_n"],
                    "Positive weight": positive_n,
                    "Zero weight": zero_n,
                    "Missing weight": missing_n,
                    "Positive weight, %": record[
                        "positive_landmark_weight_percent"
                    ],
                    "Median": record["positive_weight_median"],
                    "P99": record["positive_weight_p99"],
                    "CV": cv,
                    "Kish effective N, weights only": kish,
                }
            )
    return visible, machine


def design_summary_rows(
    panels: dict[str, pd.DataFrame],
    spines: dict[str, pd.DataFrame],
    wave_machine: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    mapping = verify_variable_map_contract()
    visible: list[dict[str, Any]] = []
    machine: dict[str, Any] = {}
    for cohort in COHORTS:
        panel = panels[cohort]
        spine = spines[cohort]
        observed = spine["observed_in_wave"].eq(True).fillna(False)
        denominator = int(observed.sum())
        rows = [row for row in mapping if row["cohort"] == cohort]
        cluster_map = next(
            row for row in rows if row["target_variable"] == "survey_cluster_id"
        )
        summary_id_semantics = {
            "HRS": {
                "id_uniqueness_scope": (
                    "Mixed: stratum/cluster cohort-wide; household wave-specific"
                ),
                "cross_wave_stability": (
                    "Stratum/cluster fixed; household not assumed stable"
                ),
            },
            "ELSA": {
                "id_uniqueness_scope": "Wave-specific",
                "cross_wave_stability": "Not assumed stable across waves",
            },
            "KLoSA": {
                "id_uniqueness_scope": (
                    "Mixed: stratum cohort-wide; household wave-specific; "
                    "cluster unavailable"
                ),
                "cross_wave_stability": (
                    "Stratum fixed; household not assumed stable; cluster unavailable"
                ),
            },
        }[cohort]
        record = {
            "observed_person_waves": denominator,
            "stratum_nonmissing_n": int(
                panel.loc[observed, "survey_stratum_id"].notna().sum()
            ),
            "cluster_nonmissing_n": int(
                panel.loc[observed, "survey_cluster_id"].notna().sum()
            ),
            "household_nonmissing_n": int(
                panel.loc[observed, "household_id"].notna().sum()
            ),
            "id_uniqueness_scope": summary_id_semantics["id_uniqueness_scope"],
            "cross_wave_stability": summary_id_semantics["cross_wave_stability"],
            "cluster_limitation": cluster_map["permitted_downstream_use"],
            "wave_detail": [row for row in wave_machine if row["cohort"] == cohort],
        }
        machine[cohort] = record
        visible.append(
            {
                "Cohort": cohort,
                "Observed person-waves": denominator,
                "Stratum available, %": _rate(record["stratum_nonmissing_n"], denominator),
                "Cluster available, %": _rate(record["cluster_nonmissing_n"], denominator),
                "Household available, %": _rate(record["household_nonmissing_n"], denominator),
                "ID uniqueness scope": record["id_uniqueness_scope"],
                "Cross-wave stability": record["cross_wave_stability"],
                "Permitted downstream use": record["cluster_limitation"],
            }
        )
    return visible, machine


def _weight_status(values: pd.Series) -> tuple[int, int, int, int, float | None]:
    n = len(values)
    positive = int(values.gt(0).sum())
    zero = int(values.eq(0).sum())
    missing = int(values.isna().sum())
    return n, positive, zero, missing, _rate(positive, n)


def select_landmark_weight(
    index_exposure: pd.DataFrame,
    panel: pd.DataFrame,
) -> pd.DataFrame:
    """只按冻结 index shock 的精确 t+1 键读取 raw weight，不做后续波替换。"""
    required = {"cohort", "person_id", "tplus1_wave", "is_index_shock"}
    missing = required.difference(index_exposure.columns)
    if missing:
        raise RuntimeError(f"index exposure 缺少字段：{sorted(missing)}")
    index = index_exposure.loc[index_exposure["is_index_shock"].eq(True)].copy()
    if index["person_id"].duplicated().any():
        raise RuntimeError("每人必须最多一个 index shock")
    lookup = panel[
        [
            "cohort",
            "person_id",
            "wave",
            "person_weight_raw",
            "survey_stratum_id",
            "survey_cluster_id",
            "household_id",
        ]
    ].rename(columns={"wave": "tplus1_wave"})
    return index.merge(
        lookup,
        on=["cohort", "person_id", "tplus1_wave"],
        how="left",
        validate="one_to_one",
    )


def _nine_covariate_complete(
    index: pd.DataFrame, covariates: pd.DataFrame
) -> pd.Series:
    time_varying = [
        "age",
        "sex",
        "education",
        "marital_status",
        "household_size",
        "labour_force_status",
        "household_income_rank",
        "current_smoking",
    ]
    at_t = covariates[["person_id", "wave", *time_varying]].copy()
    at_t["cov_t_complete"] = at_t[time_varying].notna().all(axis=1)
    at_t = at_t[["person_id", "wave", "cov_t_complete"]].rename(
        columns={"wave": "candidate_shock_wave"}
    )
    at_tm1 = covariates[["person_id", "wave", "wealth_quintile"]].copy()
    at_tm1["wealth_q_complete"] = at_tm1["wealth_quintile"].notna()
    at_tm1 = at_tm1[["person_id", "wave", "wealth_q_complete"]].rename(
        columns={"wave": "tminus1_wave"}
    )
    joined = index[["person_id", "candidate_shock_wave", "tminus1_wave"]].merge(
        at_t, on=["person_id", "candidate_shock_wave"], how="left", validate="one_to_one"
    ).merge(
        at_tm1, on=["person_id", "tminus1_wave"], how="left", validate="one_to_one"
    )
    return (
        joined["cov_t_complete"].fillna(False)
        & joined["wealth_q_complete"].fillna(False)
    ).astype(bool)


def readiness_diagnostics(
    panels: dict[str, pd.DataFrame],
    upstream: dict[str, Any],
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
]:
    marginal_visible: list[dict[str, Any]] = []
    joint_visible: list[dict[str, Any]] = []
    machine: dict[str, Any] = {}
    for cohort in COHORTS:
        exposure = _load_snapshot_panel(upstream, "exposure_panel_v1", cohort)
        fi = _load_snapshot_panel(upstream, "fi_panel_v2", cohort)
        covariates = _load_snapshot_panel(upstream, "covariate_panel_v1", cohort)
        for frame in (exposure, fi, covariates):
            frame["person_id"] = frame["person_id"].astype("string")
        index = exposure.loc[exposure["is_index_shock"].eq(True)].copy()
        if len(index) != EXPECTED_INDEX_COUNTS[cohort]:
            raise RuntimeError(f"{cohort}: index shock 数量与冻结预期不一致")
        if index["person_id"].duplicated().any():
            raise RuntimeError(f"{cohort}: 每人必须最多一个 index shock")

        data = select_landmark_weight(index, panels[cohort])
        fi_t = fi[["person_id", "wave", "fi_primary"]].rename(
            columns={"wave": "candidate_shock_wave", "fi_primary": "fi_t"}
        )
        fi_landmark = fi[["person_id", "wave", "fi_primary"]].rename(
            columns={"wave": "tplus1_wave", "fi_primary": "fi_landmark"}
        )
        data = data.merge(
            fi_t,
            on=["person_id", "candidate_shock_wave"],
            how="left",
            validate="one_to_one",
        ).merge(
            fi_landmark,
            on=["person_id", "tplus1_wave"],
            how="left",
            validate="one_to_one",
        )
        later = data[["person_id", "tplus1_wave"]].merge(
            fi.loc[fi["fi_primary"].notna(), ["person_id", "wave"]],
            on="person_id",
            how="left",
        )
        later = (
            later.loc[later["wave"].gt(later["tplus1_wave"])]
            .groupby("person_id")
            .size()
            .gt(0)
            .rename("later_fi")
        )
        data = data.merge(later, on="person_id", how="left")
        data["later_fi"] = data["later_fi"].fillna(False)
        complete_covariates = _nine_covariate_complete(index, covariates)
        data["nine_covariate_complete"] = complete_covariates.to_numpy()

        classifiable = data["recovery_status"].notna()
        longitudinal = (
            classifiable
            & data["fi_t"].notna()
            & data["fi_landmark"].notna()
            & data["later_fi"]
        )
        complete = longitudinal & data["nine_covariate_complete"]
        marginal = {
            "index_shocks": len(data),
            "recovery_classifiable": int(classifiable.sum()),
            "fi_t_available": int(data["fi_t"].notna().sum()),
            "landmark_fi_available": int(data["fi_landmark"].notna().sum()),
            "landmark_and_later_fi_available": int(
                (data["fi_landmark"].notna() & data["later_fi"]).sum()
            ),
            "nine_covariate_complete": int(data["nine_covariate_complete"].sum()),
        }
        if marginal != EXPECTED_MARGINAL[cohort]:
            raise RuntimeError(
                f"{cohort}: 边际支持人数未复现：{marginal} != {EXPECTED_MARGINAL[cohort]}"
            )
        marginal_visible.append(
            {
                "Cohort": cohort,
                "Index shocks": marginal["index_shocks"],
                "Recovery classifiable": marginal["recovery_classifiable"],
                "FI at t": marginal["fi_t_available"],
                "Landmark FI": marginal["landmark_fi_available"],
                "Landmark FI and later FI": marginal[
                    "landmark_and_later_fi_available"
                ],
                "Nine covariates complete": marginal["nine_covariate_complete"],
            }
        )

        subsets = {
            "recovery_classifiable": classifiable,
            "longitudinal_measurement_support": longitudinal,
            "complete_covariate_diagnostic_subset": complete,
        }
        joint: dict[str, Any] = {}
        for name, mask in subsets.items():
            status = _weight_status(data.loc[mask, "person_weight_raw"])
            if status[:4] != EXPECTED_JOINT[cohort][name]:
                raise RuntimeError(
                    f"{cohort} {name}: 联合支持人数未复现："
                    f"{status[:4]} != {EXPECTED_JOINT[cohort][name]}"
                )
            record = {
                "n": status[0],
                "positive_weight_n": status[1],
                "zero_weight_n": status[2],
                "missing_weight_n": status[3],
                "positive_landmark_weight_percent": status[4],
            }
            joint[name] = record
            joint_visible.append(
                {
                    "Cohort": cohort,
                    "Diagnostic subset": name,
                    "N": status[0],
                    "Positive landmark weight": status[1],
                    "Zero weight": status[2],
                    "Missing weight": status[3],
                    "Positive landmark weight, %": status[4],
                }
            )

        household = data.loc[
            data["household_id"].notna(),
            ["tplus1_wave", "household_id"],
        ]
        sizes = household.groupby(["tplus1_wave", "household_id"]).size()
        shared = sizes.loc[sizes.ge(2)]
        household_diagnostic = {
            "landmark_household_nonmissing_n": len(household),
            "same_wave_shared_household_n": len(shared),
            "participants_in_same_wave_shared_households_n": int(shared.sum()),
            "maximum_participants_per_same_wave_household": (
                int(sizes.max()) if len(sizes) else None
            ),
            "interpretation": (
                "same-wave landmark source diagnostic only; not a count of stable "
                "cross-wave clustering units"
            ),
        }
        machine[cohort] = {
            "marginal_measurement_availability": marginal,
            "joint_support": joint,
            "same_wave_household_diagnostic": household_diagnostic,
            "diagnostic_only": True,
            "not_final_eligibility_or_analysis_sample": True,
        }
    return marginal_visible, joint_visible, machine


def definition_rows(mapping: list[dict[str, str]]) -> list[dict[str, Any]]:
    return [
        {
            "Cohort": row["cohort"],
            "Panel field": row["target_variable"],
            "Source": row["source_variable"],
            "Definition": row["source_definition"],
            "Waves": row["available_waves"],
            "ID uniqueness scope": row["id_uniqueness_scope"],
            "Cross-wave stability": row["cross_wave_stability"],
            "Permitted downstream use": row["permitted_downstream_use"],
            "Selection": row["selection_status"],
        }
        for row in mapping
    ]


def _runtime_paths() -> tuple[Path, Path]:
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
    *,
    definitions: list[dict[str, Any]],
    weight_rows: list[dict[str, Any]],
    design_rows: list[dict[str, Any]],
    marginal_rows: list[dict[str, Any]],
    joint_rows: list[dict[str, Any]],
    output_path: Path,
    preview_dir: Path,
) -> dict[str, Any]:
    """用 artifact-tool 创建并复载检查三表 review workbook。"""
    blocks = [definitions, weight_rows, design_rows, marginal_rows, joint_rows]
    if any(not block for block in blocks):
        raise RuntimeError("Weight review workbook 的所有表块均不得为空")
    payload = {
        "definitions": definitions,
        "weights": weight_rows,
        "design": design_rows,
        "marginal": marginal_rows,
        "joint": joint_rows,
    }
    node, node_modules = _runtime_paths()
    with tempfile.TemporaryDirectory(prefix="weight_design_xlsx_") as tmp_name:
        tmp = Path(tmp_name)
        payload_path = tmp / "payload.json"
        builder_path = tmp / WORKBOOK_BUILDER_PATH.name
        payload_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        shutil.copy2(WORKBOOK_BUILDER_PATH, builder_path)
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
                "artifact-tool Weight review workbook 构建失败：\n"
                + (completed.stdout or "")
                + "\n"
                + (completed.stderr or "")
            )
        if not output_path.is_file():
            raise RuntimeError("artifact-tool 未生成 Weight review workbook")
        stdout = completed.stdout.strip()
        forbidden = (
            "#REF!",
            "#DIV/0!",
            "#VALUE!",
            "#NAME?",
            "#NUM!",
            "#NULL!",
            "#SPILL!",
            "#CALC!",
        )
        if any(token in stdout for token in forbidden):
            raise RuntimeError("Weight review workbook 公式错误扫描未通过")
        previews = sorted(str(path) for path in preview_dir.glob("*.png"))
        if len(previews) != 3:
            raise RuntimeError("Weight review workbook 未完成三 Sheet 渲染")
        return {
            "artifact_tool_stdout": stdout,
            "post_save_reload": "PASS",
            "formula_error_scan": "PASS",
            "preview_files": previews,
        }


DICTIONARY_TEXT = """# Weight and Survey-Design Panel v1 数据字典

## 状态与粒度

每行表示一个 `cohort + person_id + wave`。行集、主键和顺序与冻结 Full Spine v2 完全一致。本文件是来源层 release candidate；正式 freeze 前不得用于 eligibility-final、analysis panel 或模型。

| Column | dtype | nullable | 定义 |
|---|---|---:|---|
| `cohort` | string | no | 队列标识。 |
| `person_id` | string | no | 队列内个人标识。 |
| `wave` | int16 | no | 队列全局波次。 |
| `person_weight_raw` | float64 | yes | 官方同波横截面个人权重。HRS 为 `rWwtcrnh`；ELSA/KLoSA 为 `rWwtresp`。零和 NA 严格区分，不缩放、不跨波回填。 |
| `survey_stratum_id` | string | yes | 来源支持的分层 ID。HRS/KLoSA 为固定个人属性；ELSA 为同波 `rWstrat`。 |
| `survey_cluster_id` | string | yes | HRS 为 `RAESTRAT:RAEHSAMP`复合键；ELSA 为官方同波 `rWclust`但不声称原始 PSU；KLoSA 因 `RAECLUST`仅为 residential type 而全为 NA。 |
| `household_id` | string | yes | 来源支持的同波 household ID，仅在访谈行赋值。 |

## 缺失与 ID 语义

- Stata tagged missing、空字符串和全点字符串统一为 NA；权重零值保留，负值阻止发布。
- HRS composite cluster 只有两个组成项都有效时才生成；已核实的 `99/99` EGENX 组合置为 NA，不把其他 `99`泛化为缺失。
- Wave-specific household/cluster ID 不假定跨波稳定。未验证跨波链接前，仅可用于同波来源诊断，不得直接定义 pooled-landmark 模型的全局聚类单位。
- ELSA landmark W3-W9 的 `rWclust`与同波 household serial 相同；两列可同值，但保留不同概念角色。

## Landmark 与模型边界

最终 eligibility 层只能按冻结 index shock 精确读取 `t+1`行的 `person_weight_raw`；缺失或零权重不得用其他波次替换。Mean-one scaling、WeMix 权重层级、家庭/PSU/strata 方差处理和最终加权 estimand 均未在本候选中锁定。

Kish effective N 只反映权重分布，不是混合模型的最终有效样本量。Participant-clustered standard errors 也不能自动处理不同参与者之间的家庭相关。
"""


def _git_output(arguments: list[str], *, binary: bool = False) -> bytes | str:
    completed = subprocess.run(
        ["git", *arguments], cwd=ROOT, check=True, capture_output=True
    )
    return completed.stdout if binary else completed.stdout.decode("utf-8", errors="replace")


def git_provenance() -> dict[str, Any]:
    commit = str(_git_output(["rev-parse", "HEAD"])).strip()
    status = str(_git_output(["status", "--porcelain=v1", "--untracked-files=all"]))
    diff = _git_output(["diff", "--binary", "--", "."], binary=True)
    return {
        "git_commit": commit,
        "git_dirty": bool(status.strip()),
        "git_diff_sha256": hashlib.sha256(diff).hexdigest(),
        "git_status_porcelain": status.splitlines(),
    }


def _release_contract_snapshot() -> dict[str, Any]:
    mapping = verify_variable_map_contract()
    verify_candidate_design()
    return {
        "runner_version": RUNNER_VERSION,
        "builder_version": BUILDER_VERSION,
        "builder_sha256": sha256_file(Path(__file__)),
        "workbook_builder_sha256": sha256_file(WORKBOOK_BUILDER_PATH),
        "synthetic_semantics_script_sha256": sha256_file(SYNTHETIC_SEMANTICS_PATH),
        "spec_version": SPEC_VERSION,
        "specification_sha256": specification_hash(),
        "map_version": MAP_VERSION,
        "mapping_sha256": sha256_file(VARIABLE_MAP_PATH),
        "mapping_records_sha256": _stable_hash(mapping),
        "design_register_sha256": sha256_file(DESIGN_REGISTER_PATH),
        "design_subset_sha256": design_subset_hash(),
        "git_provenance": git_provenance(),
    }


def _final_output_path(cohort: str) -> Path:
    return (
        DERIVED_ROOT
        / cohort
        / "09_weight_design_panel"
        / f"{cohort.lower()}_weight_design_panel_v1.parquet"
    )


def _manifest_path(cohort: str) -> Path:
    return WEIGHT_MANIFEST_ROOT / f"{cohort.lower()}_weight_design_panel_v1.json"


def _candidate_paths() -> list[Path]:
    return [
        *(_final_output_path(cohort) for cohort in COHORTS),
        *(_manifest_path(cohort) for cohort in COHORTS),
        REVIEW_PATH,
        DICTIONARY_PATH,
        RELEASE_CANDIDATE_PATH,
        FORMAL_FREEZE_PATH,
    ]


def _verify_release_absent() -> None:
    existing = [str(path) for path in _candidate_paths() if path.exists()]
    if existing:
        raise RuntimeError("拒绝覆盖已有 Weight Panel v1 candidate 资产：" + "; ".join(existing))


def _serialise_source_context(context: dict[str, Any]) -> dict[str, Any]:
    return {
        "raw_path": str(context["raw_path"].resolve()),
        "raw_sha256": context["raw_sha256"],
        "raw_bytes": context["raw_bytes"],
        "raw_mtime_ns": context["raw_mtime_ns"],
        "dataset_version": context["dataset_version"],
        "codebook_path": _relative(context["codebook_path"]),
        "codebook_sha256": context["codebook_sha256"],
        "codebook_version": context["codebook_version"],
    }


def prepare_measurements(
    *,
    source_contexts: dict[str, dict[str, Any]],
    upstream: dict[str, Any],
    load_spine: Callable[[str], tuple[pd.DataFrame, Path, dict[str, Any]]],
) -> dict[str, Any]:
    panels: dict[str, pd.DataFrame] = {}
    spines: dict[str, pd.DataFrame] = {}
    source_qc: dict[str, Any] = {}
    selected_variables: dict[str, list[str]] = {}
    for cohort in COHORTS:
        spine, _spine_path, _spine_manifest = load_spine(cohort)
        if len(spine) != EXPECTED_SPINE_ROWS[cohort]:
            raise RuntimeError(f"{cohort}: Full Spine 行数与冻结预期不一致")
        raw, variables = read_raw_source(source_contexts[cohort]["raw_path"], cohort)
        panel, cohort_source_qc = build_weight_design_panel(raw, spine, cohort)
        validate_weight_design_panel(panel, spine, cohort)
        panels[cohort] = panel
        spines[cohort] = spine
        source_qc[cohort] = cohort_source_qc
        selected_variables[cohort] = variables
        del raw
    weight_visible, weight_machine = wave_qc_rows(panels, spines)
    design_visible, design_machine = design_summary_rows(
        panels, spines, weight_machine
    )
    marginal_visible, joint_visible, readiness_machine = readiness_diagnostics(
        panels, upstream
    )
    return {
        "panels": panels,
        "spines": spines,
        "source_qc": source_qc,
        "selected_variables": selected_variables,
        "definition_visible": definition_rows(verify_variable_map_contract()),
        "weight_visible": weight_visible,
        "weight_machine": weight_machine,
        "design_visible": design_visible,
        "design_machine": design_machine,
        "marginal_visible": marginal_visible,
        "joint_visible": joint_visible,
        "readiness_machine": readiness_machine,
    }


def _build_manifest(
    cohort: str,
    *,
    panel: pd.DataFrame,
    staged_output: Path,
    source_context: dict[str, Any],
    selected_variables: list[str],
    source_qc: dict[str, Any],
    weight_qc: list[dict[str, Any]],
    design_qc: dict[str, Any],
    readiness: dict[str, Any],
    synthetic_semantics: dict[str, Any],
    upstream: dict[str, Any],
    contract: dict[str, Any],
) -> dict[str, Any]:
    provenance = contract["git_provenance"]
    return {
        "manifest_schema_version": "weight_design_panel_manifest_v1",
        "release_version": RELEASE_VERSION,
        "release_status": "READY_FOR_REVIEW_SOURCE_READY",
        "model_scaling_and_variance_status": "PENDING_VALIDATION",
        "cohort": cohort,
        "generated_utc": datetime.now(UTC).isoformat(),
        "git_commit": provenance["git_commit"],
        "git_dirty": provenance["git_dirty"],
        "git_diff_sha256": provenance["git_diff_sha256"],
        "runner_version": contract["runner_version"],
        "builder_version": contract["builder_version"],
        "builder_sha256": contract["builder_sha256"],
        "workbook_builder_sha256": contract["workbook_builder_sha256"],
        "spec_version": contract["spec_version"],
        "specification_sha256": contract["specification_sha256"],
        "map_version": contract["map_version"],
        "mapping_sha256": contract["mapping_sha256"],
        "mapping_records_sha256": contract["mapping_records_sha256"],
        "design_register_sha256": contract["design_register_sha256"],
        "design_subset_sha256": contract["design_subset_sha256"],
        "key": list(KEY_COLUMNS),
        "row_count": len(panel),
        "key_unique": not panel.duplicated(list(KEY_COLUMNS)).any(),
        "key_set_and_order_equal_full_spine": True,
        "schema": _schema(panel),
        "schema_contract": [
            {
                "name": column,
                "dtype": WEIGHT_PANEL_DTYPES[column],
                "nullable": WEIGHT_PANEL_NULLABLE[column],
            }
            for column in WEIGHT_PANEL_COLUMNS
        ],
        "source": {
            **_serialise_source_context(source_context),
            "selected_variables": selected_variables,
            "selected_variable_count": len(selected_variables),
        },
        "source_qc": source_qc,
        "wave_weight_qc": weight_qc,
        "design_id_qc": design_qc,
        "landmark_support_diagnostic": readiness,
        "synthetic_weight_semantics": synthetic_semantics,
        "landmark_rule": {
            "wave": "exact t+1 after frozen index-shock chronology",
            "later_wave_fallback": False,
            "diagnostic_only_in_this_stage": True,
        },
        "raw_weight_policy": {
            "zero_retained": True,
            "missing_retained": True,
            "negative_prohibited": True,
            "scaled_weight_stored": False,
            "mean_one_scaling_locked": False,
        },
        "id_scope_policy": {
            "cross_wave_stability_assumed": False,
            "pooled_landmark_clustering_authorized": False,
            "klosa_raeclust_used_as_psu": False,
        },
        "upstream_frozen_snapshot": upstream,
        "output": {
            "path": _relative(_final_output_path(cohort)),
            "sha256": sha256_file(staged_output),
        },
        "forbidden_actions_confirmed_absent": [
            "final eligibility filtering",
            "weight imputation or cross-wave fallback",
            "weight rescaling",
            "analysis panel construction",
            "model fitting",
            "variance-method approval",
        ],
    }


def _build_release_candidate(
    *,
    staged_outputs: dict[str, Path],
    staged_manifests: dict[str, Path],
    staged_review: Path,
    staged_dictionary: Path,
    prepared: dict[str, Any],
    synthetic_semantics: dict[str, Any],
    upstream: dict[str, Any],
    contract: dict[str, Any],
) -> dict[str, Any]:
    provenance = contract["git_provenance"]
    return {
        "release_candidate_schema_version": "weight_design_panel_release_candidate_v1",
        "release_status": "READY_FOR_REVIEW_SOURCE_READY",
        "model_scaling_and_variance_status": "PENDING_VALIDATION",
        "release_version": RELEASE_VERSION,
        "generated_utc": datetime.now(UTC).isoformat(),
        "git_commit": provenance["git_commit"],
        "git_dirty": provenance["git_dirty"],
        "git_diff_sha256": provenance["git_diff_sha256"],
        "runner_version": contract["runner_version"],
        "builder_version": contract["builder_version"],
        "builder_sha256": contract["builder_sha256"],
        "workbook_builder_sha256": contract["workbook_builder_sha256"],
        "synthetic_semantics_script_sha256": contract[
            "synthetic_semantics_script_sha256"
        ],
        "spec_version": contract["spec_version"],
        "specification_sha256": contract["specification_sha256"],
        "mapping": {
            "path": _relative(VARIABLE_MAP_PATH),
            "sha256": contract["mapping_sha256"],
            "records_sha256": contract["mapping_records_sha256"],
            "map_version": contract["map_version"],
        },
        "design_register": {
            "path": _relative(DESIGN_REGISTER_PATH),
            "full_file_sha256": contract["design_register_sha256"],
            "design_subset_sha256": contract["design_subset_sha256"],
            "source_decisions_locked": SOURCE_DECISIONS,
            "model_decisions_remaining_unlocked": list(PENDING_MODEL_DECISIONS),
        },
        "common_key": list(KEY_COLUMNS),
        "physical_columns": list(WEIGHT_PANEL_COLUMNS),
        "common_schema_contract": [
            {
                "name": column,
                "dtype": WEIGHT_PANEL_DTYPES[column],
                "nullable": WEIGHT_PANEL_NULLABLE[column],
            }
            for column in WEIGHT_PANEL_COLUMNS
        ],
        "cohorts": {
            cohort: {
                "parquet_path": _relative(_final_output_path(cohort)),
                "parquet_sha256": sha256_file(staged_outputs[cohort]),
                "manifest_path": _relative(_manifest_path(cohort)),
                "manifest_sha256": sha256_file(staged_manifests[cohort]),
                "row_count": len(prepared["panels"][cohort]),
                "index_shocks": prepared["readiness_machine"][cohort][
                    "marginal_measurement_availability"
                ]["index_shocks"],
                "joint_support": prepared["readiness_machine"][cohort][
                    "joint_support"
                ],
            }
            for cohort in COHORTS
        },
        "review_workbook": {
            "path": _relative(REVIEW_PATH),
            "sha256": sha256_file(staged_review),
            "sheets": [
                "1_Definition_Contract",
                "2_Wave_Weight_Design_QC",
                "3_Landmark_Readiness",
            ],
            "post_save_reload": "PASS",
            "formula_error_scan": "PASS",
            "three_sheet_render": "PASS",
        },
        "data_dictionary": {
            "path": _relative(DICTIONARY_PATH),
            "sha256": sha256_file(staged_dictionary),
        },
        "synthetic_weight_semantics": synthetic_semantics,
        "protected_inputs_before_and_after_identical": True,
        "release_contract_before_and_after_identical": True,
        "upstream_frozen_snapshot": upstream,
        "repository_test_baseline": {
            "command": "uv run --locked pytest -q",
            "passed": 64,
            "failed": 1,
            "known_failure": "test_supplementary_table_s3_reflects_v2_boundary",
            "release_requirement": "same known failure only; no new failures",
        },
        "direct_freeze_rule": (
            "Direct freezing without rebuilding is permitted only when approval changes "
            "status or documentation. Any change to source mapping, masking rules, "
            "landmark measurement specification, or schema invalidates this candidate."
        ),
        "formal_freeze_created": False,
        "downstream_use_authorized": False,
        "next_stage_status": "HUMAN_REVIEW_REQUIRED_BEFORE_SOURCE_FREEZE",
    }


def _publish_candidate_transaction(staged_by_final: list[tuple[Path, Path]]) -> None:
    published: list[Path] = []
    try:
        for staged, final in staged_by_final:
            if final.exists():
                raise RuntimeError(f"拒绝覆盖已有文件：{final}")
            final.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(staged), str(final))
            published.append(final)
    except Exception:
        for final in reversed(published):
            resolved = final.resolve()
            if ROOT.resolve() not in resolved.parents:
                raise RuntimeError(f"拒绝回滚删除仓库外文件：{final}")
            if final.is_file():
                final.unlink()
        raise


def run_synthetic_weight_semantics(output_path: Path) -> dict[str, Any]:
    """通过项目 R 入口运行仅含合成数据的 WeMix/lme4 语义对账。"""
    completed = subprocess.run(
        [
            "powershell.exe",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(RUN_R_PATH),
            str(SYNTHETIC_SEMANTICS_PATH),
            str(output_path),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "合成 WeMix/lme4 权重语义对账失败：\n"
            + (completed.stdout or "")
            + "\n"
            + (completed.stderr or "")
        )
    if not output_path.is_file():
        raise RuntimeError("合成权重语义脚本未生成 TSV")
    records = pd.read_csv(output_path, sep="\t", dtype="string")
    if list(records.columns) != ["metric", "value"] or records["metric"].duplicated().any():
        raise RuntimeError("合成权重语义 TSV 契约不一致")
    raw_result = dict(zip(records["metric"], records["value"], strict=True))
    integer_fields = {
        "participant_n",
        "observation_n",
        "minimum_visits",
        "maximum_visits",
    }
    float_fields = {
        "conditional_unconditional_coef_max_abs",
        "conditional_unconditional_var_max_abs",
        "unit_wemix_lme4_coef_max_abs",
        "unit_wemix_lme4_sigma_abs",
        "raw_participant_weight_mean",
        "scaled_participant_weight_mean",
        "raw_scaled_fixed_effect_max_abs_difference",
        "raw_scaled_variance_component_max_abs_difference",
        "raw_scaled_standard_error_max_abs_difference",
        "raw_scaled_objective_difference",
    }
    boolean_fields = {
        "unequal_followup_present",
        "participant_weight_divided_by_visit_count",
        "conditional_cWeights",
        "unconditional_cWeights",
    }
    result: dict[str, Any] = {}
    for key, value in raw_result.items():
        if key in integer_fields:
            result[key] = int(value)
        elif key in float_fields:
            result[key] = float(value)
        elif key in boolean_fields:
            result[key] = value.upper() == "TRUE"
        else:
            result[key] = value
    if result.get("status") != "PASS_SYNTHETIC_SEMANTICS_ONLY":
        raise RuntimeError("合成权重语义结果状态异常")
    return result


def run_weight_design_panel_release(
    cohorts: list[str],
    *,
    load_spine: Callable[[str], tuple[pd.DataFrame, Path, dict[str, Any]]],
) -> dict[str, Any]:
    """构建三队列七列 Weight/Design Panel v1 来源候选。"""
    if tuple(cohorts) != COHORTS:
        raise RuntimeError("weights v1 必须使用 --cohort all 一次性构建三队列共同候选")
    _verify_release_absent()
    contract_before = _release_contract_snapshot()
    upstream_before = protected_upstream_snapshot()
    source_before = {cohort: verify_source_context(cohort) for cohort in COHORTS}
    prepared = prepare_measurements(
        source_contexts=source_before,
        upstream=upstream_before,
        load_spine=load_spine,
    )
    preview_dir = Path(tempfile.gettempdir()) / "weight_design_panel_review_v1_previews"
    if preview_dir.exists():
        shutil.rmtree(preview_dir)

    with tempfile.TemporaryDirectory(prefix="weight_design_panel_v1_") as tmp_name:
        staging = Path(tmp_name)
        synthetic_semantics = run_synthetic_weight_semantics(
            staging / "synthetic_weight_semantics.tsv"
        )
        staged_outputs: dict[str, Path] = {}
        for cohort in COHORTS:
            staged = staging / _final_output_path(cohort).name
            prepared["panels"][cohort].to_parquet(
                staged, index=False, compression="zstd"
            )
            reloaded = pd.read_parquet(staged)
            validate_weight_design_panel(
                reloaded, prepared["spines"][cohort], cohort
            )
            pd.testing.assert_frame_equal(
                reloaded,
                prepared["panels"][cohort],
                check_dtype=True,
                check_exact=True,
            )
            staged_outputs[cohort] = staged

        staged_review = staging / REVIEW_PATH.name
        workbook_checks = create_review_workbook(
            definitions=prepared["definition_visible"],
            weight_rows=prepared["weight_visible"],
            design_rows=prepared["design_visible"],
            marginal_rows=prepared["marginal_visible"],
            joint_rows=prepared["joint_visible"],
            output_path=staged_review,
            preview_dir=preview_dir,
        )
        if workbook_checks["post_save_reload"] != "PASS":
            raise RuntimeError("Weight review workbook 保存后复载未通过")

        staged_dictionary = staging / DICTIONARY_PATH.name
        staged_dictionary.write_text(DICTIONARY_TEXT, encoding="utf-8")
        staged_manifests: dict[str, Path] = {}
        for cohort in COHORTS:
            manifest = _build_manifest(
                cohort,
                panel=prepared["panels"][cohort],
                staged_output=staged_outputs[cohort],
                source_context=source_before[cohort],
                selected_variables=prepared["selected_variables"][cohort],
                source_qc=prepared["source_qc"][cohort],
                weight_qc=[
                    row for row in prepared["weight_machine"] if row["cohort"] == cohort
                ],
                design_qc=prepared["design_machine"][cohort],
                readiness=prepared["readiness_machine"][cohort],
                synthetic_semantics=synthetic_semantics,
                upstream=upstream_before,
                contract=contract_before,
            )
            staged_manifest = staging / _manifest_path(cohort).name
            _json_write(staged_manifest, manifest)
            staged_manifests[cohort] = staged_manifest

        source_after = {cohort: verify_source_context(cohort) for cohort in COHORTS}
        if {
            cohort: _serialise_source_context(source_before[cohort])
            for cohort in COHORTS
        } != {
            cohort: _serialise_source_context(source_after[cohort])
            for cohort in COHORTS
        }:
            raise RuntimeError("Weight 构建期间 raw/codebook 发生变化")
        upstream_after = protected_upstream_snapshot()
        if upstream_before != upstream_after:
            raise RuntimeError("Weight 构建期间上游 frozen 资产发生变化")
        if contract_before != _release_contract_snapshot():
            raise RuntimeError("Weight 构建期间 builder/mapping/design/Git provenance 发生变化")

        staged_candidate = staging / RELEASE_CANDIDATE_PATH.name
        _json_write(
            staged_candidate,
            _build_release_candidate(
                staged_outputs=staged_outputs,
                staged_manifests=staged_manifests,
                staged_review=staged_review,
                staged_dictionary=staged_dictionary,
                prepared=prepared,
                synthetic_semantics=synthetic_semantics,
                upstream=upstream_before,
                contract=contract_before,
            ),
        )
        if contract_before != _release_contract_snapshot():
            raise RuntimeError("Weight 发布前 release contract 发生变化")
        staged_by_final = [
            *((staged_outputs[cohort], _final_output_path(cohort)) for cohort in COHORTS),
            (staged_review, REVIEW_PATH),
            (staged_dictionary, DICTIONARY_PATH),
            *((staged_manifests[cohort], _manifest_path(cohort)) for cohort in COHORTS),
            (staged_candidate, RELEASE_CANDIDATE_PATH),
        ]
        _publish_candidate_transaction(staged_by_final)

    return {
        "release_status": "READY_FOR_REVIEW_SOURCE_READY",
        "model_scaling_and_variance_status": "PENDING_VALIDATION",
        "release_candidate_path": str(RELEASE_CANDIDATE_PATH),
        "review_workbook": str(REVIEW_PATH),
        "preview_dir": str(preview_dir),
        "cohorts": {
            cohort: {
                "rows": len(prepared["panels"][cohort]),
                "output": str(_final_output_path(cohort)),
                "joint_support": prepared["readiness_machine"][cohort]["joint_support"],
            }
            for cohort in COHORTS
        },
        "formal_freeze_created": False,
    }


def _verify_recorded_file(path_value: str, expected_sha256: str, label: str) -> Path:
    """核对候选记录的文件路径和哈希。"""
    path = _repo_path(path_value)
    if not path.is_file():
        raise RuntimeError(f"{label}: 文件不存在：{path}")
    actual = sha256_file(path)
    if actual != expected_sha256:
        raise RuntimeError(f"{label}: SHA-256 漂移：{actual} != {expected_sha256}")
    return path


def amend_weight_candidate_after_review(*, review_text_sha256: str) -> dict[str, Any]:
    """只更新人工复核后的 workbook 哈希和锁定状态，不重建任何 panel。"""
    if FORMAL_FREEZE_PATH.exists():
        raise RuntimeError("Weight/Design Panel v1 已冻结，拒绝修改候选")
    if not RELEASE_CANDIDATE_PATH.is_file():
        raise RuntimeError("Weight/Design release candidate 不存在")
    release = json.loads(RELEASE_CANDIDATE_PATH.read_text(encoding="utf-8"))
    if release.get("release_status") != "READY_FOR_REVIEW_SOURCE_READY":
        raise RuntimeError("候选状态不是 READY_FOR_REVIEW_SOURCE_READY")
    prior_candidate_sha256 = sha256_file(RELEASE_CANDIDATE_PATH)
    prior_workbook_sha256 = release["review_workbook"]["sha256"]
    if prior_workbook_sha256 == sha256_file(REVIEW_PATH):
        raise RuntimeError("review workbook 哈希未变化，未检测到获准的文案修正")

    for cohort in COHORTS:
        record = release["cohorts"][cohort]
        _verify_recorded_file(
            record["parquet_path"], record["parquet_sha256"], f"{cohort} Parquet"
        )
        _verify_recorded_file(
            record["manifest_path"], record["manifest_sha256"], f"{cohort} manifest"
        )
    _verify_recorded_file(
        release["data_dictionary"]["path"],
        release["data_dictionary"]["sha256"],
        "data dictionary",
    )
    _verify_recorded_file(
        release["mapping"]["path"], release["mapping"]["sha256"], "source map"
    )
    if release["upstream_frozen_snapshot"] != protected_upstream_snapshot():
        raise RuntimeError("人工复核期间上游 frozen 资产发生漂移")

    release["release_status"] = "READY_TO_FREEZE"
    release["review_workbook"]["sha256"] = sha256_file(REVIEW_PATH)
    release["design_register"]["full_file_sha256"] = sha256_file(
        DESIGN_REGISTER_PATH
    )
    release["design_register"]["design_subset_sha256"] = design_subset_hash()
    release["design_register"].pop("source_decisions_pending_human_lock", None)
    release["design_register"]["source_decisions_locked"] = SOURCE_DECISIONS
    release["post_review_metadata_amendment"] = {
        "amended_utc": datetime.now(UTC).isoformat(),
        "prior_candidate_sha256": prior_candidate_sha256,
        "prior_review_workbook_sha256": prior_workbook_sha256,
        "amended_review_workbook_sha256": release["review_workbook"]["sha256"],
        "review_text_sha256": review_text_sha256,
        "scope": (
            "Workbook ID-scope wording and design-register source-lock status only; "
            "no data, source mapping, masking, schema, manifest, count, or missingness change"
        ),
        "clarifications": [
            "HRS stratum/composite cluster are fixed attributes while household ID is wave-specific",
            "ELSA stratum/cluster/household IDs are wave-specific and not assumed stable across waves",
            "KLoSA stratum is fixed, household ID is wave-specific, and cluster remains unavailable",
        ],
        "parquet_rebuilt": False,
        "cohort_manifest_rebuilt": False,
        "data_dictionary_changed": False,
        "numeric_or_formula_change": False,
        "candidate_data_builder_sha256": release["builder_sha256"],
        "post_review_documentation_builder_sha256": sha256_file(Path(__file__)),
        "post_review_workbook_editor_sha256": sha256_file(WORKBOOK_BUILDER_PATH),
    }
    release["next_stage_status"] = "HUMAN_REVIEW_APPROVED_FOR_SOURCE_FREEZE"
    _json_write(RELEASE_CANDIDATE_PATH, release)
    return release


def verify_weight_candidate_for_freeze() -> dict[str, Any]:
    """在写 freeze 前重载候选、来源、上游和七列面板。"""
    if not RELEASE_CANDIDATE_PATH.is_file():
        raise RuntimeError("Weight/Design release candidate 不存在")
    release = json.loads(RELEASE_CANDIDATE_PATH.read_text(encoding="utf-8"))
    if release.get("release_status") != "READY_TO_FREEZE":
        raise RuntimeError("Weight/Design candidate 尚未获准冻结")
    if release.get("model_scaling_and_variance_status") != "PENDING_VALIDATION":
        raise RuntimeError("模型 scaling/variance 状态不得在 source freeze 中改变")
    if release.get("physical_columns") != list(WEIGHT_PANEL_COLUMNS):
        raise RuntimeError("Weight/Design candidate 七列顺序漂移")
    if release["design_register"]["full_file_sha256"] != sha256_file(
        DESIGN_REGISTER_PATH
    ):
        raise RuntimeError("design register 完整哈希漂移")
    if release["design_register"]["design_subset_sha256"] != design_subset_hash():
        raise RuntimeError("weight/design 锁定决策子集漂移")
    _verify_recorded_file(
        release["mapping"]["path"], release["mapping"]["sha256"], "source map"
    )
    _verify_recorded_file(
        release["review_workbook"]["path"],
        release["review_workbook"]["sha256"],
        "review workbook",
    )
    _verify_recorded_file(
        release["data_dictionary"]["path"],
        release["data_dictionary"]["sha256"],
        "data dictionary",
    )
    current_upstream = protected_upstream_snapshot()
    if release["upstream_frozen_snapshot"] != current_upstream:
        raise RuntimeError("上游 frozen snapshot 漂移")

    for cohort in COHORTS:
        record = release["cohorts"][cohort]
        panel_path = _verify_recorded_file(
            record["parquet_path"], record["parquet_sha256"], f"{cohort} Parquet"
        )
        manifest_path = _verify_recorded_file(
            record["manifest_path"], record["manifest_sha256"], f"{cohort} manifest"
        )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        raw_path = Path(manifest["source"]["raw_path"])
        if sha256_file(raw_path) != manifest["source"]["raw_sha256"]:
            raise RuntimeError(f"{cohort}: raw DTA 哈希漂移")
        codebook_path = _repo_path(manifest["source"]["codebook_path"])
        if sha256_file(codebook_path) != manifest["source"]["codebook_sha256"]:
            raise RuntimeError(f"{cohort}: 匹配 codebook 哈希漂移")

        panel = pd.read_parquet(panel_path)
        spine_record = current_upstream["spine_v2"]["cohorts"][cohort]
        spine = pd.read_parquet(
            _repo_path(spine_record["data_path"]), columns=list(KEY_COLUMNS)
        )
        if tuple(panel.columns) != WEIGHT_PANEL_COLUMNS:
            raise RuntimeError(f"{cohort}: 七列 schema 漂移")
        if len(panel) != EXPECTED_SPINE_ROWS[cohort] or len(panel) != len(spine):
            raise RuntimeError(f"{cohort}: 行数漂移")
        for column, dtype in WEIGHT_PANEL_DTYPES.items():
            if str(panel[column].dtype) != dtype:
                raise RuntimeError(
                    f"{cohort} {column}: dtype 漂移 {panel[column].dtype} != {dtype}"
                )
        if not panel[list(KEY_COLUMNS)].equals(spine[list(KEY_COLUMNS)]):
            raise RuntimeError(f"{cohort}: Full Spine 键或行序漂移")
        if panel["person_weight_raw"].lt(0).fillna(False).any():
            raise RuntimeError(f"{cohort}: 出现负权重")
        if cohort == "KLoSA" and panel["survey_cluster_id"].notna().any():
            raise RuntimeError("KLoSA survey_cluster_id 必须全为 NA")
    return release


def freeze_weight_design_panel_v1(
    *, review_text_sha256: str, test_summary: dict[str, Any]
) -> dict[str, Any]:
    """引用获准候选的精确哈希写入唯一 source freeze，不重建数据。"""
    if FORMAL_FREEZE_PATH.exists():
        raise RuntimeError("Weight/Design Panel v1 freeze 已存在，拒绝覆盖")
    release = verify_weight_candidate_for_freeze()
    if release["post_review_metadata_amendment"]["review_text_sha256"] != review_text_sha256:
        raise RuntimeError("人工复核文本哈希与候选记录不一致")
    freeze = {
        "freeze_record_schema_version": "weight_design_panel_freeze_record_v1",
        "freeze_status": "FROZEN",
        "source_readiness_status": "SOURCE_READY",
        "model_scaling_and_variance_status": "PENDING_VALIDATION",
        "release_version": RELEASE_VERSION,
        "generated_utc": datetime.now(UTC).isoformat(),
        "source_release_candidate": {
            "path": _relative(RELEASE_CANDIDATE_PATH),
            "sha256": sha256_file(RELEASE_CANDIDATE_PATH),
            "release_status": release["release_status"],
            "prior_to_review_amendment_sha256": release[
                "post_review_metadata_amendment"
            ]["prior_candidate_sha256"],
        },
        "git_commit": release["git_commit"],
        "git_dirty": release["git_dirty"],
        "git_diff_sha256_at_candidate_build": release["git_diff_sha256"],
        "runner_version": RUNNER_VERSION,
        "candidate_builder_version": release["builder_version"],
        "candidate_builder_sha256": release["builder_sha256"],
        "post_review_builder_version": BUILDER_VERSION,
        "post_review_builder_sha256": sha256_file(Path(__file__)),
        "workbook_builder_sha256_at_candidate_build": release[
            "workbook_builder_sha256"
        ],
        "post_review_workbook_editor_sha256": sha256_file(WORKBOOK_BUILDER_PATH),
        "spec_version": release["spec_version"],
        "specification_sha256": release["specification_sha256"],
        "mapping": release["mapping"],
        "design_register": release["design_register"],
        "common_key": release["common_key"],
        "physical_columns": release["physical_columns"],
        "common_schema_contract": release["common_schema_contract"],
        "cohorts": release["cohorts"],
        "review_workbook": release["review_workbook"],
        "data_dictionary": release["data_dictionary"],
        "human_review": {
            "authorization": (
                "User approved source freezing after workbook review, conditional only "
                "on correcting the HRS/KLoSA ID-scope summary wording"
            ),
            "review_text_sha256": review_text_sha256,
            "review_outcome": "Scientific source structure approved for SOURCE READY freeze",
        },
        "post_review_metadata_amendment": release[
            "post_review_metadata_amendment"
        ],
        "synthetic_weight_semantics": release["synthetic_weight_semantics"],
        "source_decisions_locked": list(SOURCE_DECISIONS),
        "model_decisions_remaining_unlocked": list(PENDING_MODEL_DECISIONS),
        "upstream_frozen_snapshot": release["upstream_frozen_snapshot"],
        "release_integrity": {
            "candidate_and_all_listed_asset_hashes_verified": True,
            "raw_dta_and_matched_codebook_hashes_verified": True,
            "full_spine_key_set_and_order_verified": True,
            "seven_column_schema_and_dtypes_verified": True,
            "zero_missing_and_klosa_cluster_rules_verified": True,
            "protected_upstream_hashes_unchanged": True,
            "parquet_rebuilt_for_freeze": False,
            "cohort_manifest_rebuilt_for_freeze": False,
            "downstream_assets_created": False,
        },
        "pre_freeze_validation": test_summary,
        "overwrite_policy": (
            "Weight/Design Panel v1, its manifests, review workbook, dictionary, "
            "release candidate, and this freeze record are immutable after this freeze; "
            "later source-definition or structural changes require a new version."
        ),
        "downstream_boundary": (
            "Landmark ledger, eligibility-final, analysis panel, real-data models, and meta "
            "remain unauthorized. Mean-one scaling, WeMix final weight levels, and household/"
            "survey-design variance handling remain unlocked."
        ),
        "next_stage_status": "WEIGHTED_MODEL_SCALING_AND_VARIANCE_METHOD_NOT_YET_LOCKED",
    }
    _json_write(FORMAL_FREEZE_PATH, freeze)
    return freeze


__all__ = [
    "BUILDER_VERSION",
    "COHORT_SPECS",
    "FORMAL_FREEZE_PATH",
    "RELEASE_CANDIDATE_PATH",
    "REVIEW_PATH",
    "WEIGHT_PANEL_COLUMNS",
    "WEIGHT_PANEL_DTYPES",
    "WEIGHT_PANEL_NULLABLE",
    "build_weight_design_panel",
    "amend_weight_candidate_after_review",
    "freeze_weight_design_panel_v1",
    "run_weight_design_panel_release",
    "select_landmark_weight",
    "specification",
    "specification_hash",
    "validate_weight_design_panel",
    "verify_weight_candidate_for_freeze",
    "verify_candidate_design",
    "verify_variable_map_contract",
]
