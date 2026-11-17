"""三队列 Covariate Panel v1：person-wave 测量、QC 与候选发布。"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import pandas as pd
from pandas.io.stata import StataMissingValue

from panel_core import (
    RUNNER_VERSION,
    normalise_person_id,
    resolve_cohort_dataset_path,
    sha256_file,
)
from wealth_panel import HRS_CPI_SOURCE_URL, HRS_CPI_U_ANNUAL


ROOT = Path(__file__).resolve().parents[2]
ADMIN = ROOT / "00_admin"
REGISTRY_PATH = ADMIN / "cohort_registry.csv"
CODEBOOK_MANIFEST_PATH = ADMIN / "codebooks" / "codebook_manifest.csv"
DESIGN_REGISTER_PATH = ADMIN / "panel_design_register.csv"
VARIABLE_MAP_PATH = ADMIN / "variable_maps" / "covariate_variable_map.csv"
MANIFEST_ROOT = ADMIN / "panel_manifests"
COVARIATE_MANIFEST_ROOT = MANIFEST_ROOT / "covariate_panel"
DERIVED_ROOT = ROOT / "01_data" / "derived" / "panel_components"
WEALTH_FREEZE_PATH = MANIFEST_ROOT / "wealth_panel" / "wealth_panel_v1_freeze.json"
EXPOSURE_FREEZE_PATH = MANIFEST_ROOT / "exposure_panel" / "exposure_panel_v1_freeze.json"
REVIEW_PATH = ROOT / "04_collaboration" / "covariate_panel_review_v1.xlsx"
DICTIONARY_PATH = COVARIATE_MANIFEST_ROOT / "covariate_panel_data_dictionary.md"
RELEASE_CANDIDATE_PATH = (
    COVARIATE_MANIFEST_ROOT / "covariate_panel_v1_release_candidate.json"
)
FORMAL_FREEZE_PATH = COVARIATE_MANIFEST_ROOT / "covariate_panel_v1_freeze.json"

COHORTS = ("HRS", "ELSA", "KLoSA")
RELEASE_VERSION = "v1"
BUILDER_VERSION = "covariate_panel_builder_v1.0.1"
MAP_VERSION = "covariate_variable_map_v1"
SPEC_VERSION = "covariate_measurement_spec_v1"

KEY_COLUMNS = ("cohort", "person_id", "wave")
COVARIATE_PANEL_COLUMNS = (
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
)
COVARIATE_PANEL_DTYPES = {
    "cohort": "string",
    "person_id": "string",
    "wave": "int16",
    "age": "float64",
    "sex": "string",
    "education": "string",
    "marital_status": "string",
    "household_size": "Int16",
    "labour_force_status": "string",
    "household_income_rank": "float64",
    "current_smoking": "boolean",
    "wealth_quintile": "Int8",
}
COVARIATE_PANEL_NULLABLE = {
    "cohort": False,
    "person_id": False,
    "wave": False,
    "age": True,
    "sex": True,
    "education": True,
    "marital_status": True,
    "household_size": True,
    "labour_force_status": True,
    "household_income_rank": True,
    "current_smoking": True,
    "wealth_quintile": True,
}

ANALYTIC_TYPES = {
    "cohort": "identifier",
    "person_id": "identifier",
    "wave": "time_index",
    "age": "continuous",
    "sex": "categorical",
    "education": "categorical_ordered",
    "marital_status": "categorical",
    "household_size": "continuous_count",
    "labour_force_status": "categorical",
    "household_income_rank": "continuous_rank",
    "current_smoking": "binary",
    "wealth_quintile": "categorical_ordered",
}

EXPECTED_SPINE_ROWS = {"HRS": 723_744, "ELSA": 216_790, "KLoSA": 100_566}
EXPECTED_INDEX_COUNTS = {"HRS": 8_204, "ELSA": 1_284, "KLoSA": 3_125}

SEX_LABELS = {1: "Male", 2: "Female"}
EDUCATION_LABELS = {1: "Low", 2: "Medium", 3: "High"}
MARITAL_LABELS = {
    "partnered": "Married/partnered",
    "separated": "Separated/divorced",
    "widowed": "Widowed",
    "never": "Never married",
}
LABOUR_LABELS = {
    "working": "Working",
    "unemployed": "Unemployed",
    "retired": "Retired or partly retired",
    "inactive": "Other economically inactive",
}

# Gateway 文件把 Stata tagged missing 物理存成固定负码；这些只适用于
# 分类/计数/年龄字段，绝不能套到允许真实负值的连续收入。
GATEWAY_SPECIAL_MISSING_CODES = {-4, -13, -15, -16, -18}
GATEWAY_SPECIAL_MISSING_LABELS = {
    -4: ".d",
    -13: ".m",
    -15: ".o",
    -16: ".p",
    -18: ".r",
}

# 收入使用独立的参考期语义；1991 是 HRS W1 上一日历年所需 CPI-U。
HRS_INCOME_CPI_U_ANNUAL = {1991: 136.2, **HRS_CPI_U_ANNUAL}
KLOSA_INCOME_REFERENCE_YEAR = {
    1: 2005,
    2: 2007,
    3: 2009,
    4: 2011,
    5: 2013,
    6: 2015,
    7: 2017,
    8: 2019,
    9: 2021,
}


@dataclass(frozen=True)
class CohortCovariateSpec:
    """单队列来源字段和协调规则。"""

    cohort: str
    wave_min: int
    wave_max: int
    person_id_source: str
    age_template: str
    education_source: str
    marital_template: str
    household_size_template: str
    labour_template: str
    income_template: str
    weight_template: str
    smoking_template: str
    income_base_year: int
    income_reference_unit: str
    age_definition: str
    education_high_definition: str


COHORT_SPECS = {
    "HRS": CohortCovariateSpec(
        cohort="HRS",
        wave_min=1,
        wave_max=16,
        person_id_source="hhidpn",
        age_template="r{wave}agey_m",
        education_source="raeduc",
        marital_template="r{wave}mstat",
        household_size_template="h{wave}hhres",
        labour_template="r{wave}lbrf",
        income_template="h{wave}itot",
        weight_template="r{wave}wtcrnh",
        smoking_template="r{wave}smoken",
        income_base_year=2022,
        income_reference_unit="respondent-and-spouse previous-calendar-year income",
        age_definition="integer age at the midpoint between interview start and end dates",
        education_high_definition="some college or above",
    ),
    "ELSA": CohortCovariateSpec(
        cohort="ELSA",
        wave_min=1,
        wave_max=10,
        person_id_source="idauniq",
        age_template="r{wave}agey",
        education_source="raeducl",
        marital_template="r{wave}mstat",
        household_size_template="h{wave}hhres",
        labour_template="r{wave}lbrf_e",
        income_template="h{wave}itot",
        weight_template="r{wave}wtresp",
        smoking_template="r{wave}smoken",
        income_base_year=2020,
        income_reference_unit="benefit-unit annualised nominal income flows",
        age_definition="age in completed years, top-coded at 90 (90 means 90 or older)",
        education_high_definition="high education level in harmonised RAEDUCL",
    ),
    "KLoSA": CohortCovariateSpec(
        cohort="KLoSA",
        wave_min=1,
        wave_max=9,
        person_id_source="pid",
        age_template="r{wave}agey",
        education_source="raeducl",
        marital_template="r{wave}mstat",
        household_size_template="h{wave}hhres",
        labour_template="r{wave}lbrf_k",
        income_template="r{wave}itothhinc",
        weight_template="r{wave}wtresp",
        smoking_template="r{wave}smoken",
        income_base_year=2022,
        income_reference_unit="past-year whole-household income after tax and deductions",
        age_definition="survey year minus birth year; not birthday-exact completed age",
        education_high_definition="completed two-year college, university, or postgraduate education",
    ),
}

LOCKED_COVARIATE_DESIGN = {
    "covariate_panel_schema": "person_wave_12_columns_v1",
    "covariate_fixed_attributes": "sex_and_education",
    "covariate_education_mapping": "three_level_no_elsa_fallback",
    "covariate_marital_mapping": "four_level_explicit_codes",
    "covariate_labour_mapping": "four_level_partly_retired_explicit",
    "income_rank_definition": "weighted_within_cohort_wave_midrank",
    "income_price_year_definition": "cohort_specific_income_reference_year",
    "wealth_quintile_definition": "unweighted_wave_midrank_tie_preserving",
}

MAPPING_COLUMNS = (
    "cohort",
    "target_variable",
    "source_variable",
    "source_definition",
    "available_waves",
    "original_coding",
    "harmonised_coding",
    "measurement_level",
    "value_provenance",
    "missing_code_policy",
    "cohort_limitation",
    "harmonisation_status",
    "intended_analysis_time",
    "model_role",
    "analytic_type",
    "map_version",
)

# 锁住完整 CSV 记录（而不只是目标变量名），避免说明文件与可执行来源/编码漂移。
VARIABLE_MAP_RECORDS_SHA256 = (
    "8fb94533759d08dc55344565a6a160ed0e511753384a77b956ac548492baec75"
)


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


def _tag_counts(values: pd.Series) -> dict[str, int]:
    tags = values.loc[values.map(_is_stata_missing)].map(str)
    return {str(key): int(value) for key, value in tags.value_counts().items()}


def _gateway_special_counts(values: pd.Series) -> dict[str, int]:
    """统计 Gateway 数值特殊缺失；不得用于允许真实负值的收入字段。"""
    tagged = values.map(_is_stata_missing).astype(bool)
    numeric = pd.to_numeric(values.mask(tagged), errors="coerce")
    special = numeric.loc[numeric.isin(GATEWAY_SPECIAL_MISSING_CODES)].astype(int)
    counts = special.value_counts().sort_index()
    return {
        f"{int(code)} ({GATEWAY_SPECIAL_MISSING_LABELS[int(code)]})": int(count)
        for code, count in counts.items()
    }


def _numeric_series(values: pd.Series, variable: str) -> pd.Series:
    tagged = values.map(_is_stata_missing).astype(bool)
    clean = values.mask(tagged)
    numeric = pd.to_numeric(clean, errors="coerce")
    invalid = clean.notna() & numeric.isna()
    if invalid.any():
        examples = clean.loc[invalid].astype(str).drop_duplicates().head(8).tolist()
        raise RuntimeError(f"{variable}: 出现无法解释的非数值：{examples}")
    return numeric.astype("float64")


def _structural_numeric(values: pd.Series, variable: str) -> pd.Series:
    """把已核实的 Gateway 特殊负码转为 NA；保留其他数值供范围检查。"""
    numeric = _numeric_series(values, variable)
    return numeric.mask(numeric.isin(GATEWAY_SPECIAL_MISSING_CODES))


def _integer_codes(values: pd.Series, variable: str) -> pd.Series:
    numeric = _structural_numeric(values, variable)
    finite = numeric.notna()
    noninteger = finite & numeric.mod(1).ne(0)
    if noninteger.any():
        examples = numeric.loc[noninteger].drop_duplicates().head(8).tolist()
        raise RuntimeError(f"{variable}: 出现非整数代码：{examples}")
    return numeric.astype("Int16")


def _map_codes(
    values: pd.Series,
    mapping: dict[int, str],
    variable: str,
) -> pd.Series:
    codes = _integer_codes(values, variable)
    unknown = sorted(set(codes.dropna().astype(int)).difference(mapping))
    if unknown:
        raise RuntimeError(f"{variable}: 出现未知实质代码：{unknown}")
    result = codes.map(mapping)
    return result.astype("string")


def _education_mapping(cohort: str) -> dict[int, str]:
    if cohort == "HRS":
        return {1: "Low", 2: "Medium", 3: "Medium", 4: "High", 5: "High"}
    return EDUCATION_LABELS


def _marital_mapping(cohort: str) -> dict[int, str]:
    if cohort == "HRS":
        return {
            1: MARITAL_LABELS["partnered"],
            2: MARITAL_LABELS["partnered"],
            3: MARITAL_LABELS["partnered"],
            4: MARITAL_LABELS["separated"],
            5: MARITAL_LABELS["separated"],
            6: MARITAL_LABELS["separated"],
            7: MARITAL_LABELS["widowed"],
            8: MARITAL_LABELS["never"],
        }
    return {
        1: MARITAL_LABELS["partnered"],
        3: MARITAL_LABELS["partnered"],
        4: MARITAL_LABELS["separated"],
        5: MARITAL_LABELS["separated"],
        7: MARITAL_LABELS["widowed"],
        8: MARITAL_LABELS["never"],
    }


def _labour_mapping(cohort: str) -> dict[int, str]:
    if cohort == "KLoSA":
        return {
            1: LABOUR_LABELS["working"],
            2: LABOUR_LABELS["working"],
            3: LABOUR_LABELS["working"],
            4: LABOUR_LABELS["working"],
            5: LABOUR_LABELS["unemployed"],
            6: LABOUR_LABELS["retired"],
            7: LABOUR_LABELS["retired"],
            8: LABOUR_LABELS["inactive"],
            9: LABOUR_LABELS["inactive"],
        }
    return {
        1: LABOUR_LABELS["working"],
        2: LABOUR_LABELS["working"],
        3: LABOUR_LABELS["unemployed"],
        4: LABOUR_LABELS["retired"],
        5: LABOUR_LABELS["retired"],
        6: LABOUR_LABELS["inactive"],
        7: LABOUR_LABELS["inactive"],
    }


def income_price_years(spine: pd.DataFrame, cohort: str) -> tuple[pd.Series, pd.Series]:
    """按收入参考期而非财富存量时点生成价格年份。"""
    observed = spine["observed_in_wave"].eq(True).fillna(False)
    interview = pd.to_numeric(spine["interview_year"], errors="coerce")
    nominal = pd.to_numeric(spine["nominal_year"], errors="coerce")
    if cohort == "HRS":
        price_year = interview.sub(1).where(interview.notna(), nominal.sub(1))
        source = pd.Series(pd.NA, index=spine.index, dtype="string")
        source.loc[observed & interview.notna()] = "interview_year_minus_1"
        source.loc[observed & interview.isna() & nominal.notna()] = (
            "nominal_year_minus_1_fallback"
        )
    elif cohort == "ELSA":
        price_year = interview.where(interview.notna(), nominal)
        source = pd.Series(pd.NA, index=spine.index, dtype="string")
        source.loc[observed & interview.notna()] = "interview_year"
        source.loc[observed & interview.isna() & nominal.notna()] = (
            "nominal_year_fallback"
        )
    elif cohort == "KLoSA":
        price_year = spine["wave"].map(KLOSA_INCOME_REFERENCE_YEAR)
        source = pd.Series(pd.NA, index=spine.index, dtype="string")
        source.loc[observed] = "locked_wave_reference_year"
    else:
        raise KeyError(cohort)
    missing = observed & price_year.isna()
    if missing.any():
        raise RuntimeError(f"{cohort}: 访谈行缺少收入 price year")
    noninteger = observed & price_year.notna() & price_year.mod(1).ne(0)
    if noninteger.any():
        raise RuntimeError(f"{cohort}: 收入 price year 出现非整数")
    return price_year.astype("Int16"), source


def weighted_midrank(
    target_values: pd.Series | Iterable[float],
    reference_values: pd.Series | Iterable[float],
    reference_weights: pd.Series | Iterable[float],
) -> pd.Series:
    """把目标值映射到加权参考分布；并列值共享同一 fractional rank。"""
    target = pd.Series(target_values, dtype="float64")
    ref = pd.DataFrame(
        {
            "value": pd.Series(reference_values, dtype="float64"),
            "weight": pd.Series(reference_weights, dtype="float64"),
        }
    )
    valid = (
        ref["value"].notna()
        & np.isfinite(ref["value"])
        & ref["weight"].notna()
        & np.isfinite(ref["weight"])
        & ref["weight"].gt(0)
    )
    ref = ref.loc[valid]
    if ref.empty:
        raise RuntimeError("收入 rank 没有有效的正权重参考分布")
    grouped = ref.groupby("value", sort=True, observed=True)["weight"].sum()
    values = grouped.index.to_numpy(dtype=float)
    weights = grouped.to_numpy(dtype=float)
    total = float(weights.sum())
    if not np.isfinite(total) or total <= 0:
        raise RuntimeError("收入 rank 参考权重总和无效")
    cumulative = np.cumsum(weights)
    midranks = (cumulative - 0.5 * weights) / total

    result = pd.Series(np.nan, index=target.index, dtype="float64")
    target_valid = target.notna() & np.isfinite(target)
    query = target.loc[target_valid].to_numpy(dtype=float)
    insertion = np.searchsorted(values, query, side="left")
    mapped = np.empty(len(query), dtype=float)
    for index, (value, position) in enumerate(zip(query, insertion, strict=True)):
        if position < len(values) and values[position] == value:
            mapped[index] = midranks[position]
        elif position == 0:
            mapped[index] = 0.0
        elif position == len(values):
            mapped[index] = 1.0
        else:
            mapped[index] = cumulative[position - 1] / total
    result.loc[target_valid] = mapped
    return result


def wealth_quintile_from_midrank(
    values: pd.Series | Iterable[float],
) -> pd.Series:
    """用 (average_rank - 0.5) / N 构建不拆并列值的样本财富五分位。"""
    source = pd.Series(values, dtype="float64")
    valid = source.notna() & np.isfinite(source)
    result = pd.Series(pd.NA, index=source.index, dtype="Int8")
    if not valid.any():
        return result
    selected = source.loc[valid]
    average_rank = selected.rank(method="average", ascending=True)
    midrank = (average_rank - 0.5) / len(selected)
    quintile = np.ceil(5.0 * midrank).clip(1, 5).astype("int8")
    result.loc[valid] = quintile.to_numpy()
    return result


def specification() -> dict[str, Any]:
    return {
        "spec_version": SPEC_VERSION,
        "grain": "one row per cohort + person_id + wave",
        "physical_columns": list(COVARIATE_PANEL_COLUMNS),
        "analytic_types": ANALYTIC_TYPES,
        "fixed_attributes": ["sex", "education"],
        "noninterview_policy": (
            "sex and education may remain populated; all other substantive covariates are NA"
        ),
        "project_imputation": False,
        "official_harmonised_imputation_values_retained": True,
        "future_person_level_backfill": False,
        "income_rank": {
            "formula": (
                "(positive reference weight below income + half positive reference "
                "weight tied at income) / total positive reference weight"
            ),
            "reference": (
                "within cohort and wave; observed respondents of all ages with valid "
                "income and positive cross-sectional personal weight"
            ),
            "target": "all observed respondents with valid income, regardless of own weight",
            "household_deduplication": False,
            "equivalisation": False,
            "future_model_weight": False,
        },
        "wealth_quintile": {
            "formula": "ceil(5 * ((average rank - 0.5) / N))",
            "reference": (
                "within cohort and wave; all observed nonmissing frozen Wealth Panel values"
            ),
            "survey_weighted": False,
            "ties_split": False,
            "zero_and_negative_included": True,
        },
        "cohorts": {
            cohort: {
                "wave_range": [spec.wave_min, spec.wave_max],
                "age_source": spec.age_template,
                "age_definition": spec.age_definition,
                "education_source": spec.education_source,
                "education_high_definition": spec.education_high_definition,
                "income_source": spec.income_template,
                "income_weight": spec.weight_template,
                "income_base_year": spec.income_base_year,
                "income_reference_unit": spec.income_reference_unit,
            }
            for cohort, spec in COHORT_SPECS.items()
        },
    }


def specification_hash() -> str:
    return _stable_hash(specification())


def _mapping_records() -> list[dict[str, str]]:
    """从已登记 CSV 返回 mapping；可执行契约由 CSV 与规范共同约束。"""
    return _read_csv_rows(VARIABLE_MAP_PATH)


def verify_variable_map_contract() -> list[dict[str, str]]:
    with VARIABLE_MAP_PATH.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != MAPPING_COLUMNS:
            raise RuntimeError("covariate_variable_map.csv 列契约不一致")
        rows = list(reader)
    if len(rows) != 27:
        raise RuntimeError("covariate_variable_map.csv 必须为 3 队列 × 9 协变量")
    keys = [(row["cohort"], row["target_variable"]) for row in rows]
    if len(keys) != len(set(keys)):
        raise RuntimeError("covariate_variable_map.csv 队列×变量键不唯一")
    if _stable_hash(rows) != VARIABLE_MAP_RECORDS_SHA256:
        raise RuntimeError(
            "covariate_variable_map.csv 完整记录与可执行 Covariate mapping 不一致"
        )
    expected_targets = set(COVARIATE_PANEL_COLUMNS[3:])
    for cohort in COHORTS:
        actual = {row["target_variable"] for row in rows if row["cohort"] == cohort}
        if actual != expected_targets:
            raise RuntimeError(f"{cohort}: covariate mapping 目标变量不完整")
    for row in rows:
        spec = COHORT_SPECS[row["cohort"]]
        target = row["target_variable"]
        expected_sources = {
            "age": spec.age_template,
            "sex": "ragender",
            "education": spec.education_source,
            "marital_status": spec.marital_template,
            "household_size": spec.household_size_template,
            "labour_force_status": spec.labour_template,
            "household_income_rank": f"{spec.income_template}; {spec.weight_template}",
            "current_smoking": spec.smoking_template,
            "wealth_quintile": "wealth_total_real_equiv from frozen Wealth Panel v1",
        }
        if row["source_variable"] != expected_sources[target]:
            raise RuntimeError(
                f"{row['cohort']} {target}: source_variable 与可执行 spec 不一致"
            )
        expected_waves = f"W{spec.wave_min}-W{spec.wave_max}"
        if row["available_waves"] != expected_waves:
            raise RuntimeError(f"{row['cohort']} {target}: available_waves 不一致")
        expected_level = (
            "person-level fixed" if target in {"sex", "education"} else "person-wave"
        )
        if row["measurement_level"] != expected_level:
            raise RuntimeError(f"{row['cohort']} {target}: measurement_level 不一致")
        expected_time = "t-1" if target == "wealth_quintile" else "t"
        if row["intended_analysis_time"] != expected_time:
            raise RuntimeError(f"{row['cohort']} {target}: intended_analysis_time 不一致")
        if row["analytic_type"] != ANALYTIC_TYPES[target]:
            raise RuntimeError(f"{row['cohort']} {target}: analytic_type 不一致")
        if row["map_version"] != MAP_VERSION:
            raise RuntimeError("covariate_variable_map.csv map_version 不一致")
    elsa_education = next(
        row for row in rows if row["cohort"] == "ELSA" and row["target_variable"] == "education"
    )
    if elsa_education["source_variable"] != "raeducl" or "raedyrs_e" in json.dumps(
        elsa_education, ensure_ascii=False
    ).lower():
        raise RuntimeError("ELSA education 必须只使用 raeducl，禁止 fallback")
    return rows


def verify_locked_design() -> dict[str, dict[str, str]]:
    rows = {row["decision_id"]: row for row in _read_csv_rows(DESIGN_REGISTER_PATH)}
    resolved: dict[str, dict[str, str]] = {}
    for decision_id, selected in LOCKED_COVARIATE_DESIGN.items():
        row = rows.get(decision_id)
        if row is None:
            raise RuntimeError(f"design register 缺少 {decision_id}")
        if row["status"] != "locked" or row["selected_value"] != selected:
            raise RuntimeError(
                f"{decision_id} 未按 Covariate Panel v1 锁定："
                f"status={row['status']} selected={row['selected_value']}"
            )
        resolved[decision_id] = row
    return resolved


def covariate_design_subset_hash() -> str:
    rows = verify_locked_design()
    return _stable_hash([rows[key] for key in LOCKED_COVARIATE_DESIGN])


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
        if row["cohort"] == cohort
    ]
    if len(rows) != 1 or rows[0]["usage_status"] != "authoritative":
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
        (MANIFEST_ROOT / "spine" / "spine_v2_freeze.json").read_text(encoding="utf-8")
    )
    frozen = spine_freeze["cohorts"][cohort]
    if raw_sha != frozen["source_dataset_sha256"]:
        raise RuntimeError(f"{cohort}: raw 哈希与冻结 Spine 来源不一致")
    if codebook_sha != codebook["sha256"] or codebook_sha != frozen["source_codebook_sha256"]:
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


def required_source_variables(cohort: str, spine: pd.DataFrame) -> list[str]:
    spec = COHORT_SPECS[cohort]
    variables = [spec.person_id_source, "ragender", spec.education_source]
    for wave in range(spec.wave_min, spec.wave_max + 1):
        variables.extend(
            [
                spec.age_template.format(wave=wave),
                spec.marital_template.format(wave=wave),
                spec.household_size_template.format(wave=wave),
                spec.labour_template.format(wave=wave),
                spec.income_template.format(wave=wave),
                spec.weight_template.format(wave=wave),
                spec.smoking_template.format(wave=wave),
            ]
        )
    if cohort != "HRS":
        price_year, _source = income_price_years(spine, cohort)
        years = sorted(
            set(price_year.dropna().astype(int).tolist()).union({spec.income_base_year})
        )
        variables.extend(f"c{year}cpindex" for year in years)
    return list(dict.fromkeys(variables))


def read_raw_source(source_path: Path, cohort: str, spine: pd.DataFrame) -> pd.DataFrame:
    return pd.read_stata(
        source_path,
        columns=required_source_variables(cohort, spine),
        convert_categoricals=False,
        convert_missing=True,
        preserve_dtypes=True,
    )


def extract_income_price_index(
    raw: pd.DataFrame, spine: pd.DataFrame, cohort: str
) -> tuple[dict[int, float], dict[str, Any]]:
    spec = COHORT_SPECS[cohort]
    price_year, price_source = income_price_years(spine, cohort)
    observed = spine["observed_in_wave"].eq(True).fillna(False)
    years = sorted(
        set(price_year.loc[observed].astype(int).tolist()).union({spec.income_base_year})
    )
    if cohort == "HRS":
        missing = sorted(set(years).difference(HRS_INCOME_CPI_U_ANNUAL))
        if missing:
            raise RuntimeError(f"HRS: 收入 price year 无 CPI-U 支持：{missing}")
        index = {year: float(HRS_INCOME_CPI_U_ANNUAL[year]) for year in years}
        source = HRS_CPI_SOURCE_URL
    else:
        index: dict[int, float] = {}
        for year in years:
            variable = f"c{year}cpindex"
            numeric = _numeric_series(raw[variable], variable).dropna().unique()
            if len(numeric) != 1 or not np.isfinite(numeric[0]) or numeric[0] <= 0:
                raise RuntimeError(f"{cohort}: {variable} 必须恰有一个正有限值")
            index[year] = float(numeric[0])
        source = "harmonised source cYYYYcpindex (World Bank WDI; 2010=100)"
    return index, {
        "source": source,
        "base_year": spec.income_base_year,
        "values": {str(year): index[year] for year in sorted(index)},
        "price_year_source_counts": {
            key: int(value)
            for key, value in Counter(price_source.loc[observed].dropna().astype(str)).items()
        },
        "config_sha256": _stable_hash(index),
    }


def _source_long(raw: pd.DataFrame, cohort: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    spec = COHORT_SPECS[cohort]
    person_id = normalise_person_id(raw[spec.person_id_source], cohort)
    if person_id.duplicated().any():
        raise RuntimeError(f"{cohort}: raw person ID 不唯一")
    fixed = pd.DataFrame(
        {
            "person_id": person_id,
            "sex": _map_codes(raw["ragender"], SEX_LABELS, "ragender"),
            "education": _map_codes(
                raw[spec.education_source],
                _education_mapping(cohort),
                spec.education_source,
            ),
        }
    )
    tagged: dict[str, Counter[str]] = {
        "sex": Counter(),
        "education": Counter(),
        "age": Counter(),
        "marital_status": Counter(),
        "household_size": Counter(),
        "labour_force_status": Counter(),
        "income": Counter(),
        "income_weight": Counter(),
        "current_smoking": Counter(),
    }
    tagged["sex"].update(_tag_counts(raw["ragender"]))
    tagged["education"].update(_tag_counts(raw[spec.education_source]))
    gateway_special: dict[str, Counter[str]] = {
        group: Counter()
        for group in (
            "sex",
            "education",
            "age",
            "marital_status",
            "household_size",
            "labour_force_status",
            "current_smoking",
        )
    }
    if cohort != "HRS":
        gateway_special["sex"].update(_gateway_special_counts(raw["ragender"]))
        gateway_special["education"].update(
            _gateway_special_counts(raw[spec.education_source])
        )
    frames: list[pd.DataFrame] = []
    for wave in range(spec.wave_min, spec.wave_max + 1):
        names = {
            "age": spec.age_template.format(wave=wave),
            "marital_status": spec.marital_template.format(wave=wave),
            "household_size": spec.household_size_template.format(wave=wave),
            "labour_force_status": spec.labour_template.format(wave=wave),
            "income": spec.income_template.format(wave=wave),
            "income_weight": spec.weight_template.format(wave=wave),
            "current_smoking": spec.smoking_template.format(wave=wave),
        }
        for group, variable in names.items():
            tagged[group].update(_tag_counts(raw[variable]))
            if cohort != "HRS" and group not in {"income", "income_weight"}:
                gateway_special[group].update(_gateway_special_counts(raw[variable]))
        household = _structural_numeric(
            raw[names["household_size"]], names["household_size"]
        )
        noninteger = household.notna() & household.mod(1).ne(0)
        if noninteger.any():
            raise RuntimeError(f"{cohort} W{wave}: household size 出现非整数")
        smoking_codes = _integer_codes(raw[names["current_smoking"]], names["current_smoking"])
        smoking_unknown = sorted(set(smoking_codes.dropna().astype(int)).difference({0, 1}))
        if smoking_unknown:
            raise RuntimeError(
                f"{cohort} W{wave}: current smoking 出现未知代码 {smoking_unknown}"
            )
        smoking = smoking_codes.map({0: False, 1: True}).astype("boolean")
        frame = pd.DataFrame(
            {
                "person_id": person_id,
                "wave": pd.Series(wave, index=raw.index, dtype="int16"),
                "age": _structural_numeric(raw[names["age"]], names["age"]),
                "marital_status": _map_codes(
                    raw[names["marital_status"]],
                    _marital_mapping(cohort),
                    names["marital_status"],
                ),
                "household_size": household.astype("Int16"),
                "labour_force_status": _map_codes(
                    raw[names["labour_force_status"]],
                    _labour_mapping(cohort),
                    names["labour_force_status"],
                ),
                "income_nominal": _numeric_series(raw[names["income"]], names["income"]),
                "income_weight": _numeric_series(
                    raw[names["income_weight"]], names["income_weight"]
                ),
                "current_smoking": smoking,
            }
        )
        frames.append(frame)
    long = pd.concat(frames, ignore_index=True)
    long = long.merge(fixed, on="person_id", how="left", validate="many_to_one")
    return long, {
        "tagged_missing_counts": {
            group: dict(sorted(counter.items())) for group, counter in tagged.items()
        },
        "gateway_special_missing_counts": {
            group: dict(sorted(counter.items()))
            for group, counter in gateway_special.items()
        },
        "gateway_special_missing_policy": (
            "reported only for structural age/category/count fields; continuous "
            "income and income weights retain genuine negative values"
        ),
        "fixed_attribute_missing_n": {
            "sex": int(fixed["sex"].isna().sum()),
            "education": int(fixed["education"].isna().sum()),
        },
        "elsa_education_fallback_used": False,
        "project_imputation_performed": False,
        "official_harmonised_values_retained": True,
    }


def _income_rank_by_wave(
    aligned: pd.DataFrame,
    cohort: str,
    price_index: dict[int, float],
) -> tuple[pd.Series, list[dict[str, Any]]]:
    spec = COHORT_SPECS[cohort]
    observed = aligned["observed_in_wave"].eq(True).fillna(False)
    price_year, price_source = income_price_years(aligned, cohort)
    supported = price_year.loc[observed].astype(int).isin(price_index)
    if not supported.all():
        bad = sorted(price_year.loc[observed].astype(int).loc[~supported].unique())
        raise RuntimeError(f"{cohort}: 收入 price year 无 CPI 支持：{bad}")
    row_index = price_year.map(price_index).astype("Float64")
    income = aligned["income_nominal"].astype("float64")
    valid_income = observed & income.notna() & np.isfinite(income)
    real = pd.Series(np.nan, index=aligned.index, dtype="float64")
    real.loc[valid_income] = (
        income.loc[valid_income]
        * float(price_index[spec.income_base_year])
        / row_index.loc[valid_income].astype(float)
    )
    rank = pd.Series(np.nan, index=aligned.index, dtype="float64")
    records: list[dict[str, Any]] = []
    for wave in range(spec.wave_min, spec.wave_max + 1):
        wave_mask = aligned["wave"].eq(wave)
        target_mask = wave_mask & valid_income
        weight = aligned["income_weight"].astype("float64")
        positive_weight_mask = (
            wave_mask
            & observed
            & weight.notna()
            & np.isfinite(weight)
            & weight.gt(0)
        )
        reference_mask = target_mask & positive_weight_mask
        if not reference_mask.any():
            raise RuntimeError(f"{cohort} W{wave}: 无有效收入 rank 参考分布")
        mapped = weighted_midrank(
            real.loc[target_mask],
            real.loc[reference_mask],
            weight.loc[reference_mask],
        )
        rank.loc[target_mask] = mapped.to_numpy(dtype=float)
        distribution = (
            pd.DataFrame(
                {
                    "price_year": price_year.loc[target_mask].astype(int),
                    "source": price_source.loc[target_mask].astype("string"),
                }
            )
            .value_counts(sort=False)
            .reset_index(name="n")
            .sort_values(["price_year", "source"], kind="stable")
        )
        target_n = int(target_mask.sum())
        price_rows = [
            {
                "income_price_year": int(row.price_year),
                "source": str(row.source),
                "n": int(row.n),
                "percent": 100.0 * int(row.n) / target_n,
            }
            for row in distribution.itertuples(index=False)
        ]
        records.append(
            {
                "cohort": cohort,
                "wave": wave,
                "income_rank_reference_n": int(reference_mask.sum()),
                "income_rank_reference_positive_weight_n": int(
                    positive_weight_mask.sum()
                ),
                "income_rank_reference_weight_sum": float(weight.loc[reference_mask].sum()),
                "income_rank_target_nonmissing_n": target_n,
                "income_price_year_unique_n": len(price_rows),
                "income_price_year_distribution": price_rows,
            }
        )
    rank.loc[~observed] = np.nan
    return rank, records


def _wealth_quintiles_by_wave(
    wealth_panel: pd.DataFrame, spine: pd.DataFrame, cohort: str
) -> tuple[pd.Series, list[dict[str, Any]]]:
    if list(wealth_panel.columns[:3]) != list(KEY_COLUMNS):
        raise RuntimeError(f"{cohort}: Wealth Panel 键列不正确")
    pd.testing.assert_frame_equal(
        wealth_panel.loc[:, list(KEY_COLUMNS)],
        spine.loc[:, list(KEY_COLUMNS)].astype(
            {"cohort": "string", "person_id": "string", "wave": "int16"}
        ),
        check_dtype=True,
        check_exact=True,
    )
    observed = spine["observed_in_wave"].eq(True).fillna(False)
    wealth = wealth_panel["wealth_total_real_equiv"].astype("float64")
    result = pd.Series(pd.NA, index=spine.index, dtype="Int8")
    records: list[dict[str, Any]] = []
    for wave in range(COHORT_SPECS[cohort].wave_min, COHORT_SPECS[cohort].wave_max + 1):
        mask = spine["wave"].eq(wave) & observed & wealth.notna() & np.isfinite(wealth)
        quintile = wealth_quintile_from_midrank(wealth.loc[mask])
        result.loc[mask] = quintile.to_numpy()
        counts = result.loc[mask].value_counts().sort_index()
        records.append(
            {
                "cohort": cohort,
                "wave": wave,
                "reference_n": int(mask.sum()),
                "zero_wealth_n": int((mask & wealth.eq(0)).sum()),
                "negative_wealth_n": int((mask & wealth.lt(0)).sum()),
                "quintile_counts": {str(int(k)): int(v) for k, v in counts.items()},
                "ties_split": False,
                "survey_weighted": False,
            }
        )
    return result, records


def build_covariate_panel(
    raw: pd.DataFrame,
    spine: pd.DataFrame,
    wealth_panel: pd.DataFrame,
    cohort: str,
    price_index: dict[int, float] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """以冻结 Full Spine 的键、行集与顺序构建 12 列测量面板。"""
    if cohort not in COHORTS:
        raise KeyError(cohort)
    if spine.duplicated(list(KEY_COLUMNS)).any():
        raise RuntimeError(f"{cohort}: Spine 主键不唯一")
    source_long, source_qc = _source_long(raw, cohort)
    if set(source_long["person_id"].unique()) != set(spine["person_id"].astype("string").unique()):
        raise RuntimeError(f"{cohort}: raw 与 Spine 人员集合不一致")
    aligned = spine.loc[
        :, [*KEY_COLUMNS, "observed_in_wave", "interview_year", "nominal_year"]
    ].copy()
    aligned["_row_order"] = np.arange(len(aligned), dtype=np.int64)
    aligned = aligned.merge(
        source_long,
        on=["person_id", "wave"],
        how="left",
        sort=False,
        validate="one_to_one",
    ).sort_values("_row_order", kind="stable")
    if len(aligned) != len(spine):
        raise RuntimeError(f"{cohort}: source join 改变了 Spine 行数")
    if price_index is None:
        price_index, price_context = extract_income_price_index(raw, spine, cohort)
    else:
        price_context = {
            "source": "test_or_explicit_price_index",
            "base_year": COHORT_SPECS[cohort].income_base_year,
            "values": {str(k): float(v) for k, v in sorted(price_index.items())},
            "config_sha256": _stable_hash(price_index),
        }
    observed = aligned["observed_in_wave"].eq(True).fillna(False)
    noninterview = ~observed
    time_varying = (
        "age",
        "marital_status",
        "household_size",
        "labour_force_status",
        "current_smoking",
    )
    for column in time_varying:
        aligned.loc[noninterview, column] = pd.NA
    invalid_household = observed & aligned["household_size"].notna() & aligned["household_size"].le(0)
    if invalid_household.any():
        counts = aligned.loc[invalid_household, "household_size"].value_counts().to_dict()
        raise RuntimeError(f"{cohort}: household size 出现零或负值，需人工核查：{counts}")
    income_rank, income_reference = _income_rank_by_wave(aligned, cohort, price_index)
    wealth_quintile, wealth_reference = _wealth_quintiles_by_wave(
        wealth_panel, spine, cohort
    )
    output = spine.loc[:, list(KEY_COLUMNS)].copy().astype(
        {"cohort": "string", "person_id": "string", "wave": "int16"}
    )
    output["age"] = aligned["age"].astype("float64").to_numpy()
    output["sex"] = aligned["sex"].astype("string").array
    output["education"] = aligned["education"].astype("string").array
    output["marital_status"] = aligned["marital_status"].astype("string").array
    output["household_size"] = aligned["household_size"].astype("Int16").array
    output["labour_force_status"] = aligned["labour_force_status"].astype("string").array
    output["household_income_rank"] = income_rank.to_numpy(dtype=float)
    output["current_smoking"] = aligned["current_smoking"].astype("boolean").array
    output["wealth_quintile"] = wealth_quintile.array
    output = output.loc[:, list(COVARIATE_PANEL_COLUMNS)]
    diagnostics = {
        "source_qc": source_qc,
        "price_index": price_context,
        "income_reference_by_wave": income_reference,
        "wealth_reference_by_wave": wealth_reference,
        "household_size_max": (
            int(output["household_size"].max())
            if output["household_size"].notna().any()
            else None
        ),
        "noninterview_policy_applied": True,
        "fixed_attributes": ["sex", "education"],
    }
    return output, diagnostics


def validate_covariate_panel(
    frame: pd.DataFrame, spine: pd.DataFrame, cohort: str
) -> None:
    """验证 12 列 schema、Full Spine 对齐、掩蔽规则和变量范围。"""
    if list(frame.columns) != list(COVARIATE_PANEL_COLUMNS):
        raise RuntimeError(f"{cohort}: Covariate Panel 列顺序不一致")
    actual_dtypes = {column: str(frame[column].dtype) for column in frame.columns}
    if actual_dtypes != COVARIATE_PANEL_DTYPES:
        raise RuntimeError(f"{cohort}: Covariate Panel dtype 不一致：{actual_dtypes}")
    forbidden_nulls = {
        column: int(frame[column].isna().sum())
        for column, nullable in COVARIATE_PANEL_NULLABLE.items()
        if not nullable and frame[column].isna().any()
    }
    if forbidden_nulls:
        raise RuntimeError(f"{cohort}: Covariate Panel 非空键出现缺失：{forbidden_nulls}")
    if len(frame) != len(spine) or frame.duplicated(list(KEY_COLUMNS)).any():
        raise RuntimeError(f"{cohort}: Covariate Panel 行数或主键不正确")
    pd.testing.assert_frame_equal(
        frame.loc[:, list(KEY_COLUMNS)],
        spine.loc[:, list(KEY_COLUMNS)].astype(
            {"cohort": "string", "person_id": "string", "wave": "int16"}
        ),
        check_dtype=True,
        check_exact=True,
    )
    noninterview = ~spine["observed_in_wave"].eq(True).fillna(False)
    must_be_missing = [
        "age",
        "marital_status",
        "household_size",
        "labour_force_status",
        "household_income_rank",
        "current_smoking",
        "wealth_quintile",
    ]
    if not frame.loc[noninterview, must_be_missing].isna().all(axis=None):
        raise RuntimeError(f"{cohort}: 非访谈行出现逐波协变量值")
    for column in ("sex", "education"):
        unique_per_person = frame.groupby("person_id", observed=True)[column].nunique(dropna=True)
        if int(unique_per_person.max()) > 1:
            raise RuntimeError(f"{cohort}: 固定属性 {column} 在个人内不稳定")
    allowed = {
        "sex": set(SEX_LABELS.values()),
        "education": set(EDUCATION_LABELS.values()),
        "marital_status": set(MARITAL_LABELS.values()),
        "labour_force_status": set(LABOUR_LABELS.values()),
    }
    for column, values in allowed.items():
        unexpected = sorted(set(frame[column].dropna().astype(str)).difference(values))
        if unexpected:
            raise RuntimeError(f"{cohort}: {column} 出现非法值：{unexpected}")
    age = frame["age"].dropna().to_numpy(dtype=float)
    if not np.isfinite(age).all() or ((age < 0) | (age > 120)).any():
        raise RuntimeError(f"{cohort}: age 超出 0–120 的来源支持范围")
    household = frame["household_size"].dropna().astype(int)
    if (household <= 0).any():
        raise RuntimeError(f"{cohort}: household_size 必须为正整数或 NA")
    income_rank = frame["household_income_rank"].dropna().to_numpy(dtype=float)
    if not np.isfinite(income_rank).all() or ((income_rank < 0) | (income_rank > 1)).any():
        raise RuntimeError(f"{cohort}: household_income_rank 必须位于 [0,1]")
    smoking = set(frame["current_smoking"].dropna().astype(bool).unique())
    if not smoking.issubset({False, True}):
        raise RuntimeError(f"{cohort}: current_smoking 出现非法值")
    quintile = set(frame["wealth_quintile"].dropna().astype(int).unique())
    if not quintile.issubset({1, 2, 3, 4, 5}):
        raise RuntimeError(f"{cohort}: wealth_quintile 出现非法值")
    forbidden = ("fi_", "shock_", "recovery_", "analysis_", "model_", "psu", "strata")
    if any(column.startswith(forbidden) for column in frame.columns):
        raise RuntimeError(f"{cohort}: Covariate Panel 含越界字段")


def _frozen_paths(record: dict[str, Any]) -> tuple[Path, str, Path, str]:
    """统一读取不同历史 freeze schema 中的 panel 与 manifest 字段。"""
    if "parquet_path" in record:
        data_path = _repo_path(record["parquet_path"])
        data_sha = record["parquet_sha256"]
        manifest_path = _repo_path(record["manifest_path"])
        manifest_sha = record["manifest_sha256"]
    else:
        data_path = _repo_path(record["path"])
        data_sha = record["sha256"]
        manifest_key = (
            "build_manifest_path"
            if "build_manifest_path" in record
            else "active_manifest_path"
        )
        sha_key = (
            "build_manifest_sha256"
            if "build_manifest_sha256" in record
            else "active_manifest_sha256"
        )
        manifest_path = _repo_path(record[manifest_key])
        manifest_sha = record[sha_key]
    return data_path, data_sha, manifest_path, manifest_sha


def _verify_frozen_component(
    name: str, freeze_path: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    if not freeze_path.is_file():
        raise RuntimeError(f"缺少上游 freeze：{freeze_path}")
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    if freeze.get("freeze_status") != "FROZEN":
        raise RuntimeError(f"{name}: 上游状态不是 FROZEN")
    cohort_records: dict[str, Any] = {}
    for cohort, record in freeze.get("cohorts", {}).items():
        data_path, data_sha, manifest_path, manifest_sha = _frozen_paths(record)
        observed_data_sha = sha256_file(data_path)
        observed_manifest_sha = sha256_file(manifest_path)
        if observed_data_sha != data_sha:
            raise RuntimeError(f"{name} {cohort}: frozen panel 哈希不一致")
        if observed_manifest_sha != manifest_sha:
            raise RuntimeError(f"{name} {cohort}: frozen manifest 哈希不一致")
        cohort_records[cohort] = {
            "data_path": _relative(data_path),
            "data_sha256": observed_data_sha,
            "manifest_path": _relative(manifest_path),
            "manifest_sha256": observed_manifest_sha,
        }
    extras: dict[str, Any] = {}
    for field in ("review_workbook", "data_dictionary"):
        record = freeze.get(field)
        if not isinstance(record, dict) or "path" not in record or "sha256" not in record:
            continue
        path = _repo_path(record["path"])
        observed = sha256_file(path)
        if observed != record["sha256"]:
            raise RuntimeError(f"{name}: {field} 哈希不一致")
        extras[field] = {"path": _relative(path), "sha256": observed}
    return freeze, {
        "freeze_path": _relative(freeze_path),
        "freeze_sha256": sha256_file(freeze_path),
        "cohorts": cohort_records,
        **extras,
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
    "wealth_panel_v1": WEALTH_FREEZE_PATH,
    "exposure_panel_v1": EXPOSURE_FREEZE_PATH,
}


def protected_upstream_snapshot() -> dict[str, Any]:
    """逐项验证七个冻结组件的 panel/manifest 哈希。"""
    snapshot: dict[str, Any] = {}
    for name, path in UPSTREAM_FREEZES.items():
        _freeze, record = _verify_frozen_component(name, path)
        snapshot[name] = record
    return snapshot


def load_frozen_wealth_panel(cohort: str) -> tuple[pd.DataFrame, Path, dict[str, Any]]:
    freeze = json.loads(WEALTH_FREEZE_PATH.read_text(encoding="utf-8"))
    if freeze.get("freeze_status") != "FROZEN":
        raise RuntimeError("Wealth Panel v1 不是 FROZEN")
    item = freeze["cohorts"][cohort]
    path = _repo_path(item["parquet_path"])
    manifest_path = _repo_path(item["manifest_path"])
    if sha256_file(path) != item["parquet_sha256"]:
        raise RuntimeError(f"{cohort}: Wealth Panel v1 parquet 哈希不一致")
    if sha256_file(manifest_path) != item["manifest_sha256"]:
        raise RuntimeError(f"{cohort}: Wealth Panel v1 manifest 哈希不一致")
    frame = pd.read_parquet(path)
    if item["row_count"] != len(frame):
        raise RuntimeError(f"{cohort}: Wealth Panel 行数与 freeze 不一致")
    return frame, path, item


def load_frozen_exposure_panel(cohort: str) -> tuple[pd.DataFrame, Path, dict[str, Any]]:
    freeze = json.loads(EXPOSURE_FREEZE_PATH.read_text(encoding="utf-8"))
    if freeze.get("freeze_status") != "FROZEN":
        raise RuntimeError("Exposure Panel v1 不是 FROZEN")
    item = freeze["cohorts"][cohort]
    path = _repo_path(item["parquet_path"])
    manifest_path = _repo_path(item["manifest_path"])
    if sha256_file(path) != item["parquet_sha256"]:
        raise RuntimeError(f"{cohort}: Exposure Panel v1 parquet 哈希不一致")
    if sha256_file(manifest_path) != item["manifest_sha256"]:
        raise RuntimeError(f"{cohort}: Exposure Panel v1 manifest 哈希不一致")
    frame = pd.read_parquet(path)
    if item["row_count"] != len(frame):
        raise RuntimeError(f"{cohort}: Exposure Panel 行数与 freeze 不一致")
    if int(frame["is_index_shock"].eq(True).sum()) != EXPECTED_INDEX_COUNTS[cohort]:
        raise RuntimeError(f"{cohort}: index shock 数与锁定 release 不一致")
    return frame, path, item


def _rate(numerator: int, denominator: int) -> float | None:
    return None if denominator == 0 else 100.0 * numerator / denominator


def _rate_text(numerator: int, denominator: int) -> str:
    if denominator == 0:
        return f"{numerator:,} (NA)"
    return f"{numerator:,} ({100.0 * numerator / denominator:.1f}%)"


def _range_text(values: pd.Series, *, digits: int = 1) -> str:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if numeric.empty:
        return "NA"
    return f"{numeric.min():,.{digits}f}–{numeric.max():,.{digits}f}"


def _category_text(values: pd.Series) -> str:
    counts = values.dropna().astype(str).value_counts()
    if counts.empty:
        return "NA"
    return "; ".join(f"{key}: {int(value):,}" for key, value in counts.items())


def wave_qc_rows(
    panels: dict[str, pd.DataFrame],
    spines: dict[str, pd.DataFrame],
    diagnostics: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    visible: list[dict[str, Any]] = []
    machine: dict[str, list[dict[str, Any]]] = {cohort: [] for cohort in COHORTS}
    variables = list(COVARIATE_PANEL_COLUMNS[3:])
    for cohort in COHORTS:
        panel = panels[cohort]
        spine = spines[cohort]
        income_by_wave = {
            row["wave"]: row for row in diagnostics[cohort]["income_reference_by_wave"]
        }
        for wave in range(COHORT_SPECS[cohort].wave_min, COHORT_SPECS[cohort].wave_max + 1):
            wave_mask = panel["wave"].eq(wave)
            observed = wave_mask & spine["observed_in_wave"].eq(True).fillna(False)
            observed_n = int(observed.sum())
            nonmissing = {
                variable: int(panel.loc[observed, variable].notna().sum())
                for variable in variables
            }
            income_record = income_by_wave[wave]
            machine_record = {
                "cohort": cohort,
                "wave": wave,
                "full_spine_n": int(wave_mask.sum()),
                "observed_n": observed_n,
                "nonmissing_n": nonmissing,
                "nonmissing_percent": {
                    key: _rate(value, observed_n) for key, value in nonmissing.items()
                },
                "age_range": _range_text(panel.loc[observed, "age"]),
                "household_size_range": _range_text(
                    panel.loc[observed, "household_size"], digits=0
                ),
                "income_rank_range": _range_text(
                    panel.loc[observed, "household_income_rank"], digits=4
                ),
                "category_counts": {
                    variable: {
                        str(key): int(value)
                        for key, value in panel.loc[observed, variable]
                        .dropna()
                        .astype(str)
                        .value_counts()
                        .items()
                    }
                    for variable in (
                        "sex",
                        "education",
                        "marital_status",
                        "labour_force_status",
                        "current_smoking",
                        "wealth_quintile",
                    )
                },
                "income_reference": income_record,
            }
            machine[cohort].append(machine_record)
            price_years = "; ".join(
                f"{row['income_price_year']}: {row['n']:,} ({row['percent']:.1f}%)"
                for row in income_record["income_price_year_distribution"]
            )
            visible.append(
                {
                    "Cohort": cohort,
                    "Wave": wave,
                    "Interviewed N": observed_n,
                    "Age available": _rate_text(nonmissing["age"], observed_n),
                    "Sex available": _rate_text(nonmissing["sex"], observed_n),
                    "Education available": _rate_text(nonmissing["education"], observed_n),
                    "Marital available": _rate_text(nonmissing["marital_status"], observed_n),
                    "Household size available": _rate_text(nonmissing["household_size"], observed_n),
                    "Labour available": _rate_text(nonmissing["labour_force_status"], observed_n),
                    "Income rank available": _rate_text(nonmissing["household_income_rank"], observed_n),
                    "Smoking available": _rate_text(nonmissing["current_smoking"], observed_n),
                    "Wealth quintile available": _rate_text(nonmissing["wealth_quintile"], observed_n),
                    "Age range": machine_record["age_range"],
                    "Household size range": machine_record["household_size_range"],
                    "Income price year(s), N (%)": price_years,
                    "Income reference N": income_record["income_rank_reference_n"],
                    "Income reference weight sum": income_record[
                        "income_rank_reference_weight_sum"
                    ],
                }
            )
    return visible, machine


READINESS_VARIABLES = (
    "age",
    "sex",
    "education",
    "marital_status",
    "household_size",
    "labour_force_status",
    "household_income_rank",
    "current_smoking",
    "wealth_quintile_tminus1",
)


def index_shock_readiness_rows(
    panels: dict[str, pd.DataFrame],
    exposures: dict[str, pd.DataFrame],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    visible: list[dict[str, Any]] = []
    machine: dict[str, dict[str, Any]] = {}
    for cohort in COHORTS:
        index = exposures[cohort].loc[
            exposures[cohort]["is_index_shock"].eq(True),
            [
                "cohort",
                "person_id",
                "candidate_shock_wave",
                "tminus1_wave",
                "recovery_status",
            ],
        ].copy()
        if len(index) != EXPECTED_INDEX_COUNTS[cohort] or index["person_id"].duplicated().any():
            raise RuntimeError(f"{cohort}: index shock readiness 基数不正确")
        t_values = panels[cohort].rename(columns={"wave": "candidate_shock_wave"})
        joined = index.merge(
            t_values,
            on=["cohort", "person_id", "candidate_shock_wave"],
            how="left",
            validate="one_to_one",
            sort=False,
        )
        tminus1 = panels[cohort].loc[:, ["cohort", "person_id", "wave", "wealth_quintile"]].rename(
            columns={"wave": "tminus1_wave", "wealth_quintile": "wealth_quintile_tminus1"}
        )
        joined = joined.merge(
            tminus1,
            on=["cohort", "person_id", "tminus1_wave"],
            how="left",
            validate="one_to_one",
            sort=False,
        )
        if len(joined) != len(index):
            raise RuntimeError(f"{cohort}: readiness join 改变 index 行数")
        groups: list[tuple[str, pd.Series]] = [
            ("All index shocks", pd.Series(True, index=joined.index))
        ]
        for status, label in (
            ("substantial_recovery", "Substantial recovery"),
            ("persistent_or_limited_recovery", "Persistent loss or limited recovery"),
        ):
            groups.append((label, joined["recovery_status"].eq(status).fillna(False)))
        cohort_machine: dict[str, Any] = {"index_shock_n": len(joined), "groups": []}
        for label, mask in groups:
            denominator = int(mask.sum())
            missing = {
                variable: int(joined.loc[mask, variable].isna().sum())
                for variable in READINESS_VARIABLES
            }
            joint_complete = int(
                joined.loc[mask, list(READINESS_VARIABLES)].notna().all(axis=1).sum()
            )
            record = {
                "group": label,
                "n": denominator,
                "missing_n": missing,
                "joint_complete_n": joint_complete,
                "joint_complete_percent": _rate(joint_complete, denominator),
            }
            cohort_machine["groups"].append(record)
            visible.append(
                {
                    "Cohort": cohort,
                    "Recovery group": label,
                    "Index shock N": denominator,
                    "Age missing": _rate_text(missing["age"], denominator),
                    "Sex missing": _rate_text(missing["sex"], denominator),
                    "Education missing": _rate_text(missing["education"], denominator),
                    "Marital missing": _rate_text(missing["marital_status"], denominator),
                    "Household size missing": _rate_text(missing["household_size"], denominator),
                    "Labour missing": _rate_text(missing["labour_force_status"], denominator),
                    "Income rank missing": _rate_text(missing["household_income_rank"], denominator),
                    "Smoking missing": _rate_text(missing["current_smoking"], denominator),
                    "t-1 wealth quintile missing": _rate_text(
                        missing["wealth_quintile_tminus1"], denominator
                    ),
                    "Nine-covariate joint complete": _rate_text(
                        joint_complete, denominator
                    ),
                }
            )
        machine[cohort] = cohort_machine
    return visible, machine


def definition_contract_rows(mapping_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    harmonised_display = {
        "age": "Years (source-specific age convention)",
        "sex": "Male; Female",
        "education": "Low; Medium; High (ordered categories)",
        "marital_status": (
            "Married/partnered; Separated/divorced; Widowed; Never married"
        ),
        "household_size": "Original positive resident count including respondent",
        "labour_force_status": (
            "Working; Unemployed; Retired or partly retired; "
            "Other economically inactive"
        ),
        "household_income_rank": (
            "Weighted within-cohort, within-wave fractional midrank in [0, 1]"
        ),
        "current_smoking": "False; True",
        "wealth_quintile": (
            "Unweighted tie-preserving within-wave sample wealth-rank group 1–5"
        ),
    }
    rows: list[dict[str, Any]] = []
    for target in COVARIATE_PANEL_COLUMNS[3:]:
        selected = [row for row in mapping_rows if row["target_variable"] == target]
        rows.append(
            {
                "Variable": target,
                "Analytic type": ANALYTIC_TYPES[target],
                "Definition / coding": harmonised_display[target],
                "Measurement level": selected[0]["measurement_level"],
                "Intended analytic time": selected[0]["intended_analysis_time"],
                "HRS source": next(
                    f"{row['source_variable']} | {row['harmonised_coding']}"
                    for row in selected
                    if row["cohort"] == "HRS"
                ),
                "ELSA H source": next(
                    f"{row['source_variable']} | {row['harmonised_coding']}"
                    for row in selected
                    if row["cohort"] == "ELSA"
                ),
                "KLoSA source": next(
                    f"{row['source_variable']} | {row['harmonised_coding']}"
                    for row in selected
                    if row["cohort"] == "KLoSA"
                ),
                "Alignment": selected[0]["harmonisation_status"],
                "Cohort-specific note": " | ".join(
                    f"{row['cohort']}: {row['cohort_limitation']}" for row in selected
                ),
                "Model role": selected[0]["model_role"],
            }
        )
    return rows


WORKBOOK_BUILDER = r'''
import fs from "node:fs/promises";
import { FileBlob, SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const [payloadPath, outputPath, previewDir] = process.argv.slice(2);
const payload = JSON.parse(await fs.readFile(payloadPath, "utf8"));
const workbook = Workbook.create();
const fontFamily = "Arial";

function addSheet({name, title, context, headers, rows, tableName, widths, numericFormats, bodyHeight}) {
  const sheet = workbook.worksheets.add(name);
  sheet.showGridLines = false;
  sheet.getRange("A1").values = [[title]];
  sheet.getRange("A1").format.font = {name: fontFamily, size: 15, bold: true, color: "#17365D"};
  const contextRange = sheet.getRangeByIndexes(1, 0, 1, headers.length);
  contextRange.merge();
  contextRange.values = [[context]];
  contextRange.format.font = {name: fontFamily, size: 10, italic: true, color: "#4B5563"};
  contextRange.format.wrapText = true;
  contextRange.format.rowHeightPx = 42;
  sheet.getRangeByIndexes(2, 0, 1, headers.length).format.borders = {
    bottom: {style: "thin", color: "#94A3B8"},
  };
  const matrix = [headers, ...rows.map((row) => headers.map((header) => row[header] ?? null))];
  const range = sheet.getRangeByIndexes(3, 0, matrix.length, headers.length);
  range.values = matrix;
  range.format.font = {name: fontFamily, size: 9, color: "#1F2937"};
  range.format.verticalAlignment = "top";
  range.format.wrapText = true;
  const header = sheet.getRangeByIndexes(3, 0, 1, headers.length);
  header.format = {
    fill: "#1F4E78",
    font: {name: fontFamily, size: 9, bold: true, color: "#FFFFFF"},
    horizontalAlignment: "center",
    verticalAlignment: "center",
    wrapText: true,
  };
  header.format.rowHeightPx = 46;
  for (let index = 0; index < widths.length; index += 1) {
    sheet.getRangeByIndexes(0, index, matrix.length + 3, 1).format.columnWidthPx = widths[index];
  }
  if (rows.length > 0) {
    sheet.getRangeByIndexes(4, 0, rows.length, headers.length).format.rowHeightPx = bodyHeight ?? 34;
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
}

addSheet({
  name: "1_Definition_Contract",
  title: "Covariate definition contract",
  context: "Person-wave measurement layer only. Sex and education are source-supported fixed person attributes; all other substantive covariates are wave-specific. Income rank and wealth quintile use different locked reference distributions.",
  headers: payload.definition.headers,
  rows: payload.definition.rows,
  tableName: "CovariateDefinitionContract",
  widths: [145, 125, 230, 130, 120, 190, 200, 220, 120, 360, 180],
  numericFormats: {},
  bodyHeight: 104,
});

addSheet({
  name: "2_Wave_QC",
  title: "Covariate measurement by wave",
  context: "Availability percentages use interviewed person-waves. Income price-year distributions use records with valid income; reference N and weight sum describe the actual positive-weight rank reference population.",
  headers: payload.wave.headers,
  rows: payload.wave.rows,
  tableName: "CovariateWaveQC",
  widths: [70, 52, 90, 115, 110, 130, 110, 135, 115, 125, 105, 135, 88, 100, 250, 105, 145],
  numericFormats: {"Wave": "0", "Interviewed N": "#,##0", "Income reference N": "#,##0", "Income reference weight sum": "#,##0.0000"},
  bodyHeight: 42,
});

addSheet({
  name: "3_Index_Shock_Readiness",
  title: "Index-shock covariate readiness",
  context: "Diagnostic only: t values for age through smoking and t-1 wealth quintile. Joint completeness covers these nine covariates only; it excludes FI_t, future model weights, follow-up, and final analytic eligibility.",
  headers: payload.readiness.headers,
  rows: payload.readiness.rows,
  tableName: "CovariateIndexReadiness",
  widths: [70, 220, 95, 105, 100, 120, 105, 125, 105, 115, 110, 145, 180],
  numericFormats: {"Index shock N": "#,##0"},
  bodyHeight: 42,
});

const xlsx = await SpreadsheetFile.exportXlsx(workbook);
await xlsx.save(outputPath);
const saved = await SpreadsheetFile.importXlsx(await FileBlob.load(outputPath));
const overview = await saved.inspect({
  kind: "workbook,sheet,table",
  maxChars: 9000,
  tableMaxRows: 5,
  tableMaxCols: 10,
});
const formulaErrors = await saved.inspect({
  kind: "match",
  searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!|#SPILL!|#CALC!",
  options: {useRegex: true, maxResults: 100},
  summary: "post-save formula error scan",
});
await fs.mkdir(previewDir, {recursive: true});
for (const sheetName of ["1_Definition_Contract", "2_Wave_QC", "3_Index_Shock_Readiness"]) {
  const preview = await saved.render({sheetName, autoCrop: "all", scale: 1.25, format: "png"});
  await fs.writeFile(`${previewDir}/${sheetName}.png`, new Uint8Array(await preview.arrayBuffer()));
}
console.log(JSON.stringify({overview: overview.ndjson, formulaErrors: formulaErrors.ndjson}));
'''


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
    definition_rows: list[dict[str, Any]],
    wave_rows: list[dict[str, Any]],
    readiness_rows: list[dict[str, Any]],
    output_path: Path,
    preview_dir: Path,
) -> dict[str, Any]:
    """用 artifact-tool 创建、保存后复载检查并渲染三表 workbook。"""
    if not definition_rows or not wave_rows or not readiness_rows:
        raise RuntimeError("Covariate review workbook 的三张表均不得为空")
    payload = {
        "definition": {"headers": list(definition_rows[0]), "rows": definition_rows},
        "wave": {"headers": list(wave_rows[0]), "rows": wave_rows},
        "readiness": {"headers": list(readiness_rows[0]), "rows": readiness_rows},
    }
    node, node_modules = _runtime_paths()
    with tempfile.TemporaryDirectory(prefix="covariate_panel_xlsx_") as tmp_name:
        tmp = Path(tmp_name)
        payload_path = tmp / "payload.json"
        builder_path = tmp / "build_review.mjs"
        payload_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
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
                "artifact-tool Covariate review workbook 构建失败：\n"
                + (completed.stdout or "")
                + "\n"
                + (completed.stderr or "")
            )
        if not output_path.is_file():
            raise RuntimeError("artifact-tool 未生成 Covariate review workbook")
        stdout = completed.stdout.strip()
        if any(
            token in stdout
            for token in (
                "#REF!",
                "#DIV/0!",
                "#VALUE!",
                "#NAME?",
                "#NUM!",
                "#NULL!",
                "#SPILL!",
                "#CALC!",
            )
        ):
            raise RuntimeError("Covariate review workbook 公式错误扫描未通过")
        previews = sorted(str(path) for path in preview_dir.glob("*.png"))
        if len(previews) != 3:
            raise RuntimeError("Covariate review workbook 未完成三 Sheet 渲染")
        return {
            "artifact_tool_stdout": stdout,
            "post_save_reload": "PASS",
            "formula_error_scan": "PASS",
            "preview_files": previews,
        }


DICTIONARY_TEXT = """# Covariate Panel v1 数据字典

## 粒度、主键与状态

每行表示一个 `cohort + person_id + wave`。三队列行集合、主键和顺序与冻结 Full Spine v2 完全一致。该版本是供人工复核的 release candidate；在另行签署正式 freeze 前，不得作为 weights、analysis panel 或模型的正式输入。

## 12 列物理接口

| Column | dtype | nullable | analytic_type | Definition |
|---|---|---:|---|---|
| `cohort` | string | no | identifier | `HRS`、`ELSA` 或 `KLoSA`。 |
| `person_id` | string | no | identifier | 队列内无损字符串化 person ID。 |
| `wave` | int16 | no | time_index | 队列全局波次。 |
| `age` | float64 | yes | continuous | HRS 为访谈起止日期中点的整数周岁（mid-interview age）；ELSA 为完整周岁且 90 表示 90 岁及以上；KLoSA 为调查年减出生年，不自行减 1。 |
| `sex` | string | yes | categorical | `Male` / `Female`；来源支持的固定个人属性。 |
| `education` | string | yes | categorical_ordered | `Low` / `Medium` / `High`；来源支持的固定个人属性。HRS High 是 some college or above；ELSA 只用 Version H `raeducl`，绝不以 `raedyrs_e` fallback；KLoSA High 为完成两年制大学、大学或研究生教育。 |
| `marital_status` | string | yes | categorical | `Married/partnered`、`Separated/divorced`、`Widowed`、`Never married`。 |
| `household_size` | Int16 | yes | continuous_count | 家庭中居住人数，包含受访者；不分组、不截断。0 或负值阻止候选发布。 |
| `labour_force_status` | string | yes | categorical | `Working`、`Unemployed`、`Retired or partly retired`、`Other economically inactive`。ELSA `.o` 与 KLoSA `.s` 等特殊缺失保持 NA。 |
| `household_income_rank` | float64 | yes | continuous_rank | 队列内、波次内，以正的同波个人横截面调查权重构建的 weighted fractional midrank，范围 `[0,1]`。收入有效但自身权重缺失/非正者仍可查询已建立的参考分布。 |
| `current_smoking` | boolean | yes | binary | 协调来源中的当前吸烟 0/1；官方合法跳题推导值保留。 |
| `wealth_quintile` | Int8 | yes | categorical_ordered | 冻结 Wealth Panel v1 在队列内、波次内观察样本中的未加权财富位次组；`m=(average_rank-0.5)/N`，`Q=ceil(5m)`；同值不拆，零与负财富保留。后续模型按分类变量处理。 |

## 访谈行与固定属性

`sex` 和 `education` 是来源支持的 person-level 属性，可在 `observed_in_wave == FALSE` 的 Spine 行保持非缺失。其余实质协变量在非访谈行必须为 `NA`。本项目不按个人的 `t+1` 或以后测量回填缺失的 `t` 值，也不利用 recovery、FI、健康结局、随访或最终样本状态定义协变量。

## 收入 rank 参考总体

每个 cohort × wave 的参考总体为：该波实际访谈、收入有效、同波横截面个人权重为正的所有年龄受访者记录。按个人记录排名，不按家庭去重。HRS 使用 `r{w}wtcrnh`，ELSA/KLoSA 使用 `r{w}wtresp`；这只定义收入 rank，不锁定未来模型权重。HRS 收入价格年为 `interview_year-1`（缺失时 `nominal_year-1`）；ELSA 为访谈年（缺失时 nominal year）；KLoSA W1–W9 固定为 2005、2007、…、2021。各波实际参考人数、权重和 price-year 分布见 cohort manifest。

## 插补与缺失

保留 source/harmonised 数据已经提供的官方插补值；本项目不额外插补、forward-fill 或跨波回填。分类变量的 tagged/special missing 保持 `NA`，但连续收入中的真实负值不按缺失代码删除。

## 分析时点

最终分析层以后从 index-shock 波次 `t` 读取 age、sex、education、marital status、household size、labour-force status、income rank 和 current smoking，并从 `t-1` 读取 wealth quintile。该面板本身不筛选 index shock 或最终分析样本。
"""


def _schema(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return [
        {
            "name": column,
            "dtype": str(frame[column].dtype),
            "nullable": bool(frame[column].isna().any()),
            "analytic_type": ANALYTIC_TYPES[column],
        }
        for column in frame.columns
    ]


def _git_output(arguments: list[str], *, binary: bool = False) -> bytes | str:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=ROOT,
        check=True,
        capture_output=True,
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
    """冻结一次构建所用的代码、mapping、design 与 Git provenance。"""
    mapping_rows = verify_variable_map_contract()
    verify_locked_design()
    return {
        "runner_version": RUNNER_VERSION,
        "builder_version": BUILDER_VERSION,
        "builder_sha256": sha256_file(Path(__file__)),
        "spec_version": SPEC_VERSION,
        "specification_sha256": specification_hash(),
        "map_version": MAP_VERSION,
        "mapping_sha256": sha256_file(VARIABLE_MAP_PATH),
        "mapping_records_sha256": _stable_hash(mapping_rows),
        "design_register_sha256": sha256_file(DESIGN_REGISTER_PATH),
        "covariate_design_subset_sha256": covariate_design_subset_hash(),
        "git_provenance": git_provenance(),
    }


def _final_output_path(cohort: str) -> Path:
    return (
        DERIVED_ROOT
        / cohort
        / "08_covariate_panel"
        / f"{cohort.lower()}_covariate_panel_v1.parquet"
    )


def _manifest_path(cohort: str) -> Path:
    return COVARIATE_MANIFEST_ROOT / f"{cohort.lower()}_covariate_panel_v1.json"


def _candidate_release_paths() -> list[Path]:
    return [
        *(_final_output_path(cohort) for cohort in COHORTS),
        *(_manifest_path(cohort) for cohort in COHORTS),
        REVIEW_PATH,
        DICTIONARY_PATH,
        RELEASE_CANDIDATE_PATH,
        FORMAL_FREEZE_PATH,
    ]


def _verify_release_absent() -> None:
    existing = [str(path) for path in _candidate_release_paths() if path.exists()]
    if existing:
        raise RuntimeError(
            "拒绝覆盖已有 Covariate Panel v1 candidate 资产：" + "; ".join(existing)
        )


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
    load_spine: Callable[[str], tuple[pd.DataFrame, Path, dict[str, Any]]],
) -> dict[str, Any]:
    panels: dict[str, pd.DataFrame] = {}
    spines: dict[str, pd.DataFrame] = {}
    exposures: dict[str, pd.DataFrame] = {}
    diagnostics: dict[str, dict[str, Any]] = {}
    inputs: dict[str, dict[str, Any]] = {}
    for cohort in COHORTS:
        spine, spine_path, spine_manifest = load_spine(cohort)
        if len(spine) != EXPECTED_SPINE_ROWS[cohort]:
            raise RuntimeError(f"{cohort}: Full Spine 行数与冻结预期不一致")
        wealth, wealth_path, wealth_record = load_frozen_wealth_panel(cohort)
        exposure, exposure_path, exposure_record = load_frozen_exposure_panel(cohort)
        raw = read_raw_source(source_contexts[cohort]["raw_path"], cohort, spine)
        price_index, price_context = extract_income_price_index(raw, spine, cohort)
        panel, cohort_diagnostics = build_covariate_panel(
            raw, spine, wealth, cohort, price_index=price_index
        )
        cohort_diagnostics["price_index"] = price_context
        validate_covariate_panel(panel, spine, cohort)
        panels[cohort] = panel
        spines[cohort] = spine
        exposures[cohort] = exposure
        diagnostics[cohort] = cohort_diagnostics
        inputs[cohort] = {
            "spine": {
                "path": _relative(spine_path),
                "sha256": spine_manifest["output"]["sha256"],
            },
            "wealth_panel_v1": {
                "path": _relative(wealth_path),
                "sha256": wealth_record["parquet_sha256"],
            },
            "exposure_panel_v1_qc_only": {
                "path": _relative(exposure_path),
                "sha256": exposure_record["parquet_sha256"],
                "is_measurement_build_input": False,
            },
        }
        del raw
    wave_visible, wave_machine = wave_qc_rows(panels, spines, diagnostics)
    readiness_visible, readiness_machine = index_shock_readiness_rows(
        panels, exposures
    )
    return {
        "panels": panels,
        "spines": spines,
        "diagnostics": diagnostics,
        "inputs": inputs,
        "definition_visible": definition_contract_rows(verify_variable_map_contract()),
        "wave_visible": wave_visible,
        "wave_machine": wave_machine,
        "readiness_visible": readiness_visible,
        "readiness_machine": readiness_machine,
    }


def _build_manifest(
    cohort: str,
    panel: pd.DataFrame,
    *,
    staged_output: Path,
    source_context: dict[str, Any],
    build_inputs: dict[str, Any],
    diagnostics: dict[str, Any],
    wave_qc: list[dict[str, Any]],
    readiness: dict[str, Any],
    upstream_snapshot: dict[str, Any],
    contract_snapshot: dict[str, Any],
) -> dict[str, Any]:
    overall = next(row for row in readiness["groups"] if row["group"] == "All index shocks")
    provenance = contract_snapshot["git_provenance"]
    return {
        "manifest_schema_version": "covariate_panel_manifest_v1",
        "release_version": RELEASE_VERSION,
        "release_status": "READY_TO_FREEZE",
        "cohort": cohort,
        "generated_utc": datetime.now(UTC).isoformat(),
        "git_commit": provenance["git_commit"],
        "git_dirty": provenance["git_dirty"],
        "git_diff_sha256": provenance["git_diff_sha256"],
        "runner_version": contract_snapshot["runner_version"],
        "builder_version": contract_snapshot["builder_version"],
        "builder_sha256": contract_snapshot["builder_sha256"],
        "spec_version": contract_snapshot["spec_version"],
        "specification_sha256": contract_snapshot["specification_sha256"],
        "map_version": contract_snapshot["map_version"],
        "mapping_sha256": contract_snapshot["mapping_sha256"],
        "mapping_records_sha256": contract_snapshot["mapping_records_sha256"],
        "design_register_sha256": contract_snapshot["design_register_sha256"],
        "covariate_design_subset_sha256": contract_snapshot[
            "covariate_design_subset_sha256"
        ],
        "key": list(KEY_COLUMNS),
        "row_count": len(panel),
        "key_unique": not panel.duplicated(list(KEY_COLUMNS)).any(),
        "key_set_and_order_equal_full_spine": True,
        "schema": _schema(panel),
        "schema_contract": [
            {
                "name": column,
                "dtype": COVARIATE_PANEL_DTYPES[column],
                "nullable": COVARIATE_PANEL_NULLABLE[column],
                "analytic_type": ANALYTIC_TYPES[column],
            }
            for column in COVARIATE_PANEL_COLUMNS
        ],
        "build_inputs": {
            "raw": _serialise_source_context(source_context),
            **build_inputs,
            "design_register": {
                "path": _relative(DESIGN_REGISTER_PATH),
                "sha256": contract_snapshot["design_register_sha256"],
                "covariate_design_subset_sha256": contract_snapshot[
                    "covariate_design_subset_sha256"
                ],
            },
            "variable_map": {
                "path": _relative(VARIABLE_MAP_PATH),
                "sha256": contract_snapshot["mapping_sha256"],
                "records_sha256": contract_snapshot["mapping_records_sha256"],
            },
        },
        "fixed_attribute_policy": {
            "fixed": ["sex", "education"],
            "may_be_populated_on_noninterview_rows": True,
            "other_substantive_covariates_missing_on_noninterview_rows": True,
        },
        "imputation_policy": {
            "official_harmonised_values_retained": True,
            "project_imputation_or_cross_wave_fill": False,
        },
        "source_qc": diagnostics["source_qc"],
        "income_price_index": diagnostics["price_index"],
        "income_rank_reference_by_wave": diagnostics["income_reference_by_wave"],
        "wealth_quintile_reference_by_wave": diagnostics["wealth_reference_by_wave"],
        "wave_qc": wave_qc,
        "index_shock_readiness": {
            "index_shocks": readiness["index_shock_n"],
            "missing_n": overall["missing_n"],
            "nine_covariate_joint_complete_n": overall["joint_complete_n"],
            "nine_covariate_joint_complete_percent": overall[
                "joint_complete_percent"
            ],
            "groups": readiness["groups"],
            "diagnostic_only": True,
            "excludes_fi_model_weight_followup_and_final_eligibility": True,
        },
        "upstream_frozen_snapshot": upstream_snapshot,
        "output": {
            "path": _relative(_final_output_path(cohort)),
            "sha256": sha256_file(staged_output),
        },
        "forbidden_actions_confirmed_absent": [
            "final analytic eligibility filtering",
            "project imputation",
            "FI or health outcome construction",
            "recovery or shock redefinition",
            "future-wave personal backfill",
            "model weighting",
            "model fitting",
        ],
    }


def _build_release_candidate(
    *,
    staged_outputs: dict[str, Path],
    staged_manifests: dict[str, Path],
    staged_review: Path,
    staged_dictionary: Path,
    prepared: dict[str, Any],
    upstream_snapshot: dict[str, Any],
    contract_snapshot: dict[str, Any],
) -> dict[str, Any]:
    provenance = contract_snapshot["git_provenance"]
    return {
        "release_candidate_schema_version": "covariate_panel_release_candidate_v1",
        "release_status": "READY_TO_FREEZE",
        "release_version": RELEASE_VERSION,
        "generated_utc": datetime.now(UTC).isoformat(),
        "git_commit": provenance["git_commit"],
        "git_dirty": provenance["git_dirty"],
        "git_diff_sha256": provenance["git_diff_sha256"],
        "runner_version": contract_snapshot["runner_version"],
        "builder_version": contract_snapshot["builder_version"],
        "builder_sha256": contract_snapshot["builder_sha256"],
        "spec_version": contract_snapshot["spec_version"],
        "specification_sha256": contract_snapshot["specification_sha256"],
        "mapping": {
            "path": _relative(VARIABLE_MAP_PATH),
            "sha256": contract_snapshot["mapping_sha256"],
            "records_sha256": contract_snapshot["mapping_records_sha256"],
            "map_version": contract_snapshot["map_version"],
        },
        "design_register": {
            "path": _relative(DESIGN_REGISTER_PATH),
            "full_file_sha256": contract_snapshot["design_register_sha256"],
            "covariate_design_subset_sha256": contract_snapshot[
                "covariate_design_subset_sha256"
            ],
            "locked_decisions": LOCKED_COVARIATE_DESIGN,
        },
        "common_key": list(KEY_COLUMNS),
        "physical_columns": list(COVARIATE_PANEL_COLUMNS),
        "analytic_types": ANALYTIC_TYPES,
        "common_schema_contract": [
            {
                "name": column,
                "dtype": COVARIATE_PANEL_DTYPES[column],
                "nullable": COVARIATE_PANEL_NULLABLE[column],
                "analytic_type": ANALYTIC_TYPES[column],
            }
            for column in COVARIATE_PANEL_COLUMNS
        ],
        "cohorts": {
            cohort: {
                "parquet_path": _relative(_final_output_path(cohort)),
                "parquet_sha256": sha256_file(staged_outputs[cohort]),
                "manifest_path": _relative(_manifest_path(cohort)),
                "manifest_sha256": sha256_file(staged_manifests[cohort]),
                "row_count": len(prepared["panels"][cohort]),
                "index_shocks": prepared["readiness_machine"][cohort][
                    "index_shock_n"
                ],
                "nine_covariate_joint_complete_n": next(
                    row["joint_complete_n"]
                    for row in prepared["readiness_machine"][cohort]["groups"]
                    if row["group"] == "All index shocks"
                ),
            }
            for cohort in COHORTS
        },
        "review_workbook": {
            "path": _relative(REVIEW_PATH),
            "sha256": sha256_file(staged_review),
            "sheets": [
                "1_Definition_Contract",
                "2_Wave_QC",
                "3_Index_Shock_Readiness",
            ],
            "post_save_reload": "PASS",
            "formula_error_scan": "PASS",
            "three_sheet_render": "PASS",
        },
        "data_dictionary": {
            "path": _relative(DICTIONARY_PATH),
            "sha256": sha256_file(staged_dictionary),
        },
        "protected_inputs_before_and_after_identical": True,
        "release_contract_before_and_after_identical": True,
        "upstream_frozen_snapshot": upstream_snapshot,
        "repository_test_baseline": {
            "command": "uv run --locked pytest -q",
            "passed": 47,
            "failed": 1,
            "known_failure": (
                "test_supplementary_table_s3_reflects_v2_boundary"
            ),
            "covariate_release_requirement": "same known failure only; no new failures",
        },
        "formal_freeze_created": False,
        "downstream_use_authorized": False,
        "formal_freeze_rule": (
            "After human review, a formal freeze must reference these exact candidate "
            "hashes and must not rebuild any Parquet, manifest, workbook, or dictionary."
        ),
        "next_stage_status": "HUMAN_REVIEW_REQUIRED_BEFORE_FREEZE",
    }


def _publish_candidate_transaction(
    staged_by_final: list[tuple[Path, Path]],
) -> None:
    """按 candidate JSON 最后顺序发布；任一失败则删除本次已发布文件。"""
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


def run_covariate_panel_release(
    cohorts: list[str],
    *,
    load_spine: Callable[[str], tuple[pd.DataFrame, Path, dict[str, Any]]],
) -> dict[str, Any]:
    """构建三队列 Covariate Panel v1 并发布 READY_TO_FREEZE 候选。"""
    if tuple(cohorts) != COHORTS:
        raise RuntimeError(
            "covariate v1 必须使用 --cohort all 一次性构建三队列共同候选"
        )
    _verify_release_absent()
    contract_before = _release_contract_snapshot()
    upstream_before = protected_upstream_snapshot()
    source_before = {
        cohort: verify_source_context(cohort) for cohort in COHORTS
    }
    prepared = prepare_measurements(
        source_contexts=source_before,
        load_spine=load_spine,
    )
    preview_dir = Path(tempfile.gettempdir()) / "covariate_panel_review_v1_previews"
    if preview_dir.exists():
        shutil.rmtree(preview_dir)

    with tempfile.TemporaryDirectory(prefix="covariate_panel_v1_") as tmp_name:
        staging = Path(tmp_name)
        staged_outputs: dict[str, Path] = {}
        for cohort in COHORTS:
            staged = staging / _final_output_path(cohort).name
            prepared["panels"][cohort].to_parquet(
                staged, index=False, compression="zstd"
            )
            reloaded = pd.read_parquet(staged)
            validate_covariate_panel(
                reloaded, prepared["spines"][cohort], cohort
            )
            staged_outputs[cohort] = staged

        staged_review = staging / REVIEW_PATH.name
        workbook_checks = create_review_workbook(
            prepared["definition_visible"],
            prepared["wave_visible"],
            prepared["readiness_visible"],
            staged_review,
            preview_dir,
        )
        if workbook_checks["post_save_reload"] != "PASS":
            raise RuntimeError("Covariate review workbook 保存后复载未通过")

        staged_dictionary = staging / DICTIONARY_PATH.name
        staged_dictionary.write_text(DICTIONARY_TEXT, encoding="utf-8")
        staged_manifests: dict[str, Path] = {}
        for cohort in COHORTS:
            manifest = _build_manifest(
                cohort,
                prepared["panels"][cohort],
                staged_output=staged_outputs[cohort],
                source_context=source_before[cohort],
                build_inputs=prepared["inputs"][cohort],
                diagnostics=prepared["diagnostics"][cohort],
                wave_qc=prepared["wave_machine"][cohort],
                readiness=prepared["readiness_machine"][cohort],
                upstream_snapshot=upstream_before,
                contract_snapshot=contract_before,
            )
            staged_manifest = staging / _manifest_path(cohort).name
            _json_write(staged_manifest, manifest)
            staged_manifests[cohort] = staged_manifest

        source_after = {
            cohort: verify_source_context(cohort) for cohort in COHORTS
        }
        if {
            cohort: _serialise_source_context(source_before[cohort])
            for cohort in COHORTS
        } != {
            cohort: _serialise_source_context(source_after[cohort])
            for cohort in COHORTS
        }:
            raise RuntimeError("Covariate 构建期间 raw/codebook 发生变化")
        upstream_after = protected_upstream_snapshot()
        if upstream_before != upstream_after:
            raise RuntimeError("Covariate 构建期间上游 frozen 资产发生变化")
        contract_after = _release_contract_snapshot()
        if contract_before != contract_after:
            raise RuntimeError(
                "Covariate 构建期间 builder/mapping/design/Git provenance 发生变化"
            )

        staged_candidate = staging / RELEASE_CANDIDATE_PATH.name
        _json_write(
            staged_candidate,
            _build_release_candidate(
                staged_outputs=staged_outputs,
                staged_manifests=staged_manifests,
                staged_review=staged_review,
                staged_dictionary=staged_dictionary,
                prepared=prepared,
                upstream_snapshot=upstream_before,
                contract_snapshot=contract_before,
            ),
        )
        if contract_before != _release_contract_snapshot():
            raise RuntimeError(
                "Covariate 发布前 builder/mapping/design/Git provenance 发生变化"
            )
        staged_by_final = [
            *((staged_outputs[cohort], _final_output_path(cohort)) for cohort in COHORTS),
            (staged_review, REVIEW_PATH),
            (staged_dictionary, DICTIONARY_PATH),
            *((staged_manifests[cohort], _manifest_path(cohort)) for cohort in COHORTS),
            (staged_candidate, RELEASE_CANDIDATE_PATH),
        ]
        _publish_candidate_transaction(staged_by_final)

    return {
        "release_status": "READY_TO_FREEZE",
        "release_candidate_path": str(RELEASE_CANDIDATE_PATH),
        "preview_dir": str(preview_dir),
        "cohorts": {
            cohort: {
                "rows": len(prepared["panels"][cohort]),
                "index_shocks": prepared["readiness_machine"][cohort][
                    "index_shock_n"
                ],
                "nine_covariate_joint_complete_n": next(
                    row["joint_complete_n"]
                    for row in prepared["readiness_machine"][cohort]["groups"]
                    if row["group"] == "All index shocks"
                ),
                "output": str(_final_output_path(cohort)),
            }
            for cohort in COHORTS
        },
        "review_workbook": str(REVIEW_PATH),
        "formal_freeze_created": False,
    }


__all__ = [
    "ANALYTIC_TYPES",
    "BUILDER_VERSION",
    "COHORT_SPECS",
    "COVARIATE_PANEL_COLUMNS",
    "COVARIATE_PANEL_DTYPES",
    "COVARIATE_PANEL_NULLABLE",
    "FORMAL_FREEZE_PATH",
    "RELEASE_CANDIDATE_PATH",
    "REVIEW_PATH",
    "ROOT",
    "build_covariate_panel",
    "create_review_workbook",
    "income_price_years",
    "run_covariate_panel_release",
    "specification",
    "specification_hash",
    "validate_covariate_panel",
    "verify_locked_design",
    "verify_variable_map_contract",
    "wealth_quintile_from_midrank",
    "weighted_midrank",
]
