"""三队列 Wealth Panel v1：逐波财富测量、QC、manifest 与冻结。"""

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
from typing import Any, Callable

import numpy as np
import pandas as pd
from pandas.io.stata import StataMissingValue

from outcome_components import KEY_COLUMNS, verify_prepanels
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
VARIABLE_MAP_PATH = ADMIN / "variable_maps" / "wealth_variable_map.csv"
MANIFEST_ROOT = ADMIN / "panel_manifests"
WEALTH_MANIFEST_ROOT = MANIFEST_ROOT / "wealth_panel"
DERIVED_ROOT = ROOT / "01_data" / "derived" / "panel_components"
FI_FREEZE_PATH = MANIFEST_ROOT / "fi_panel" / "fi_panel_v2_freeze.json"
REVIEW_PATH = ROOT / "04_collaboration" / "wealth_panel_review_v1.xlsx"
REVIEW_REVISION_PATH = ROOT / "04_collaboration" / "wealth_panel_review_v1_1.xlsx"
REVIEW_REVISION_MANIFEST_PATH = (
    WEALTH_MANIFEST_ROOT / "wealth_panel_review_v1_1_manifest.json"
)
DICTIONARY_PATH = WEALTH_MANIFEST_ROOT / "wealth_panel_data_dictionary.md"
FREEZE_PATH = WEALTH_MANIFEST_ROOT / "wealth_panel_v1_freeze.json"

COHORTS = ("HRS", "ELSA", "KLoSA")
RELEASE_VERSION = "v1"
BUILDER_VERSION = "wealth_panel_builder_v1.0.1"
MAP_VERSION = "wealth_variable_map_v1"
SPEC_VERSION = "wealth_measurement_spec_v1"
REVIEW_REVISION_VERSION = "v1.1"

WEALTH_PANEL_COLUMNS = (
    "cohort",
    "person_id",
    "wave",
    "wealth_total_real_equiv",
    "wealth_equiv_divisor",
    "wealth_source_flag",
)

WEALTH_PANEL_DTYPES = {
    "cohort": "string",
    "person_id": "string",
    "wave": "int16",
    "wealth_total_real_equiv": "float64",
    "wealth_equiv_divisor": "Float64",
    "wealth_source_flag": "string",
}

HRS_CPI_SOURCE_URL = (
    "https://www.bls.gov/cpi/tables/supplemental-files/"
    "historical-cpi-u-202312.pdf"
)
HRS_CODEBOOK_URL = "https://hrs.isr.umich.edu/documentation/codebooks"
ELSA_SOURCE_URL = "https://www.g2aging.org/harmonized-elsa"
KLOSA_SOURCE_URL = "https://www.g2aging.org/harmonized-klosa/creation-code"

# BLS CPI-U：US city average、all items、annual average、1982-84=100。
# 2023 仅服务于 HRS W16 中实际于 2023 年完成的访谈；基期仍为 2022。
HRS_CPI_U_ANNUAL = {
    1992: 140.3,
    1993: 144.5,
    1994: 148.2,
    1995: 152.4,
    1996: 156.9,
    1997: 160.5,
    1998: 163.0,
    1999: 166.6,
    2000: 172.2,
    2001: 177.1,
    2002: 179.9,
    2003: 184.0,
    2004: 188.9,
    2005: 195.3,
    2006: 201.6,
    2007: 207.342,
    2008: 215.303,
    2009: 214.537,
    2010: 218.056,
    2011: 224.939,
    2012: 229.594,
    2013: 232.957,
    2014: 236.736,
    2015: 237.017,
    2016: 240.007,
    2017: 245.120,
    2018: 251.107,
    2019: 255.657,
    2020: 258.811,
    2021: 270.970,
    2022: 292.655,
    2023: 304.702,
}

FLAG_ALLOWED_CODES = {
    "HRS": set(),
    "ELSA": {1, 2, 3, 5, 6, 7, 9, 10},
    "KLoSA": {-2, -1, 1, 2, 3, 5, 6, 7},
}

FLAG_LABELS = {
    "ELSA": {
        1: "continuous value",
        2: "closed range bracket",
        3: "open range bracket",
        5: "no value/bracket",
        6: "no asset",
        7: "don't know ownership",
        9: "non-responding spouse",
        10: "institutional interview",
    },
    "KLoSA": {
        -2: "no imputation: value missing",
        -1: "no imputation: don't know ownership",
        1: "continuous value",
        2: "complete bracket",
        3: "incomplete bracket",
        5: "no value/bracket",
        6: "no asset",
        7: "don't know ownership",
    },
}


@dataclass(frozen=True)
class CohortWealthSpec:
    """单队列财富来源与计算契约。"""

    cohort: str
    wave_min: int
    wave_max: int
    person_id_source: str
    wealth_template: str
    flag_template: str | None
    flag_wave_max: int | None
    couple_template: str | None
    base_year: int
    reporting_level: str
    nominal_unit: str
    final_unit: str
    design_value: str


COHORT_SPECS = {
    "HRS": CohortWealthSpec(
        cohort="HRS",
        wave_min=1,
        wave_max=16,
        person_id_source="hhidpn",
        wealth_template="h{wave}atotw",
        flag_template=None,
        flag_wave_max=None,
        couple_template="h{wave}cpl",
        base_year=2022,
        reporting_level="Household/couple",
        nominal_unit="Current USD",
        final_unit="2022 USD, equivalised",
        design_value="h_atotw_real2022_equiv",
    ),
    "ELSA": CohortWealthSpec(
        cohort="ELSA",
        wave_min=1,
        wave_max=10,
        person_id_source="idauniq",
        wealth_template="h{wave}atotb",
        flag_template="h{wave}aftotb",
        flag_wave_max=10,
        couple_template="h{wave}cpl",
        base_year=2020,
        reporting_level="Family/couple",
        nominal_unit="Current GBP",
        final_unit="2020 GBP, equivalised",
        design_value="h_atotb_real2020_equiv",
    ),
    "KLoSA": CohortWealthSpec(
        cohort="KLoSA",
        wave_min=1,
        wave_max=9,
        person_id_source="pid",
        wealth_template="r{wave}atotb",
        flag_template="r{wave}aftotb",
        flag_wave_max=7,
        couple_template=None,
        base_year=2022,
        reporting_level="Respondent",
        nominal_unit="10,000 current KRW",
        final_unit="2022 KRW, 10,000-unit scale",
        design_value="r_atotb_real2022_individual",
    ),
}

MAPPING_COLUMNS = (
    "cohort",
    "total_wealth_source",
    "source_flag_source",
    "wave_range",
    "source_definition",
    "reporting_level",
    "nominal_unit",
    "real_base_year",
    "price_index_source",
    "price_index_formula",
    "couple_status_source",
    "couple_status_rule",
    "unknown_status_action",
    "map_version",
)

MAPPING_RECORDS = (
    {
        "cohort": "HRS",
        "total_wealth_source": "h{wave}atotw",
        "source_flag_source": "",
        "wave_range": "W1-W16",
        "source_definition": "Total wealth excluding IRA/Keogh assets; assets excluding IRA/Keogh minus debts",
        "reporting_level": "household/couple",
        "nominal_unit": "current USD",
        "real_base_year": "2022",
        "price_index_source": "BLS CPI-U; US city average; all items; annual average; 1982-84=100; 1992-2023 table",
        "price_index_formula": "wealth * CPI_U(2022) / CPI_U(price_year) / divisor",
        "couple_status_source": "h{wave}cpl",
        "couple_status_rule": "0 -> 1.0; 1 -> 1.5",
        "unknown_status_action": "stop if wealth is observed and couple status is not 0 or 1",
        "map_version": MAP_VERSION,
    },
    {
        "cohort": "ELSA",
        "total_wealth_source": "h{wave}atotb",
        "source_flag_source": "h{wave}aftotb",
        "wave_range": "W1-W10",
        "source_definition": "Total family wealth including second home; source total excludes vehicles",
        "reporting_level": "family/couple",
        "nominal_unit": "current GBP",
        "real_base_year": "2020",
        "price_index_source": "Gateway Harmonized ELSA cYYYYcpindex; World Bank WDI; 2010=100",
        "price_index_formula": "wealth * c2020cpindex / cPRICEYEARcpindex / divisor",
        "couple_status_source": "h{wave}cpl",
        "couple_status_rule": "0 -> 1.0; 1 -> 1.5",
        "unknown_status_action": "stop if wealth is observed and couple status is not 0 or 1",
        "map_version": MAP_VERSION,
    },
    {
        "cohort": "KLoSA",
        "total_wealth_source": "r{wave}atotb",
        "source_flag_source": "r{wave}aftotb W1-W7",
        "wave_range": "W1-W9",
        "source_definition": "Respondent net value of total assets",
        "reporting_level": "respondent",
        "nominal_unit": "10,000 KRW",
        "real_base_year": "2022",
        "price_index_source": "Gateway Harmonized KLoSA cYYYYcpindex; World Bank WDI; 2010=100",
        "price_index_formula": "wealth * c2022cpindex / cPRICEYEARcpindex / 1.0",
        "couple_status_source": "",
        "couple_status_rule": "respondent-level; divisor = 1.0",
        "unknown_status_action": "not applicable",
        "map_version": MAP_VERSION,
    },
)

LOCKED_WEALTH_DESIGN = {
    "wealth_definition_HRS": "h_atotw_real2022_equiv",
    "wealth_definition_ELSA": "h_atotb_real2020_equiv",
    "wealth_definition_KLoSA": "r_atotb_real2022_individual",
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


def hrs_cpi_config_hash() -> str:
    """返回版本化 HRS CPI-U 表的内容哈希。"""
    return _stable_hash(
        {
            "series": "CPI-U US city average all items annual average",
            "base_year": 2022,
            "source_url": HRS_CPI_SOURCE_URL,
            "values": HRS_CPI_U_ANNUAL,
        }
    )


def specification() -> dict[str, Any]:
    """返回正式、可哈希的 Wealth Panel v1 测量规范。"""
    return {
        "spec_version": SPEC_VERSION,
        "output_column": "wealth_total_real_equiv",
        "formula": "nominal wealth * CPI(base year) / CPI(price year) / equivalence divisor",
        "price_year_rule": "interview_year; use nominal_year only when interview_year is missing",
        "unsupported_actual_year_action": "stop",
        "noninterview_policy": "all wealth fields are missing",
        "zero_negative_policy": "preserve",
        "source_total_policy": "use official total wealth field; do not reconstruct from components",
        "cohorts": {
            cohort: {
                "wave_range": [spec.wave_min, spec.wave_max],
                "wealth_template": spec.wealth_template,
                "flag_template": spec.flag_template,
                "flag_wave_max": spec.flag_wave_max,
                "couple_template": spec.couple_template,
                "base_year": spec.base_year,
                "reporting_level": spec.reporting_level,
                "nominal_unit": spec.nominal_unit,
                "final_unit": spec.final_unit,
                "allowed_flag_codes": sorted(FLAG_ALLOWED_CODES[cohort]),
            }
            for cohort, spec in COHORT_SPECS.items()
        },
        "hrs_cpi_config_sha256": hrs_cpi_config_hash(),
    }


def specification_hash() -> str:
    return _stable_hash(specification())


def verify_variable_map_contract() -> None:
    """确认可执行 mapping 与登记 CSV 完全一致。"""
    with VARIABLE_MAP_PATH.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != MAPPING_COLUMNS:
            raise RuntimeError("wealth_variable_map.csv 列契约不一致")
        rows = list(reader)
    if rows != list(MAPPING_RECORDS):
        raise RuntimeError("wealth_variable_map.csv 与可执行 Wealth mapping 不一致")


def verify_locked_design() -> dict[str, dict[str, str]]:
    """确认三队列财富定义已精确锁定。"""
    rows = {row["decision_id"]: row for row in _read_csv_rows(DESIGN_REGISTER_PATH)}
    resolved: dict[str, dict[str, str]] = {}
    for decision_id, selected in LOCKED_WEALTH_DESIGN.items():
        row = rows.get(decision_id)
        if row is None:
            raise RuntimeError(f"design register 缺少 {decision_id}")
        if row["status"] != "locked" or row["selected_value"] != selected:
            raise RuntimeError(
                f"{decision_id} 未按 Wealth Panel v1 锁定："
                f"status={row['status']} selected={row['selected_value']}"
            )
        resolved[decision_id] = row
    return resolved


def _registry_record(cohort: str) -> dict[str, str]:
    rows = [row for row in _read_csv_rows(REGISTRY_PATH) if row["cohort"] == cohort]
    if len(rows) != 1:
        raise RuntimeError(f"{cohort}: cohort_registry 必须恰有一行")
    if rows[0]["version_pair_verified"].upper() != "TRUE":
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
    if len(rows) != 1:
        raise RuntimeError(f"{cohort}: codebook_manifest 必须恰有一行")
    if rows[0]["usage_status"] != "authoritative":
        raise RuntimeError(f"{cohort}: codebook 不是 authoritative")
    return rows[0]


def verify_source_context(cohort: str) -> dict[str, Any]:
    """核对 raw 与 codebook 哈希，并绑定到冻结 Spine 来源。"""
    registry = _registry_record(cohort)
    codebook = _codebook_record(cohort)
    spec = COHORT_SPECS[cohort]
    if registry["person_id_source"] != spec.person_id_source:
        raise RuntimeError(f"{cohort}: person ID 与 Wealth spec 不一致")
    raw_path = Path(registry["dataset_path"])
    codebook_path = _repo_path(codebook["local_path"])
    raw_sha = sha256_file(raw_path)
    codebook_sha = sha256_file(codebook_path)
    spine_freeze = json.loads(
        (MANIFEST_ROOT / "spine" / "spine_v2_freeze.json").read_text(
            encoding="utf-8"
        )
    )
    frozen = spine_freeze["cohorts"][cohort]
    if raw_sha != frozen["source_dataset_sha256"]:
        raise RuntimeError(f"{cohort}: raw 哈希与冻结 Spine 来源不一致")
    if codebook_sha != codebook["sha256"] or codebook_sha != frozen["source_codebook_sha256"]:
        raise RuntimeError(f"{cohort}: codebook 哈希不一致")
    return {
        "raw_path": raw_path,
        "raw_sha256": raw_sha,
        "codebook_path": codebook_path,
        "codebook_sha256": codebook_sha,
        "dataset_version": registry["dataset_version_expected"],
        "codebook_version": codebook["codebook_version"],
        "person_id_source": registry["person_id_source"],
    }


def _is_stata_missing(value: Any) -> bool:
    return isinstance(value, StataMissingValue)


def _tag_counts(values: pd.Series) -> dict[str, int]:
    tags = values.loc[values.map(_is_stata_missing)].map(str)
    return {str(key): int(value) for key, value in tags.value_counts().items()}


def _numeric_series(values: pd.Series, variable: str) -> pd.Series:
    """将 Stata missing 转 NA，同时拒绝无法解释的非数值内容。"""
    tag_mask = values.map(_is_stata_missing).astype(bool)
    clean = values.mask(tag_mask)
    numeric = pd.to_numeric(clean, errors="coerce")
    invalid = clean.notna() & numeric.isna()
    if invalid.any():
        examples = clean.loc[invalid].astype(str).drop_duplicates().head(8).tolist()
        raise RuntimeError(f"{variable}: 出现无法解释的非数值：{examples}")
    return numeric.astype("float64")


def _normalise_flag_value(value: Any, cohort: str, variable: str) -> Any:
    if _is_stata_missing(value) or pd.isna(value):
        return pd.NA
    try:
        numeric = float(value)
    except (TypeError, ValueError) as error:
        raise RuntimeError(f"{variable}: 非数值 flag {value!r}") from error
    if not np.isfinite(numeric) or not numeric.is_integer():
        raise RuntimeError(f"{variable}: 非整数 flag {value!r}")
    code = int(numeric)
    if code not in FLAG_ALLOWED_CODES[cohort]:
        raise RuntimeError(f"{variable}: 未知 flag code {code}")
    return str(code)


def price_years_from_spine(spine: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """按已锁定顺序选择 price year，并返回 actual/fallback 来源。"""
    observed = spine["observed_in_wave"].eq(True).fillna(False)
    interview = pd.to_numeric(spine["interview_year"], errors="coerce")
    nominal = pd.to_numeric(spine["nominal_year"], errors="coerce")
    price_year = interview.where(interview.notna(), nominal)
    source = pd.Series(pd.NA, index=spine.index, dtype="string")
    source.loc[observed & interview.notna()] = "interview_year"
    source.loc[observed & interview.isna() & nominal.notna()] = "nominal_year_fallback"
    missing = observed & price_year.isna()
    if missing.any():
        raise RuntimeError("正式访谈行缺少 interview_year 且 nominal_year 也不可用")
    noninteger = observed & price_year.notna() & price_year.mod(1).ne(0)
    if noninteger.any():
        raise RuntimeError("price year 出现非整数")
    return price_year.astype("Int16"), source


def required_source_variables(cohort: str, spine: pd.DataFrame) -> list[str]:
    """返回正式构建所需的最小 raw 列集合。"""
    spec = COHORT_SPECS[cohort]
    variables = [spec.person_id_source]
    for wave in range(spec.wave_min, spec.wave_max + 1):
        variables.append(spec.wealth_template.format(wave=wave))
        if spec.couple_template:
            variables.append(spec.couple_template.format(wave=wave))
        if spec.flag_template and wave <= int(spec.flag_wave_max or 0):
            variables.append(spec.flag_template.format(wave=wave))
    if cohort != "HRS":
        price_year, _source = price_years_from_spine(spine)
        years = sorted(
            set(price_year.dropna().astype(int).tolist()).union({spec.base_year})
        )
        variables.extend(f"c{year}cpindex" for year in years)
    return list(dict.fromkeys(variables))


def read_raw_source(source_path: Path, cohort: str, spine: pd.DataFrame) -> pd.DataFrame:
    """从只读 DTA 直接读取最小来源列。"""
    return pd.read_stata(
        source_path,
        columns=required_source_variables(cohort, spine),
        convert_categoricals=False,
        convert_missing=True,
        preserve_dtypes=True,
    )


def extract_price_index(
    raw: pd.DataFrame,
    spine: pd.DataFrame,
    cohort: str,
) -> tuple[dict[int, float], dict[str, Any]]:
    """取得 HRS 固定 CPI-U 或 harmonized 文件内 cYYYYcpindex。"""
    spec = COHORT_SPECS[cohort]
    price_year, price_source = price_years_from_spine(spine)
    observed = spine["observed_in_wave"].eq(True).fillna(False)
    years = sorted(set(price_year.loc[observed].astype(int).tolist()).union({spec.base_year}))
    if cohort == "HRS":
        missing = sorted(set(years).difference(HRS_CPI_U_ANNUAL))
        if missing:
            raise RuntimeError(f"HRS: 实际 price year 无 CPI-U 支持：{missing}")
        values = {year: float(HRS_CPI_U_ANNUAL[year]) for year in years}
        source = {
            "type": "versioned_builder_config",
            "source_url": HRS_CPI_SOURCE_URL,
            "config_sha256": hrs_cpi_config_hash(),
        }
    else:
        values: dict[int, float] = {}
        source_columns: dict[str, float] = {}
        for year in years:
            variable = f"c{year}cpindex"
            if variable not in raw:
                raise RuntimeError(f"{cohort}: 缺少 {variable}")
            numeric = _numeric_series(raw[variable], variable).dropna()
            unique = np.unique(numeric.to_numpy(dtype=float))
            if len(unique) != 1 or unique[0] <= 0:
                raise RuntimeError(
                    f"{cohort}: {variable} 不是唯一正常数：{unique.tolist()}"
                )
            values[year] = float(unique[0])
            source_columns[variable] = float(unique[0])
        source = {
            "type": "embedded_cYYYYcpindex",
            "source_columns": source_columns,
            "config_sha256": _stable_hash(source_columns),
        }
    if spec.base_year not in values:
        raise RuntimeError(f"{cohort}: CPI 基期 {spec.base_year} 不可用")
    counts = Counter(price_source.loc[observed].dropna().astype(str))
    return values, {
        **source,
        "base_year": spec.base_year,
        "base_index": values[spec.base_year],
        "supported_price_years": years,
        "price_year_source_counts": {key: int(value) for key, value in counts.items()},
    }


def _source_long(raw: pd.DataFrame, cohort: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    """把最小 wide 来源列整理为 person-wave，不生成长期 slice。"""
    spec = COHORT_SPECS[cohort]
    person_id = normalise_person_id(raw[spec.person_id_source], cohort)
    if person_id.duplicated().any():
        raise RuntimeError(f"{cohort}: raw person ID 不唯一")
    pieces: list[pd.DataFrame] = []
    tagged = {"wealth": Counter(), "couple": Counter(), "flag": Counter()}
    system_missing = {"wealth": 0, "couple": 0, "flag": 0}
    flag_counts: Counter[str] = Counter()

    for wave in range(spec.wave_min, spec.wave_max + 1):
        wealth_variable = spec.wealth_template.format(wave=wave)
        wealth_raw = raw[wealth_variable]
        tagged["wealth"].update(_tag_counts(wealth_raw))
        wealth = _numeric_series(wealth_raw, wealth_variable)
        system_missing["wealth"] += int(
            wealth_raw.map(lambda value: pd.isna(value) and not _is_stata_missing(value)).sum()
        )

        piece = pd.DataFrame(
            {
                "person_id": person_id,
                "wave": pd.Series(wave, index=raw.index, dtype="int16"),
                "wealth_nominal": wealth,
            }
        )

        if spec.couple_template:
            couple_variable = spec.couple_template.format(wave=wave)
            couple_raw = raw[couple_variable]
            tagged["couple"].update(_tag_counts(couple_raw))
            piece["couple_status"] = _numeric_series(couple_raw, couple_variable)
            system_missing["couple"] += int(
                couple_raw.map(
                    lambda value: pd.isna(value) and not _is_stata_missing(value)
                ).sum()
            )
        else:
            piece["couple_status"] = np.nan

        if spec.flag_template and wave <= int(spec.flag_wave_max or 0):
            flag_variable = spec.flag_template.format(wave=wave)
            flag_raw = raw[flag_variable]
            tagged["flag"].update(_tag_counts(flag_raw))
            normalised = flag_raw.map(
                lambda value: _normalise_flag_value(value, cohort, flag_variable)
            ).astype("string")
            flag_counts.update(normalised.dropna().astype(str))
            piece["wealth_source_flag"] = normalised
            system_missing["flag"] += int(
                flag_raw.map(
                    lambda value: pd.isna(value) and not _is_stata_missing(value)
                ).sum()
            )
        else:
            piece["wealth_source_flag"] = pd.Series(
                pd.NA, index=raw.index, dtype="string"
            )
        pieces.append(piece)

    long = pd.concat(pieces, ignore_index=True)
    if long.duplicated(["person_id", "wave"]).any():
        raise RuntimeError(f"{cohort}: raw person-wave 来源键不唯一")
    return long, {
        "tagged_missing_counts": {
            group: dict(sorted(counter.items())) for group, counter in tagged.items()
        },
        "system_missing_counts": system_missing,
        "flag_code_counts": dict(sorted(flag_counts.items(), key=lambda item: int(item[0]))),
    }


def build_wealth_panel(
    raw: pd.DataFrame,
    spine: pd.DataFrame,
    cohort: str,
    price_index: dict[int, float],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """以 Full Spine 键与顺序为骨架生成六列 Wealth Panel。"""
    spec = COHORT_SPECS[cohort]
    if spine.duplicated(list(KEY_COLUMNS)).any():
        raise RuntimeError(f"{cohort}: Spine 主键不唯一")
    source_long, source_qc = _source_long(raw, cohort)
    raw_people = set(source_long["person_id"].unique())
    spine_people = set(spine["person_id"].astype("string").unique())
    if raw_people != spine_people:
        raise RuntimeError(f"{cohort}: raw 与 Spine 人员集合不一致")

    aligned = spine.loc[:, [*KEY_COLUMNS, "observed_in_wave", "interview_year", "nominal_year"]].copy()
    aligned["_row_order"] = np.arange(len(aligned), dtype=np.int64)
    aligned = aligned.merge(
        source_long,
        how="left",
        on=["person_id", "wave"],
        sort=False,
        validate="one_to_one",
    ).sort_values("_row_order", kind="stable")
    if len(aligned) != len(spine):
        raise RuntimeError(f"{cohort}: source join 改变了 Spine 行数")

    observed = aligned["observed_in_wave"].eq(True).fillna(False)
    price_year, price_year_source = price_years_from_spine(aligned)
    supported = price_year.loc[observed].astype(int).isin(price_index)
    if not supported.all():
        bad = sorted(price_year.loc[observed].astype(int).loc[~supported].unique())
        raise RuntimeError(f"{cohort}: 实际 price year 无 CPI 支持：{bad}")

    wealth = aligned["wealth_nominal"].astype("float64")
    valid_wealth = observed & wealth.notna()
    divisor = pd.Series(pd.NA, index=aligned.index, dtype="Float64")
    if spec.couple_template:
        couple = aligned["couple_status"]
        invalid = valid_wealth & ~couple.isin([0.0, 1.0])
        if invalid.any():
            values = couple.loc[invalid].astype("object").drop_duplicates().head(8).tolist()
            raise RuntimeError(
                f"{cohort}: 财富有值但 couple status 不是 0/1：{values}"
            )
        divisor.loc[valid_wealth & couple.eq(0)] = 1.0
        divisor.loc[valid_wealth & couple.eq(1)] = 1.5
    else:
        divisor.loc[valid_wealth] = 1.0

    base_index = float(price_index[spec.base_year])
    row_index = price_year.map(price_index).astype("Float64")
    real_equiv = pd.Series(np.nan, index=aligned.index, dtype="float64")
    real_equiv.loc[valid_wealth] = (
        wealth.loc[valid_wealth]
        * base_index
        / row_index.loc[valid_wealth].astype(float)
        / divisor.loc[valid_wealth].astype(float)
    )

    flag = aligned["wealth_source_flag"].astype("string")
    flag.loc[~observed] = pd.NA
    divisor.loc[~observed] = pd.NA
    real_equiv.loc[~observed] = np.nan

    output = spine.loc[:, list(KEY_COLUMNS)].copy().astype(
        {"cohort": "string", "person_id": "string", "wave": "int16"}
    )
    output["wealth_total_real_equiv"] = real_equiv.to_numpy(dtype=float)
    output["wealth_equiv_divisor"] = divisor.array
    output["wealth_source_flag"] = flag.array
    output = output.loc[:, list(WEALTH_PANEL_COLUMNS)]

    expected = wealth.loc[valid_wealth] * base_index / row_index.loc[valid_wealth].astype(float) / divisor.loc[valid_wealth].astype(float)
    np.testing.assert_allclose(
        output.loc[valid_wealth, "wealth_total_real_equiv"],
        expected,
        rtol=0,
        atol=0,
    )
    source_counts = Counter(price_year_source.loc[observed].dropna().astype(str))
    return output, {
        **source_qc,
        "price_year_source_counts": {
            key: int(value) for key, value in source_counts.items()
        },
        "valid_wealth_n": int(valid_wealth.sum()),
        "zero_wealth_n": int((valid_wealth & wealth.eq(0)).sum()),
        "negative_wealth_n": int((valid_wealth & wealth.lt(0)).sum()),
        "positive_wealth_n": int((valid_wealth & wealth.gt(0)).sum()),
        "formula_max_abs_difference": 0.0,
    }


def validate_wealth_panel(
    frame: pd.DataFrame,
    spine: pd.DataFrame,
    cohort: str,
) -> None:
    """验证六列 schema、键、掩蔽、divisor 与范围边界。"""
    if list(frame.columns) != list(WEALTH_PANEL_COLUMNS):
        raise RuntimeError(f"{cohort}: Wealth Panel 列顺序不一致")
    actual_dtypes = {column: str(frame[column].dtype) for column in frame.columns}
    if actual_dtypes != WEALTH_PANEL_DTYPES:
        raise RuntimeError(
            f"{cohort}: Wealth Panel dtype 不一致：{actual_dtypes}"
        )
    if len(frame) != len(spine) or frame.duplicated(list(KEY_COLUMNS)).any():
        raise RuntimeError(f"{cohort}: Wealth Panel 行数或主键不正确")
    pd.testing.assert_frame_equal(
        frame.loc[:, list(KEY_COLUMNS)],
        spine.loc[:, list(KEY_COLUMNS)].astype(
            {"cohort": "string", "person_id": "string", "wave": "int16"}
        ),
        check_dtype=True,
        check_exact=True,
    )
    noninterview = ~spine["observed_in_wave"].eq(True).fillna(False)
    if not frame.loc[noninterview, list(WEALTH_PANEL_COLUMNS[3:])].isna().all(axis=None):
        raise RuntimeError(f"{cohort}: 非访谈行出现财富值、divisor 或 flag")
    valid = frame["wealth_total_real_equiv"].notna()
    if frame.loc[valid, "wealth_equiv_divisor"].isna().any():
        raise RuntimeError(f"{cohort}: 有效财富缺少 divisor")
    if not np.isfinite(frame.loc[valid, "wealth_total_real_equiv"].to_numpy(dtype=float)).all():
        raise RuntimeError(f"{cohort}: 财富结果含无穷值")
    divisor_values = set(frame["wealth_equiv_divisor"].dropna().astype(float))
    expected = {1.0} if cohort == "KLoSA" else {1.0, 1.5}
    if not divisor_values.issubset(expected):
        raise RuntimeError(f"{cohort}: divisor 出现非法值 {divisor_values}")
    if cohort == "KLoSA" and divisor_values != {1.0}:
        raise RuntimeError("KLoSA: 有效财富 divisor 必须恒为 1.0")
    forbidden_prefixes = (
        "fi_",
        "eligibility_",
        "lag_",
        "shock_",
        "recovery_",
        "sample_",
        "weight_",
    )
    if any(column.startswith(forbidden_prefixes) for column in frame.columns):
        raise RuntimeError(f"{cohort}: Wealth Panel 含越界字段")


def _rate_text(numerator: int, denominator: int) -> str:
    """生成便于人读的 n (%)，不以零替代未定义比例。"""
    if denominator == 0:
        return f"{numerator:,} (NA)"
    return f"{numerator:,} ({100 * numerator / denominator:.1f}%)"


def _finite(value: Any) -> float | None:
    if value is None or pd.isna(value) or not np.isfinite(float(value)):
        return None
    return float(value)


def wave_qc_rows(
    panels: dict[str, pd.DataFrame],
    spines: dict[str, pd.DataFrame],
) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    """逐波汇总正式访谈中的财富支持与分布。"""
    visible: list[dict[str, Any]] = []
    machine: dict[str, list[dict[str, Any]]] = {}
    for cohort in COHORTS:
        panel = panels[cohort]
        spine = spines[cohort]
        rows: list[dict[str, Any]] = []
        for wave in range(COHORT_SPECS[cohort].wave_min, COHORT_SPECS[cohort].wave_max + 1):
            observed = spine["wave"].eq(wave) & spine["observed_in_wave"].eq(True).fillna(False)
            wealth = panel.loc[observed, "wealth_total_real_equiv"]
            valid = wealth.notna()
            observed_n = int(observed.sum())
            valid_n = int(valid.sum())
            positive_n = int(wealth.gt(0).sum())
            zero_n = int(wealth.eq(0).sum())
            negative_n = int(wealth.lt(0).sum())
            flag_n = int(panel.loc[observed, "wealth_source_flag"].notna().sum())
            valid_values = wealth.loc[valid].astype(float)
            record = {
                "wave": wave,
                "observed_n": observed_n,
                "valid_wealth_n": valid_n,
                "valid_wealth_rate": None if observed_n == 0 else valid_n / observed_n,
                "positive_n": positive_n,
                "positive_rate_observed": None if observed_n == 0 else positive_n / observed_n,
                "zero_n": zero_n,
                "zero_rate_observed": None if observed_n == 0 else zero_n / observed_n,
                "negative_n": negative_n,
                "negative_rate_observed": None if observed_n == 0 else negative_n / observed_n,
                "flag_available_n": flag_n,
                "flag_available_rate": None if observed_n == 0 else flag_n / observed_n,
                "median": _finite(valid_values.median()) if valid_n else None,
                "p1": _finite(valid_values.quantile(0.01)) if valid_n else None,
                "p99": _finite(valid_values.quantile(0.99)) if valid_n else None,
            }
            rows.append(record)
            flag_text = (
                "Not released"
                if COHORT_SPECS[cohort].flag_template is None
                or wave > int(COHORT_SPECS[cohort].flag_wave_max or 0)
                else _rate_text(flag_n, observed_n)
            )
            visible.append(
                {
                    "Cohort": cohort,
                    "Wave": wave,
                    "Real unit": COHORT_SPECS[cohort].final_unit,
                    "Observed N": observed_n,
                    "Valid wealth, n (%)": _rate_text(valid_n, observed_n),
                    "Positive, n (%)": _rate_text(positive_n, observed_n),
                    "Zero, n (%)": _rate_text(zero_n, observed_n),
                    "Negative, n (%)": _rate_text(negative_n, observed_n),
                    "Flag available, n (%)": flag_text,
                    "Median": record["median"],
                    "P1": record["p1"],
                    "P99": record["p99"],
                }
            )
        machine[cohort] = rows
    return visible, machine


def interval_readiness_rows(
    panels: dict[str, pd.DataFrame],
    load_candidate_intervals: Callable[
        [str], tuple[pd.DataFrame, Path, dict[str, Any]]
    ],
    load_eligibility_base: Callable[
        [str], tuple[pd.DataFrame, Path, dict[str, Any]]
    ],
) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]], dict[str, Any]]:
    """以 Candidate/Eligibility 计算潜在区间支持，不改变 Wealth Panel。"""
    visible: list[dict[str, Any]] = []
    machine: dict[str, list[dict[str, Any]]] = {}
    qc_context: dict[str, Any] = {}
    for cohort in COHORTS:
        candidate, candidate_path, candidate_manifest = load_candidate_intervals(cohort)
        eligibility, eligibility_path, eligibility_manifest = load_eligibility_base(cohort)
        pd.testing.assert_frame_equal(
            candidate.loc[:, ["cohort", "person_id", "candidate_shock_wave"]],
            eligibility.loc[:, ["cohort", "person_id", "candidate_shock_wave"]],
            check_dtype=True,
            check_exact=True,
        )
        ledger = candidate.merge(
            eligibility,
            on=["cohort", "person_id", "candidate_shock_wave"],
            how="left",
            validate="one_to_one",
            sort=False,
        )
        wealth_lookup = panels[cohort].loc[
            :,
            ["person_id", "wave", "wealth_total_real_equiv", "wealth_equiv_divisor"],
        ]

        def attach_role(frame: pd.DataFrame, wave_column: str, prefix: str) -> pd.DataFrame:
            lookup = wealth_lookup.rename(
                columns={
                    "wave": wave_column,
                    "wealth_total_real_equiv": f"{prefix}_wealth",
                    "wealth_equiv_divisor": f"{prefix}_divisor",
                }
            )
            return frame.merge(
                lookup,
                on=["person_id", wave_column],
                how="left",
                validate="many_to_one",
                sort=False,
            )

        ledger = attach_role(ledger, "tminus1_wave", "tminus1")
        ledger = attach_role(ledger, "candidate_shock_wave", "t")
        ledger = attach_role(ledger, "tplus1_wave", "tplus1")
        rows: list[dict[str, Any]] = []
        for wave, group in ledger.groupby("candidate_shock_wave", sort=True):
            passed = group["structural_eligibility_status"].eq("pass")
            denominator = int(passed.sum())
            valid_pair = (
                passed
                & group["tminus1_wealth"].notna()
                & group["t_wealth"].notna()
            )
            valid_pair_n = int(valid_pair.sum())
            positive_lag = valid_pair & group["tminus1_wealth"].gt(0)
            positive_lag_n = int(positive_lag.sum())
            potential_tplus1 = positive_lag & group["tplus1_wealth"].notna()
            potential_tplus1_n = int(potential_tplus1.sum())
            divisor_changed = (
                valid_pair
                & group["tminus1_divisor"].notna()
                & group["t_divisor"].notna()
                & group["tminus1_divisor"].ne(group["t_divisor"])
            )
            divisor_changed_n = int(divisor_changed.sum())
            record = {
                "candidate_shock_wave": int(wave),
                "structural_pass_n": denominator,
                "valid_wealth_pair_n": valid_pair_n,
                "valid_wealth_pair_rate": None if denominator == 0 else valid_pair_n / denominator,
                "positive_tminus1_n": positive_lag_n,
                "positive_tminus1_rate": None if valid_pair_n == 0 else positive_lag_n / valid_pair_n,
                "potential_tplus1_wealth_n": potential_tplus1_n,
                "potential_tplus1_wealth_rate": None if positive_lag_n == 0 else potential_tplus1_n / positive_lag_n,
                "divisor_changed_n": divisor_changed_n,
                "divisor_changed_rate": None if valid_pair_n == 0 else divisor_changed_n / valid_pair_n,
            }
            rows.append(record)
            visible.append(
                {
                    "Cohort": cohort,
                    "Candidate shock wave": int(wave),
                    "Eligibility pass N": denominator,
                    "Valid wealth pair, n (%)": _rate_text(valid_pair_n, denominator),
                    "Positive t-1 wealth, n (%)": _rate_text(
                        positive_lag_n, valid_pair_n
                    ),
                    "t+1 wealth available, n (%)": (
                        "Not available by design"
                        if int(wave) == COHORT_SPECS[cohort].wave_max
                        else _rate_text(potential_tplus1_n, positive_lag_n)
                    ),
                    "Divisor changed, n (%)": _rate_text(
                        divisor_changed_n, valid_pair_n
                    ),
                }
            )
        machine[cohort] = rows
        qc_context[cohort] = {
            "candidate_intervals": {
                "path": _relative(candidate_path),
                "sha256": candidate_manifest["output"]["sha256"],
            },
            "eligibility_base": {
                "path": _relative(eligibility_path),
                "sha256": eligibility_manifest["output"]["sha256"],
            },
            "denominator_definitions": {
                "valid_wealth_pair_rate": (
                    "Eligibility Base structural_eligibility_status == pass"
                ),
                "positive_tminus1_rate": "valid wealth pairs",
                "potential_tplus1_wealth_rate": (
                    "valid wealth pairs with positive t-1 wealth"
                ),
                "divisor_changed_rate": "valid wealth pairs",
            },
            "not_a_build_input": True,
        }
    return visible, machine, qc_context


def source_contract_rows(
    price_contexts: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """生成人类阅读的 source contract。"""
    rows: list[dict[str, Any]] = []
    for cohort in COHORTS:
        spec = COHORT_SPECS[cohort]
        mapping = next(row for row in MAPPING_RECORDS if row["cohort"] == cohort)
        supported_years = price_contexts[cohort]["supported_price_years"]
        if not supported_years:
            raise RuntimeError(f"{cohort} 没有已核验的 price year")
        if cohort == "HRS":
            flag_coverage = "Not released"
            allowed_flags = "Not applicable"
            coverage = "1992–2023 annual CPI-U"
            reference = "HRS codebook; BLS CPI-U"
        elif cohort == "ELSA":
            flag_coverage = "W1-W10"
            allowed_flags = ", ".join(str(code) for code in sorted(FLAG_ALLOWED_CODES[cohort]))
            coverage = "2002–2023 annual"
            reference = "Gateway Harmonized ELSA"
        else:
            flag_coverage = "W1-W7; not released W8-W9"
            allowed_flags = ", ".join(str(code) for code in sorted(FLAG_ALLOWED_CODES[cohort]))
            coverage = "2006–2022 survey years (biennial)"
            reference = "Gateway Harmonized KLoSA"
        rows.append(
            {
                "Cohort": cohort,
                "Wealth source": mapping["total_wealth_source"],
                "Definition": mapping["source_definition"],
                "Reporting level": spec.reporting_level,
                "Nominal unit": spec.nominal_unit,
                "Real base": str(spec.base_year),
                "Price index": mapping["price_index_source"],
                "Formula": mapping["price_index_formula"],
                "Price-year rule": "interview_year; nominal_year only if interview_year is missing",
                "Couple source": mapping["couple_status_source"] or "Not applicable",
                "Equivalence rule": mapping["couple_status_rule"],
                "Source flag": mapping["source_flag_source"] or "Not released",
                "Flag coverage": flag_coverage,
                "Allowed flag codes": allowed_flags,
                "Price-year coverage": coverage,
                "Source": reference,
            }
        )
    return rows


def _percentage_from_rate_text(value: str) -> float:
    """从 `n (x.x%)` 提取百分比，供人类审阅注释使用。"""
    if "(" not in value or not value.endswith("%)"):
        raise RuntimeError(f"无法解析百分比：{value}")
    return float(value.rsplit("(", maxsplit=1)[1][:-2])


def review_audit_notes(
    wave_rows: list[dict[str, Any]], readiness_rows: list[dict[str, Any]]
) -> dict[str, Any]:
    """从可见 QC 行生成简短、可复算的 review 与下一阶段审计说明。"""

    def get_row(rows: list[dict[str, Any]], cohort: str, wave: int) -> dict[str, Any]:
        wave_key = "Wave" if "Wave" in rows[0] else "Candidate shock wave"
        return next(
            row
            for row in rows
            if row["Cohort"] == cohort and int(row[wave_key]) == wave
        )

    hrs9 = get_row(wave_rows, "HRS", 9)
    hrs10 = get_row(wave_rows, "HRS", 10)
    klosa5 = get_row(wave_rows, "KLoSA", 5)
    klosa9 = get_row(wave_rows, "KLoSA", 9)
    elsa9_interval = get_row(readiness_rows, "ELSA", 9)
    klosa8_interval = get_row(readiness_rows, "KLoSA", 8)

    def rate(row: dict[str, Any], column: str) -> float:
        return _percentage_from_rate_text(str(row[column]))

    divisor_ranges: dict[str, tuple[float, float]] = {}
    for cohort in ("HRS", "ELSA"):
        values = [
            rate(row, "Divisor changed, n (%)")
            for row in readiness_rows
            if row["Cohort"] == cohort
        ]
        divisor_ranges[cohort] = (min(values), max(values))

    wave_note = (
        "Exposure-stage audit points: HRS W9→W10 positive wealth was "
        f"{rate(hrs9, 'Positive, n (%)'):.1f}%→"
        f"{rate(hrs10, 'Positive, n (%)'):.1f}%, negative wealth "
        f"{rate(hrs9, 'Negative, n (%)'):.1f}%→"
        f"{rate(hrs10, 'Negative, n (%)'):.1f}%, and median 2022 "
        f"USD-equivalent wealth {float(hrs9['Median']):,.0f}→"
        f"{float(hrs10['Median']):,.0f}. KLoSA valid wealth was "
        f"{rate(klosa5, 'Valid wealth, n (%)'):.1f}% at W5 and "
        f"{rate(klosa9, 'Valid wealth, n (%)'):.1f}% at W9. "
        "These observations do not trigger recoding or winsorisation."
    )
    hrs_range = divisor_ranges["HRS"]
    elsa_range = divisor_ranges["ELSA"]
    readiness_note = (
        "Exposure-stage audit points: ELSA W9 and KLoSA W8 had lower t+1 "
        "wealth availability ("
        f"{rate(elsa9_interval, 't+1 wealth available, n (%)'):.1f}% and "
        f"{rate(klosa8_interval, 't+1 wealth available, n (%)'):.1f}%, "
        "respectively). "
        f"Across waves, divisor changes occurred in {hrs_range[0]:.1f}–"
        f"{hrs_range[1]:.1f}% of HRS and {elsa_range[0]:.1f}–"
        f"{elsa_range[1]:.1f}% of ELSA valid wealth pairs. Cross-tabulate "
        "shock and recovery classifications by divisor change and candidate "
        "shock wave; do not exclude these intervals a priori."
    )
    return {
        "source_urls": [
            f"HRS codebook: {HRS_CODEBOOK_URL}",
            f"BLS CPI-U table: {HRS_CPI_SOURCE_URL}",
            f"Gateway sources: {ELSA_SOURCE_URL} | {KLOSA_SOURCE_URL}",
        ],
        "wave_audit": wave_note,
        "readiness_audit": readiness_note,
    }


WORKBOOK_BUILDER = r'''
import fs from "node:fs/promises";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const [payloadPath, outputPath, previewDir] = process.argv.slice(2);
const payload = JSON.parse(await fs.readFile(payloadPath, "utf8"));
const workbook = Workbook.create();
const fontFamily = "Arial";

function addReaderSheet({name, title, context, headers, rows, tableName, widths, numericFormats, bodyHeight}) {
  const sheet = workbook.worksheets.add(name);
  sheet.showGridLines = false;
  sheet.getRange("A1").values = [[title]];
  sheet.getRange("A1").format.font = { name: fontFamily, size: 14, bold: true, color: "#1F2937" };
  const contextRange = sheet.getRangeByIndexes(1, 0, 1, headers.length);
  contextRange.merge();
  contextRange.values = [[context]];
  contextRange.format.font = { name: fontFamily, size: 10, italic: true, color: "#4B5563" };
  contextRange.format.wrapText = true;
  contextRange.format.rowHeightPx = 30;
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
  header.format.rowHeightPx = 42;
  for (let index = 0; index < widths.length; index += 1) {
    sheet.getRangeByIndexes(0, index, matrix.length + 3, 1).format.columnWidthPx = widths[index];
  }
  if (rows.length > 0) {
    const body = sheet.getRangeByIndexes(4, 0, rows.length, headers.length);
    body.format.rowHeightPx = bodyHeight ?? 34;
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

addReaderSheet({
  name: "1_Source Contract",
  title: "Wealth source contract",
  context: "One official total-wealth field per cohort. Price year uses interview year, with nominal year only when interview year is missing. Exact price-year lists remain in the cohort manifests.",
  headers: payload.source_contract.headers,
  rows: payload.source_contract.rows,
  tableName: "WealthSourceContract",
  widths: [65, 95, 205, 95, 105, 65, 185, 190, 155, 95, 125, 105, 110, 110, 135, 145],
  numericFormats: {},
  bodyHeight: 72,
});
const sourceSheet = workbook.worksheets.getItem("1_Source Contract");
for (let row = 0; row < payload.notes.source_urls.length; row += 1) {
  const note = sourceSheet.getRangeByIndexes(8 + row, 0, 1, payload.source_contract.headers.length);
  note.merge();
  note.values = [[payload.notes.source_urls[row]]];
  note.format.font = { name: fontFamily, size: 9, italic: true, color: "#4B5563" };
  note.format.wrapText = true;
  note.format.rowHeightPx = 24;
}

addReaderSheet({
  name: "2_Wave QC",
  title: "Wealth measurement by wave",
  context: "All percentages use interviewed person-waves as the denominator. Distribution summaries use non-missing real equivalised wealth.",
  headers: payload.wave_qc.headers,
  rows: payload.wave_qc.rows,
  tableName: "WealthWaveQC",
  widths: [70, 55, 155, 90, 125, 125, 110, 115, 145, 105, 105, 105],
  numericFormats: {"Wave": "0", "Observed N": "#,##0", "Median": "#,##0.0", "P1": "#,##0.0", "P99": "#,##0.0"},
  bodyHeight: 38,
});
const waveSheet = workbook.worksheets.getItem("2_Wave QC");
const waveNoteRow = 4 + payload.wave_qc.rows.length + 1;
const waveNote = waveSheet.getRangeByIndexes(
  waveNoteRow,
  0,
  1,
  payload.wave_qc.headers.length,
);
waveNote.merge();
waveNote.values = [[payload.notes.wave_audit]];
waveNote.format.font = { name: fontFamily, size: 9, italic: true, color: "#4B5563" };
waveNote.format.wrapText = true;
waveNote.format.rowHeightPx = 42;

addReaderSheet({
  name: "3_Interval Readiness",
  title: "Potential interval readiness",
  context: "Valid wealth pair is reported as a percentage of structurally eligible intervals; positive t-1 wealth as a percentage of valid wealth pairs; t+1 wealth availability as a percentage of intervals with positive t-1 wealth; and divisor changes as a percentage of valid wealth pairs. The final survey wave has no t+1 by design.",
  headers: payload.interval_readiness.headers,
  rows: payload.interval_readiness.rows,
  tableName: "WealthIntervalReadiness",
  widths: [70, 120, 110, 150, 160, 175, 150],
  numericFormats: {"Candidate shock wave": "0", "Eligibility pass N": "#,##0"},
  bodyHeight: 38,
});
const readinessSheet = workbook.worksheets.getItem("3_Interval Readiness");
readinessSheet.getRange("A2:G2").format.rowHeightPx = 52;
const readinessNoteRow = 4 + payload.interval_readiness.rows.length + 1;
const readinessNote = readinessSheet.getRangeByIndexes(
  readinessNoteRow,
  0,
  1,
  payload.interval_readiness.headers.length,
);
readinessNote.merge();
readinessNote.values = [[payload.notes.readiness_audit]];
readinessNote.format.font = { name: fontFamily, size: 9, italic: true, color: "#4B5563" };
readinessNote.format.wrapText = true;
readinessNote.format.rowHeightPx = 42;

const overview = await workbook.inspect({
  kind: "workbook,sheet,table",
  maxChars: 5000,
  tableMaxRows: 4,
  tableMaxCols: 8,
});
const formulaErrors = await workbook.inspect({
  kind: "match",
  searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!|#SPILL!|#CALC!",
  options: { useRegex: true, maxResults: 100 },
  summary: "final formula error scan",
});
await fs.mkdir(previewDir, { recursive: true });
for (const sheetName of ["1_Source Contract", "2_Wave QC", "3_Interval Readiness"]) {
  const preview = await workbook.render({ sheetName, autoCrop: "all", scale: 1.5, format: "png" });
  const safeName = sheetName.replaceAll(" ", "_");
  await fs.writeFile(`${previewDir}/${safeName}.png`, new Uint8Array(await preview.arrayBuffer()));
}
const xlsx = await SpreadsheetFile.exportXlsx(workbook);
await xlsx.save(outputPath);
console.log(JSON.stringify({overview: overview.ndjson, formulaErrors: formulaErrors.ndjson}));
'''


def _runtime_paths() -> tuple[Path, Path]:
    """取得 bundled Node 与 artifact-tool runtime。"""
    runtime = Path.home() / ".cache" / "codex-runtimes" / "codex-primary-runtime" / "dependencies"
    node = runtime / "node" / "bin" / "node.exe"
    node_modules = runtime / "node" / "node_modules"
    if not node.is_file() or not node_modules.is_dir():
        raise RuntimeError("bundled artifact-tool runtime 不可用")
    return node, node_modules


def _create_node_modules_link(link: Path, target: Path) -> None:
    """在系统临时目录创建 artifact-tool 所需目录链接。"""
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
    source_rows: list[dict[str, Any]],
    wave_rows: list[dict[str, Any]],
    readiness_rows: list[dict[str, Any]],
    output_path: Path,
    preview_dir: Path,
) -> dict[str, Any]:
    """用 artifact-tool 生成、检查并渲染三表 workbook。"""
    payload = {
        "source_contract": {
            "headers": list(source_rows[0]) if source_rows else [],
            "rows": source_rows,
        },
        "wave_qc": {
            "headers": list(wave_rows[0]) if wave_rows else [],
            "rows": wave_rows,
        },
        "interval_readiness": {
            "headers": list(readiness_rows[0]) if readiness_rows else [],
            "rows": readiness_rows,
        },
        "notes": review_audit_notes(wave_rows, readiness_rows),
    }
    node, node_modules = _runtime_paths()
    with tempfile.TemporaryDirectory(prefix="wealth_panel_xlsx_") as tmp_name:
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
                "artifact-tool workbook 构建失败：\n"
                + (completed.stdout or "")
                + "\n"
                + (completed.stderr or "")
            )
        if not output_path.is_file():
            raise RuntimeError("artifact-tool 未生成 review workbook")
        errors = completed.stdout.strip()
        if any(token in errors for token in ("#REF!", "#DIV/0!", "#VALUE!", "#NAME?")):
            raise RuntimeError("review workbook 公式错误扫描未通过")
        return {
            "artifact_tool_stdout": errors,
            "preview_files": sorted(str(path) for path in preview_dir.glob("*.png")),
        }


DICTIONARY_TEXT = """# Wealth Panel v1 数据字典

## 数据粒度与主键

每一行表示一个队列中的一个人次波次（person-wave）。主键固定为 `cohort + person_id + wave`，行集合和行顺序与冻结 Full Spine v2 完全一致。Candidate Intervals v3 和 Eligibility Base v2 只用于区间 readiness QC，不参与构建、筛行或数值计算。

## 六列接口

- `cohort`：`HRS`、`ELSA` 或 `KLoSA`。
- `person_id`：队列内稳定的字符串型参与者标识。
- `wave`：队列内登记波次。
- `wealth_total_real_equiv`：官方总财富字段经规定的年度 CPI 调整和等值化处理后的财富。HRS 为 2022 USD；ELSA 为 2020 GBP；KLoSA 为 2022 KRW、保留原 10,000 韩元单位。各队列绝对金额不应直接跨国比较。
- `wealth_equiv_divisor`：本波实际使用的等值化除数。HRS/ELSA 单身为 1.0、coupled 为 1.5；KLoSA respondent-level 财富固定为 1.0。
- `wealth_source_flag`：官方总财富来源 flag 的原始代码字符串。HRS 未发布同类 flag；KLoSA W8-W9 未发布，因此为 `NA`。代码含义见 review workbook。

## 计算与缺失

计算式为 `nominal wealth × CPI(base year) / CPI(price year) / divisor`。`price year` 优先使用 Full Spine 的实际 `interview_year`，仅当其缺失时使用 `nominal_year`。实际年份存在但 CPI 不支持时构建停止。

`NA` 表示未访谈、未测、不可用、特殊缺失，或财富有值但必要来源未通过契约验证；它不表示财富为零。零财富和负财富是有效数值并原样保留。本层不插值、不前移、不截尾、不做 winsorisation，也不生成 lag、变化率、rank、shock、recovery 或任何样本资格。
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


def _git_output(arguments: list[str], *, binary: bool = False) -> bytes | str:
    completed = subprocess.run(
        ["git", *arguments], cwd=ROOT, check=True, capture_output=True
    )
    if binary:
        return completed.stdout
    return completed.stdout.decode("utf-8", errors="replace").strip()


def git_provenance() -> dict[str, Any]:
    """记录含用户既有 dirty change 的可复现代码状态。"""
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
        normalised = relative.replace("\\", "/")
        if normalised.startswith("00_admin/panel_manifests/"):
            continue
        path = ROOT / relative
        if path.is_file() and path.suffix.lower() != ".pdf":
            inventory.append(f"{normalised}\t{sha256_file(path)}")
    digest = hashlib.sha256()
    digest.update(diff)
    digest.update("\n".join(inventory).encode("utf-8"))
    return {
        "git_commit": commit,
        "git_dirty": bool(status),
        "git_diff_sha256": digest.hexdigest(),
        "untracked_build_file_inventory": inventory,
    }


def _final_output_path(cohort: str) -> Path:
    return (
        DERIVED_ROOT
        / cohort
        / "06_wealth_panel"
        / f"{cohort.lower()}_wealth_panel_v1.parquet"
    )


def _manifest_path(cohort: str) -> Path:
    return WEALTH_MANIFEST_ROOT / f"{cohort.lower()}_wealth_panel_v1.json"


def _verify_release_absent() -> None:
    """冻结后禁止覆盖，同名孤立资产也必须人工处理。"""
    paths = [
        *(_final_output_path(cohort) for cohort in COHORTS),
        *(_manifest_path(cohort) for cohort in COHORTS),
        REVIEW_PATH,
        DICTIONARY_PATH,
        FREEZE_PATH,
    ]
    existing = [str(path) for path in paths if path.exists()]
    if existing:
        raise RuntimeError(
            "Wealth Panel v1 已存在或存在同名孤立资产，拒绝覆盖："
            + "; ".join(existing)
        )


def verify_fi_panel_v2_unchanged() -> dict[str, Any]:
    """核对 FI v2 仅作为受保护的非输入资产保持冻结。"""
    freeze = json.loads(FI_FREEZE_PATH.read_text(encoding="utf-8"))
    if freeze.get("freeze_status") != "FROZEN":
        raise RuntimeError("FI Panel v2 未冻结")
    cohorts: dict[str, Any] = {}
    for cohort, record in freeze["cohorts"].items():
        parquet = _repo_path(record["parquet_path"])
        manifest = _repo_path(record["manifest_path"])
        parquet_sha = sha256_file(parquet)
        manifest_sha = sha256_file(manifest)
        if parquet_sha != record["parquet_sha256"]:
            raise RuntimeError(f"{cohort}: FI Panel v2 Parquet 哈希不一致")
        if manifest_sha != record["manifest_sha256"]:
            raise RuntimeError(f"{cohort}: FI Panel v2 manifest 哈希不一致")
        cohorts[cohort] = {
            "parquet_path": _relative(parquet),
            "parquet_sha256": parquet_sha,
            "manifest_path": _relative(manifest),
            "manifest_sha256": manifest_sha,
        }
    review = _repo_path(freeze["review_workbook"]["path"])
    dictionary = _repo_path(freeze["data_dictionary"]["path"])
    if sha256_file(review) != freeze["review_workbook"]["sha256"]:
        raise RuntimeError("FI Panel v2 review workbook 哈希不一致")
    if sha256_file(dictionary) != freeze["data_dictionary"]["sha256"]:
        raise RuntimeError("FI Panel v2 dictionary 哈希不一致")
    return {
        "freeze_path": _relative(FI_FREEZE_PATH),
        "freeze_sha256": sha256_file(FI_FREEZE_PATH),
        "cohorts": cohorts,
        "review_workbook_sha256": freeze["review_workbook"]["sha256"],
        "data_dictionary_sha256": freeze["data_dictionary"]["sha256"],
        "is_build_input": False,
    }


def _normalise_source_contexts(
    contexts: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    return {
        cohort: {
            "raw_path": str(context["raw_path"].resolve()),
            "raw_sha256": context["raw_sha256"],
            "codebook_path": str(context["codebook_path"].resolve()),
            "codebook_sha256": context["codebook_sha256"],
            "dataset_version": context["dataset_version"],
            "codebook_version": context["codebook_version"],
        }
        for cohort, context in contexts.items()
    }


def protected_snapshot(
    *,
    load_spine: Callable[[str], tuple[pd.DataFrame, Path, dict[str, Any]]],
    load_candidate_intervals: Callable[
        [str], tuple[pd.DataFrame, Path, dict[str, Any]]
    ],
    load_eligibility_base: Callable[
        [str], tuple[pd.DataFrame, Path, dict[str, Any]]
    ],
) -> dict[str, Any]:
    """对 raw、codebook、三张 pre-panel 和 FI v2 做完整快照。"""
    sources = {cohort: verify_source_context(cohort) for cohort in COHORTS}
    prepanels = verify_prepanels(
        load_spine, load_candidate_intervals, load_eligibility_base
    )
    return {
        "sources": _normalise_source_contexts(sources),
        "prepanels": prepanels,
        "prepanel_freezes": {
            name: {
                "path": _relative(path),
                "sha256": sha256_file(path),
            }
            for name, path in {
                "spine_v2": MANIFEST_ROOT / "spine" / "spine_v2_freeze.json",
                "candidate_intervals_v3": MANIFEST_ROOT
                / "candidate_intervals"
                / "candidate_intervals_v3_freeze.json",
                "eligibility_base_v2": MANIFEST_ROOT
                / "eligibility_base"
                / "eligibility_base_v2_freeze.json",
            }.items()
        },
        "fi_panel_v2": verify_fi_panel_v2_unchanged(),
    }


def _source_contexts_from_snapshot(snapshot: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """把序列化快照恢复为读取 raw 所需的路径对象。"""
    contexts: dict[str, dict[str, Any]] = {}
    for cohort, record in snapshot["sources"].items():
        contexts[cohort] = {
            **record,
            "raw_path": Path(record["raw_path"]),
            "codebook_path": Path(record["codebook_path"]),
            "person_id_source": COHORT_SPECS[cohort].person_id_source,
        }
    return contexts


def prepare_measurements(
    *,
    source_contexts: dict[str, dict[str, Any]],
    load_spine: Callable[[str], tuple[pd.DataFrame, Path, dict[str, Any]]],
    load_candidate_intervals: Callable[
        [str], tuple[pd.DataFrame, Path, dict[str, Any]]
    ],
    load_eligibility_base: Callable[
        [str], tuple[pd.DataFrame, Path, dict[str, Any]]
    ],
) -> dict[str, Any]:
    """从冻结 Spine 和只读 raw 构建内存 panel 与全部聚合 QC。"""
    panels: dict[str, pd.DataFrame] = {}
    spines: dict[str, pd.DataFrame] = {}
    spine_inputs: dict[str, dict[str, Any]] = {}
    source_qc: dict[str, dict[str, Any]] = {}
    price_contexts: dict[str, dict[str, Any]] = {}

    for cohort in COHORTS:
        spine, spine_path, spine_manifest = load_spine(cohort)
        raw = read_raw_source(source_contexts[cohort]["raw_path"], cohort, spine)
        price_index, price_context = extract_price_index(raw, spine, cohort)
        panel, diagnostics = build_wealth_panel(raw, spine, cohort, price_index)
        validate_wealth_panel(panel, spine, cohort)
        panels[cohort] = panel
        spines[cohort] = spine
        source_qc[cohort] = diagnostics
        price_contexts[cohort] = price_context
        spine_inputs[cohort] = {
            "path": _relative(spine_path),
            "sha256": spine_manifest["output"]["sha256"],
            "manifest_path": _relative(
                MANIFEST_ROOT / "spine" / f"{cohort.lower()}_spine.json"
            ),
        }
        del raw

    wave_visible, wave_machine = wave_qc_rows(panels, spines)
    readiness_visible, readiness_machine, qc_context = interval_readiness_rows(
        panels, load_candidate_intervals, load_eligibility_base
    )
    return {
        "panels": panels,
        "spines": spines,
        "spine_inputs": spine_inputs,
        "source_qc": source_qc,
        "price_contexts": price_contexts,
        "source_visible": source_contract_rows(price_contexts),
        "wave_visible": wave_visible,
        "wave_machine": wave_machine,
        "readiness_visible": readiness_visible,
        "readiness_machine": readiness_machine,
        "qc_context": qc_context,
    }


def build_review_preview(
    *,
    output_path: Path,
    preview_dir: Path,
    load_spine: Callable[[str], tuple[pd.DataFrame, Path, dict[str, Any]]],
    load_candidate_intervals: Callable[
        [str], tuple[pd.DataFrame, Path, dict[str, Any]]
    ],
    load_eligibility_base: Callable[
        [str], tuple[pd.DataFrame, Path, dict[str, Any]]
    ],
) -> dict[str, Any]:
    """仅在系统临时目录生成正式发布前的视觉预览。"""
    verify_variable_map_contract()
    verify_locked_design()
    sources = {cohort: verify_source_context(cohort) for cohort in COHORTS}
    prepared = prepare_measurements(
        source_contexts=sources,
        load_spine=load_spine,
        load_candidate_intervals=load_candidate_intervals,
        load_eligibility_base=load_eligibility_base,
    )
    return create_review_workbook(
        prepared["source_visible"],
        prepared["wave_visible"],
        prepared["readiness_visible"],
        output_path,
        preview_dir,
    )


def _build_manifest(
    cohort: str,
    panel: pd.DataFrame,
    *,
    staged_output: Path,
    source_context: dict[str, Any],
    spine_input: dict[str, Any],
    source_qc: dict[str, Any],
    price_context: dict[str, Any],
    wave_qc: list[dict[str, Any]],
    readiness_qc: list[dict[str, Any]],
    qc_context: dict[str, Any],
    provenance: dict[str, Any],
) -> dict[str, Any]:
    return {
        "manifest_schema_version": "wealth_panel_manifest_v1",
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
        "map_version": MAP_VERSION,
        "mapping_sha256": sha256_file(VARIABLE_MAP_PATH),
        "design_register_sha256": sha256_file(DESIGN_REGISTER_PATH),
        "key": list(KEY_COLUMNS),
        "row_count": len(panel),
        "key_unique": not panel.duplicated(list(KEY_COLUMNS)).any(),
        "schema": _schema(panel),
        "source_hash": source_context["raw_sha256"],
        "codebook_hash": source_context["codebook_sha256"],
        "spine_hash": spine_input["sha256"],
        "build_inputs": {
            "raw": {
                "path": str(source_context["raw_path"]),
                "sha256": source_context["raw_sha256"],
                "dataset_version": source_context["dataset_version"],
            },
            "codebook": {
                "path": _relative(source_context["codebook_path"]),
                "sha256": source_context["codebook_sha256"],
                "codebook_version": source_context["codebook_version"],
            },
            "spine": spine_input,
            "design_register": {
                "path": _relative(DESIGN_REGISTER_PATH),
                "sha256": sha256_file(DESIGN_REGISTER_PATH),
            },
            "variable_map": {
                "path": _relative(VARIABLE_MAP_PATH),
                "sha256": sha256_file(VARIABLE_MAP_PATH),
            },
        },
        "source_contract": {
            "total_wealth_source": COHORT_SPECS[cohort].wealth_template,
            "source_flag_source": COHORT_SPECS[cohort].flag_template,
            "source_flag_wave_max": COHORT_SPECS[cohort].flag_wave_max,
            "couple_status_source": COHORT_SPECS[cohort].couple_template,
            "reporting_level": COHORT_SPECS[cohort].reporting_level,
            "nominal_unit": COHORT_SPECS[cohort].nominal_unit,
            "real_base_year": COHORT_SPECS[cohort].base_year,
            "formula": specification()["formula"],
            "price_year_rule": specification()["price_year_rule"],
        },
        "price_index": price_context,
        "source_qc": source_qc,
        "wave_qc": wave_qc,
        "interval_readiness_qc": readiness_qc,
        "qc_context": qc_context,
        "output": {
            "path": _relative(_final_output_path(cohort)),
            "sha256": sha256_file(staged_output),
        },
        "forbidden_transformations": [
            "lag",
            "wealth change",
            "rank",
            "shock",
            "recovery",
            "sample eligibility",
            "interpolation",
            "winsorisation",
        ],
    }


def _build_freeze(
    *,
    staged_outputs: dict[str, Path],
    staged_manifests: dict[str, Path],
    staged_review: Path,
    staged_dictionary: Path,
    protected_before: dict[str, Any],
    prepared: dict[str, Any],
    provenance: dict[str, Any],
) -> dict[str, Any]:
    return {
        "freeze_record_schema_version": "wealth_panel_freeze_record_v1",
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
        "map_version": MAP_VERSION,
        "mapping": {
            "path": _relative(VARIABLE_MAP_PATH),
            "sha256": sha256_file(VARIABLE_MAP_PATH),
        },
        "design_register": {
            "path": _relative(DESIGN_REGISTER_PATH),
            "sha256": sha256_file(DESIGN_REGISTER_PATH),
            "locked_decisions": LOCKED_WEALTH_DESIGN,
        },
        "common_key": list(KEY_COLUMNS),
        "physical_columns": list(WEALTH_PANEL_COLUMNS),
        "formula": specification()["formula"],
        "price_year_rule": specification()["price_year_rule"],
        "hrs_cpi_config": {
            "source_url": HRS_CPI_SOURCE_URL,
            "base_year": 2022,
            "config_sha256": hrs_cpi_config_hash(),
            "supported_years": [min(HRS_CPI_U_ANNUAL), max(HRS_CPI_U_ANNUAL)],
        },
        "cohorts": {
            cohort: {
                "parquet_path": _relative(_final_output_path(cohort)),
                "parquet_sha256": sha256_file(staged_outputs[cohort]),
                "manifest_path": _relative(_manifest_path(cohort)),
                "manifest_sha256": sha256_file(staged_manifests[cohort]),
                "row_count": len(prepared["panels"][cohort]),
                "source_hash": protected_before["sources"][cohort]["raw_sha256"],
                "spine_hash": prepared["spine_inputs"][cohort]["sha256"],
                "real_base_year": COHORT_SPECS[cohort].base_year,
                "reporting_level": COHORT_SPECS[cohort].reporting_level,
            }
            for cohort in COHORTS
        },
        "review_workbook": {
            "path": _relative(REVIEW_PATH),
            "sha256": sha256_file(staged_review),
            "sheets": [
                "1_Source Contract",
                "2_Wave QC",
                "3_Interval Readiness",
            ],
            "formula_error_scan": "PASS",
            "three_sheet_render": "PASS",
        },
        "data_dictionary": {
            "path": _relative(DICTIONARY_PATH),
            "sha256": sha256_file(staged_dictionary),
        },
        "qc_context": {
            "candidate_and_eligibility_are_not_build_inputs": True,
            "cohorts": prepared["qc_context"],
        },
        "protected_inputs_before_and_after_identical": True,
        "protected_snapshot": protected_before,
        "fi_panel_v2_is_not_a_build_input": True,
        "overwrite_policy": "Do not overwrite Wealth Panel v1 or this freeze record; create a new version.",
        "partial_publish_policy": "Build and validate all cohorts in system temporary staging; publish only after all checks pass; write freeze last.",
        "next_stage_status": "EXPOSURE_NOT_STARTED",
    }


def _publish(staged: Path, final: Path) -> None:
    if final.exists():
        raise RuntimeError(f"拒绝覆盖已有文件：{final}")
    final.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(staged), str(final))


def run_wealth_panel_release(
    cohorts: list[str],
    *,
    load_spine: Callable[[str], tuple[pd.DataFrame, Path, dict[str, Any]]],
    load_candidate_intervals: Callable[
        [str], tuple[pd.DataFrame, Path, dict[str, Any]]
    ],
    load_eligibility_base: Callable[
        [str], tuple[pd.DataFrame, Path, dict[str, Any]]
    ],
) -> dict[str, Any]:
    """构建、验证、发布并冻结三队列 Wealth Panel v1。"""
    if tuple(cohorts) != COHORTS:
        raise RuntimeError("wealth v1 必须使用 --cohort all 一次性生成三队列共同 review/freeze")
    _verify_release_absent()
    verify_variable_map_contract()
    verify_locked_design()
    protected_before = protected_snapshot(
        load_spine=load_spine,
        load_candidate_intervals=load_candidate_intervals,
        load_eligibility_base=load_eligibility_base,
    )
    source_contexts = _source_contexts_from_snapshot(protected_before)
    prepared = prepare_measurements(
        source_contexts=source_contexts,
        load_spine=load_spine,
        load_candidate_intervals=load_candidate_intervals,
        load_eligibility_base=load_eligibility_base,
    )
    provenance = git_provenance()
    preview_dir = Path(tempfile.gettempdir()) / "wealth_panel_review_v1_final_previews"
    if preview_dir.exists():
        shutil.rmtree(preview_dir)

    with tempfile.TemporaryDirectory(prefix="wealth_panel_v1_") as tmp_name:
        staging = Path(tmp_name)
        staged_outputs: dict[str, Path] = {}
        for cohort in COHORTS:
            staged = staging / _final_output_path(cohort).name
            prepared["panels"][cohort].to_parquet(staged, index=False)
            reloaded = pd.read_parquet(staged)
            validate_wealth_panel(reloaded, prepared["spines"][cohort], cohort)
            staged_outputs[cohort] = staged

        staged_review = staging / REVIEW_PATH.name
        workbook_checks = create_review_workbook(
            prepared["source_visible"],
            prepared["wave_visible"],
            prepared["readiness_visible"],
            staged_review,
            preview_dir,
        )
        if len(workbook_checks["preview_files"]) != 3:
            raise RuntimeError("review workbook 未完成三 Sheet 渲染")

        staged_dictionary = staging / DICTIONARY_PATH.name
        staged_dictionary.write_text(DICTIONARY_TEXT, encoding="utf-8")
        staged_manifests: dict[str, Path] = {}
        for cohort in COHORTS:
            manifest = _build_manifest(
                cohort,
                prepared["panels"][cohort],
                staged_output=staged_outputs[cohort],
                source_context=source_contexts[cohort],
                spine_input=prepared["spine_inputs"][cohort],
                source_qc=prepared["source_qc"][cohort],
                price_context=prepared["price_contexts"][cohort],
                wave_qc=prepared["wave_machine"][cohort],
                readiness_qc=prepared["readiness_machine"][cohort],
                qc_context=prepared["qc_context"][cohort],
                provenance=provenance,
            )
            staged_manifest = staging / _manifest_path(cohort).name
            _json_write(staged_manifest, manifest)
            staged_manifests[cohort] = staged_manifest

        protected_after = protected_snapshot(
            load_spine=load_spine,
            load_candidate_intervals=load_candidate_intervals,
            load_eligibility_base=load_eligibility_base,
        )
        if protected_before != protected_after:
            raise RuntimeError("Wealth 构建期间受保护 raw/codebook/pre-panel/FI v2 发生变化")

        staged_freeze = staging / FREEZE_PATH.name
        _json_write(
            staged_freeze,
            _build_freeze(
                staged_outputs=staged_outputs,
                staged_manifests=staged_manifests,
                staged_review=staged_review,
                staged_dictionary=staged_dictionary,
                protected_before=protected_before,
                prepared=prepared,
                provenance=provenance,
            ),
        )

        for cohort in COHORTS:
            _publish(staged_outputs[cohort], _final_output_path(cohort))
        _publish(staged_review, REVIEW_PATH)
        _publish(staged_dictionary, DICTIONARY_PATH)
        for cohort in COHORTS:
            _publish(staged_manifests[cohort], _manifest_path(cohort))
        _publish(staged_freeze, FREEZE_PATH)

    return {
        "freeze_status": "FROZEN",
        "freeze_path": str(FREEZE_PATH),
        "preview_dir": str(preview_dir),
        "cohorts": {
            cohort: {
                "rows": len(prepared["panels"][cohort]),
                "valid_wealth_n": prepared["source_qc"][cohort]["valid_wealth_n"],
                "output": str(_final_output_path(cohort)),
            }
            for cohort in COHORTS
        },
        "review_workbook": str(REVIEW_PATH),
    }


__all__ = [
    "BUILDER_VERSION",
    "COHORT_SPECS",
    "HRS_CPI_U_ANNUAL",
    "MAP_VERSION",
    "RELEASE_VERSION",
    "REVIEW_REVISION_MANIFEST_PATH",
    "REVIEW_REVISION_PATH",
    "REVIEW_REVISION_VERSION",
    "SPEC_VERSION",
    "WEALTH_PANEL_COLUMNS",
    "WEALTH_PANEL_DTYPES",
    "build_review_preview",
    "build_wealth_panel",
    "extract_price_index",
    "price_years_from_spine",
    "review_audit_notes",
    "run_wealth_panel_release",
    "specification",
    "specification_hash",
    "validate_wealth_panel",
    "verify_locked_design",
    "verify_variable_map_contract",
]
