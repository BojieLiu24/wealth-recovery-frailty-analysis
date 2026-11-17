"""三队列 panel 的设计中立核心函数。"""

from __future__ import annotations

import hashlib
import csv
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


RUNNER_VERSION = "0.13.0"
ROOT = Path(__file__).resolve().parents[2]
SHARED_SOURCE_REGISTRY_PATH = ROOT / "00_admin" / "shared_source_registry.csv"


def configured_shared_data_root() -> Path:
    """读取研究者显式配置的数据根目录，不使用作者私有存储。"""
    from portable_data_paths import configured_root
    return configured_root()


def resolve_shared_source_path(asset_key: str) -> Path:
    """按登记键定位指定版本来源；不使用候选文件搜索或隐式后备。"""
    from portable_data_paths import resolve_asset
    return resolve_asset(asset_key)


def resolve_cohort_dataset_path(cohort: str) -> Path:
    """解析当前 panel pipeline 对应的正式 cohort source 文件。"""
    asset_keys = {
        "HRS": "hrs_rand_longitudinal",
        "ELSA": "elsa_gateway_h",
        "KLoSA": "klosa_gateway_f",
    }
    if cohort not in asset_keys:
        raise KeyError(f"当前 panel pipeline 未登记 cohort：{cohort}")
    return resolve_shared_source_path(asset_keys[cohort])


def resolve_cohort_codebook_path(cohort: str) -> Path:
    """解析与当前 cohort source 配对的 authoritative codebook。"""
    asset_keys = {
        "HRS": "hrs_rand_longitudinal_codebook",
        "ELSA": "elsa_gateway_h_codebook",
        "KLoSA": "klosa_gateway_f_codebook",
    }
    if cohort not in asset_keys:
        raise KeyError(f"当前 panel pipeline 未登记 cohort codebook：{cohort}")
    return resolve_shared_source_path(asset_keys[cohort])


SPINE_CORE_COLUMNS = (
    "cohort",
    "person_id",
    "wave",
    "wave_order",
    "nominal_year",
    "predecessor_wave",
    "successor_wave",
    "person_wave_scope_status",
    "observed_in_wave",
    "sample_entry_wave",
    "interview_year",
    "interview_month",
    "interview_date",
)


CANDIDATE_SCHEMA_VERSION = "candidate_intervals_v3"


CANDIDATE_INTERVAL_COLUMNS = (
    "cohort",
    "person_id",
    "candidate_interval_order",
    "tminus1_wave",
    "candidate_shock_wave",
    "tplus1_wave",
    "tplus2_wave",
    "tminus1_nominal_year",
    "shock_wave_nominal_year",
    "tplus1_nominal_year",
    "tplus2_nominal_year",
    "observed_tminus1",
    "observed_t",
    "interval_both_observed",
    "observed_tplus1",
    "observed_tplus2",
    "observed_tplus2_or_later",
)


CANDIDATE_INTERVAL_DTYPES = {
    "cohort": "string",
    "person_id": "string",
    "candidate_interval_order": "Int16",
    "tminus1_wave": "Int16",
    "candidate_shock_wave": "Int16",
    "tplus1_wave": "Int16",
    "tplus2_wave": "Int16",
    "tminus1_nominal_year": "Int16",
    "shock_wave_nominal_year": "Int16",
    "tplus1_nominal_year": "Int16",
    "tplus2_nominal_year": "Int16",
    "observed_tminus1": "boolean",
    "observed_t": "boolean",
    "interval_both_observed": "boolean",
    "observed_tplus1": "boolean",
    "observed_tplus2": "boolean",
    "observed_tplus2_or_later": "boolean",
}


CANDIDATE_INTERVAL_NULLABLE = {
    "cohort": False,
    "person_id": False,
    "candidate_interval_order": False,
    "tminus1_wave": False,
    "candidate_shock_wave": False,
    "tplus1_wave": True,
    "tplus2_wave": True,
    "tminus1_nominal_year": False,
    "shock_wave_nominal_year": False,
    "tplus1_nominal_year": True,
    "tplus2_nominal_year": True,
    "observed_tminus1": True,
    "observed_t": True,
    "interval_both_observed": True,
    "observed_tplus1": True,
    "observed_tplus2": True,
    "observed_tplus2_or_later": True,
}


ELIGIBILITY_SCHEMA_VERSION = "eligibility_base_v2"


ELIGIBILITY_BASE_COLUMNS = (
    "cohort",
    "person_id",
    "candidate_shock_wave",
    "scope_assessment_status",
    "pair_observation_status",
    "age_assessment_status",
    "structural_eligibility_status",
)


ELIGIBILITY_BASE_DTYPES = {
    "cohort": "string",
    "person_id": "string",
    "candidate_shock_wave": "Int16",
    "scope_assessment_status": "string",
    "pair_observation_status": "string",
    "age_assessment_status": "string",
    "structural_eligibility_status": "string",
}


ELIGIBILITY_BASE_NULLABLE = {
    column: False for column in ELIGIBILITY_BASE_COLUMNS
}


ELIGIBILITY_STATUS_VALUES = {"pass", "fail", "undetermined"}


FINAL_SCOPE_STATUSES = {
    "in_scope",
    "pre_entry",
    "post_death",
    "post_permanent_exit",
    "administratively_unavailable",
    "scope_unknown",
}


def _single_mapping_template(
    mapping: pd.DataFrame,
    cohort: str,
    domain: str,
    canonical_field: str,
) -> str:
    """从已验证映射中取得唯一来源变量模板。"""
    rows = mapping.loc[
        mapping["cohort"].eq(cohort)
        & mapping["domain"].eq(domain)
        & mapping["canonical_field"].eq(canonical_field)
    ]
    if len(rows) != 1:
        raise ValueError(
            f"{cohort}: {domain}/{canonical_field} mapping 必须恰有一行"
        )
    if rows.iloc[0]["verification_status"] != "verified":
        raise ValueError(f"{cohort}: {domain}/{canonical_field} mapping 未 verified")
    return str(rows.iloc[0]["source_variable_template"])


def _sidecar_columns(source: pd.DataFrame, variable: str) -> tuple[pd.Series, pd.Series]:
    """读取来源变量的值与缺失原因并列列。"""
    value_column = f"{variable}__value"
    tag_column = f"{variable}__missing_tag"
    if value_column not in source or tag_column not in source:
        raise KeyError(f"source slice 缺少 {variable} 的值或 missing-tag 列")
    return source[value_column], source[tag_column].astype("string")


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    """以流式读取计算文件 SHA-256。"""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def select_spine_core(spine: pd.DataFrame) -> pd.DataFrame:
    """从完整审计 Spine 中选出下游分析使用的固定核心列。"""
    missing = [column for column in SPINE_CORE_COLUMNS if column not in spine.columns]
    if missing:
        raise KeyError(f"spine core 缺少字段：{missing}")

    key = ["cohort", "person_id", "wave"]
    if spine.duplicated(key).any():
        raise ValueError("spine core 来源表主键不唯一")

    core = spine.loc[:, list(SPINE_CORE_COLUMNS)].copy()
    if len(core) != len(spine) or not core.index.equals(spine.index):
        raise AssertionError("spine core 筛选改变了行数或行顺序")
    return core


def validate_candidate_intervals(candidate: pd.DataFrame) -> None:
    """验证 Candidate Interval v3 的列顺序、类型、缺失契约和主键。"""
    actual_columns = list(candidate.columns)
    expected_columns = list(CANDIDATE_INTERVAL_COLUMNS)
    if actual_columns != expected_columns:
        raise ValueError(
            "candidate intervals v3 列结构不一致："
            f"actual={actual_columns}, expected={expected_columns}"
        )

    actual_dtypes = {column: str(candidate[column].dtype) for column in actual_columns}
    if actual_dtypes != CANDIDATE_INTERVAL_DTYPES:
        raise TypeError(
            "candidate intervals v3 数据类型不一致："
            f"actual={actual_dtypes}, expected={CANDIDATE_INTERVAL_DTYPES}"
        )

    forbidden_nulls = {
        column: int(candidate[column].isna().sum())
        for column, nullable in CANDIDATE_INTERVAL_NULLABLE.items()
        if not nullable and candidate[column].isna().any()
    }
    if forbidden_nulls:
        raise ValueError(f"candidate intervals v3 非空列出现缺失：{forbidden_nulls}")

    key = ["cohort", "person_id", "candidate_shock_wave"]
    if candidate.duplicated(key).any():
        raise ValueError("candidate intervals v3 主键不唯一")


def validate_eligibility_base(eligibility: pd.DataFrame) -> None:
    """验证 Eligibility Base v2 的列、类型、取值、非空契约和主键。"""
    actual_columns = list(eligibility.columns)
    expected_columns = list(ELIGIBILITY_BASE_COLUMNS)
    if actual_columns != expected_columns:
        raise ValueError(
            "eligibility base v2 列结构不一致："
            f"actual={actual_columns}, expected={expected_columns}"
        )

    actual_dtypes = {
        column: str(eligibility[column].dtype) for column in actual_columns
    }
    if actual_dtypes != ELIGIBILITY_BASE_DTYPES:
        raise TypeError(
            "eligibility base v2 数据类型不一致："
            f"actual={actual_dtypes}, expected={ELIGIBILITY_BASE_DTYPES}"
        )

    forbidden_nulls = {
        column: int(eligibility[column].isna().sum())
        for column, nullable in ELIGIBILITY_BASE_NULLABLE.items()
        if not nullable and eligibility[column].isna().any()
    }
    if forbidden_nulls:
        raise ValueError(f"eligibility base v2 非空列出现缺失：{forbidden_nulls}")

    status_columns = (
        "scope_assessment_status",
        "pair_observation_status",
        "age_assessment_status",
        "structural_eligibility_status",
    )
    invalid_statuses = {
        column: sorted(
            set(eligibility[column].dropna().astype(str)).difference(
                ELIGIBILITY_STATUS_VALUES
            )
        )
        for column in status_columns
    }
    invalid_statuses = {
        column: values for column, values in invalid_statuses.items() if values
    }
    if invalid_statuses:
        raise ValueError(f"eligibility base v2 出现非法状态：{invalid_statuses}")

    expected_structural = _three_component_status(
        eligibility["scope_assessment_status"],
        eligibility["pair_observation_status"],
        eligibility["age_assessment_status"],
    )
    if not expected_structural.equals(eligibility["structural_eligibility_status"]):
        raise ValueError("eligibility base v2 最终状态与三个组件不一致")

    key = ["cohort", "person_id", "candidate_shock_wave"]
    if eligibility.duplicated(key).any():
        raise ValueError("eligibility base v2 主键不唯一")


def normalise_person_id(values: pd.Series, cohort: str) -> pd.Series:
    """把数值型调查 ID 无损转换为字符串。"""
    if values.isna().any():
        raise ValueError(f"{cohort}: person ID 存在缺失")

    if pd.api.types.is_numeric_dtype(values.dtype):
        numeric = pd.to_numeric(values, errors="raise")
        rounded = np.rint(numeric.to_numpy(dtype=float))
        if not np.allclose(numeric.to_numpy(dtype=float), rounded, rtol=0, atol=0):
            raise ValueError(f"{cohort}: person ID 存在非整数值")
        return pd.Series(
            [str(int(value)) for value in rounded],
            index=values.index,
            dtype="string",
        )

    result = values.astype("string").str.strip()
    if result.eq("").any():
        raise ValueError(f"{cohort}: person ID 存在空字符串")
    return result


def observed_from_source(value: pd.Series, missing_tag: pd.Series, variable: str) -> pd.Series:
    """将 inw 原始值转换为可空布尔值，同时保留未知状态。"""
    result = pd.Series(pd.NA, index=value.index, dtype="boolean")
    tag_absent = missing_tag.isna()
    result.loc[tag_absent & value.eq(1)] = True
    result.loc[tag_absent & value.eq(0)] = False

    invalid = tag_absent & value.notna() & ~value.isin([0, 1])
    if invalid.any():
        found = sorted(pd.unique(value.loc[invalid]).tolist())
        raise ValueError(f"{variable}: 发现非 0/1 的 interview indicator：{found}")
    return result


def observed_status(observed: pd.Series, missing_tag: pd.Series) -> pd.Series:
    """生成不丢失 tagged missing 的访谈状态。"""
    result = pd.Series("unknown", index=observed.index, dtype="string")
    result.loc[observed.eq(True).fillna(False)] = "respondent"
    result.loc[observed.eq(False).fillna(False)] = "nonrespondent"
    tagged = missing_tag.notna()
    result.loc[tagged] = "missing_" + missing_tag.loc[tagged].astype("string")
    return result


def _nullable_any(frame: pd.DataFrame) -> pd.Series:
    """逐行执行三值逻辑 any：有 True 为 True，仅有 False 为 False，否则为 NA。"""
    if frame.shape[1] == 0:
        return pd.Series(pd.NA, index=frame.index, dtype="boolean")
    nullable = frame.astype("boolean")
    any_true = nullable.eq(True).fillna(False).any(axis=1)
    any_unknown = nullable.isna().any(axis=1)
    result = pd.Series(False, index=frame.index, dtype="boolean")
    result.loc[any_true] = True
    result.loc[~any_true & any_unknown] = pd.NA
    return result


def _scope_rule_maps(
    scope_rules: pd.DataFrame,
    cohort: str,
) -> tuple[dict[int, str], dict[int, int], set[int], set[str]]:
    """把 codebook 已验证规则转换为访谈状态、入组波次和行政缺失映射。"""
    cohort_rules = scope_rules.loc[scope_rules["cohort"].eq(cohort)].copy()
    if cohort_rules.empty:
        raise ValueError(f"{cohort}: 缺少 scope status rules")
    if cohort_rules["verification_status"].ne("verified").any():
        raise ValueError(f"{cohort}: scope status rules 含未 verified 项")

    status_rows = cohort_rules.loc[cohort_rules["rule_type"].eq("interview_status")]
    entry_rows = cohort_rules.loc[cohort_rules["rule_type"].eq("sample_cohort_entry")]
    sample_status_rows = cohort_rules.loc[
        cohort_rules["rule_type"].eq("sample_cohort_status")
    ]
    tag_rows = cohort_rules.loc[
        cohort_rules["rule_type"].eq("sample_cohort_missing_tag")
    ]
    status_map = {
        int(row.source_code): str(row.canonical_scope_status)
        for row in status_rows.itertuples(index=False)
    }
    entry_map = {
        int(row.source_code): int(row.entry_wave)
        for row in entry_rows.itertuples(index=False)
    }
    administrative_codes = {
        int(row.source_code)
        for row in sample_status_rows.itertuples(index=False)
        if row.canonical_scope_status == "administratively_unavailable"
    }
    administrative_tags = {
        str(row.source_code) for row in tag_rows.itertuples(index=False)
    }
    if not status_map or not entry_map:
        raise ValueError(f"{cohort}: scope rules 缺少状态或入组波次映射")
    return status_map, entry_map, administrative_codes, administrative_tags


def _sample_entry_wave(
    source: pd.DataFrame,
    cohort: str,
    sample_cohort_template: str,
    waves: list[int],
    entry_map: dict[int, int],
    non_entry_codes: set[int],
) -> tuple[pd.Series, pd.Series]:
    """依据逐波官方 cohort 编码取得最早入组波次，并标记后续分类变化。"""
    variables = (
        [sample_cohort_template.format(wave=wave) for wave in waves]
        if "{wave}" in sample_cohort_template
        else [sample_cohort_template]
    )
    candidates: list[pd.Series] = []
    for variable in variables:
        value, tag = _sidecar_columns(source, variable)
        numeric = pd.to_numeric(value, errors="coerce")
        substantive = tag.isna() & numeric.notna()
        unknown_codes = sorted(
            {
                int(code)
                for code in numeric.loc[substantive].unique().tolist()
                if int(code) not in entry_map and int(code) not in non_entry_codes
            }
        )
        if unknown_codes:
            raise ValueError(
                f"{cohort}: {variable} 出现未映射 sample-cohort 编码 {unknown_codes}"
            )
        candidate = numeric.map(entry_map).astype("Int16")
        candidate.loc[~substantive] = pd.NA
        candidates.append(candidate)

    candidate_frame = pd.concat(candidates, axis=1)
    multiple_candidates = candidate_frame.nunique(axis=1, dropna=True).gt(1)
    # ELSA 的 rWcohort_e 是逐波分类；少数原样本年轻伴侣在 W3 被重分类。
    # 只取权威编码给出的最早入组波次，不用首次实际访谈推断。
    entry_wave = candidate_frame.min(axis=1, skipna=True).astype("Int16")
    return entry_wave, multiple_candidates.astype("boolean")


def _scope_status_for_wave(
    *,
    cohort: str,
    wave: int,
    interview_value: pd.Series,
    interview_tag: pd.Series,
    sample_value: pd.Series,
    sample_tag: pd.Series,
    sample_entry_wave: pd.Series,
    status_map: dict[int, str],
    administrative_codes: set[int],
    administrative_tags: set[str],
    interview_variable: str,
) -> pd.Series:
    """在 scope 与 observed 分离的前提下生成个人逐波调查范围状态。"""
    numeric = pd.to_numeric(interview_value, errors="coerce")
    substantive = interview_tag.isna() & numeric.notna()
    unknown_codes = sorted(
        {
            int(code)
            for code in numeric.loc[substantive].unique().tolist()
            if int(code) not in status_map
        }
    )
    if unknown_codes:
        raise ValueError(
            f"{cohort}: {interview_variable} 出现未映射 interview-status 编码 "
            f"{unknown_codes}"
        )

    result = pd.Series("scope_unknown", index=interview_value.index, dtype="string")
    mapped = numeric.map(status_map).astype("string")
    result.loc[substantive] = mapped.loc[substantive]

    conditional = result.eq("pre_entry_if_before_entry")
    before_entry = sample_entry_wave.notna() & sample_entry_wave.gt(wave)
    result.loc[conditional & before_entry] = "pre_entry"
    result.loc[conditional & ~before_entry] = "scope_unknown"

    # 官方 sample-cohort 入组波次可确认尚未进入，但不能覆盖相反的明确状态。
    incompatible = before_entry & ~result.isin(["pre_entry", "scope_unknown"])
    if incompatible.any():
        raise ValueError(
            f"{cohort}: wave {wave} 有 {int(incompatible.sum())} 条记录在官方入组前"
            "却带有相反的明确 interview status"
        )
    result.loc[before_entry & result.eq("scope_unknown")] = "pre_entry"

    sample_numeric = pd.to_numeric(sample_value, errors="coerce")
    administrative = sample_tag.isin(administrative_tags) | sample_numeric.isin(
        administrative_codes
    )
    result.loc[administrative] = "administratively_unavailable"

    invalid = ~result.isin(FINAL_SCOPE_STATUSES)
    if invalid.any():
        found = sorted(result.loc[invalid].dropna().unique().tolist())
        raise ValueError(f"{cohort}: 产生非法 scope status：{found}")
    return result


def build_spine(
    source: pd.DataFrame,
    registry: dict[str, str],
    wave_map: pd.DataFrame,
    structural_mapping: pd.DataFrame,
    scope_rules: pd.DataFrame,
) -> pd.DataFrame:
    """生成完整 cohort-global person-wave spine，并独立记录 scope 与 observed。"""
    cohort = registry["cohort"]
    id_source = registry["person_id_source"]
    id_value_col = f"{id_source}__value"
    id_tag_col = f"{id_source}__missing_tag"
    if id_value_col not in source or id_tag_col not in source:
        raise KeyError(f"{cohort}: source slice 缺少 person ID sidecar")
    if source[id_tag_col].notna().any():
        raise ValueError(f"{cohort}: person ID 含 tagged/system missing")

    person_id = normalise_person_id(source[id_value_col], cohort)
    if person_id.duplicated().any():
        raise ValueError(f"{cohort}: 原始宽表 person ID 不唯一")

    pieces: list[pd.DataFrame] = []
    ordered_map = wave_map.sort_values("wave_order")
    waves = ordered_map["wave"].astype(int).tolist()
    interview_status_template = _single_mapping_template(
        structural_mapping,
        cohort,
        "scope",
        "interview_status",
    )
    sample_cohort_template = _single_mapping_template(
        structural_mapping,
        cohort,
        "scope",
        "sample_cohort",
    )
    status_map, entry_map, administrative_codes, administrative_tags = _scope_rule_maps(
        scope_rules,
        cohort,
    )
    sample_entry_wave, sample_entry_wave_multiple_candidates = _sample_entry_wave(
        source,
        cohort,
        sample_cohort_template,
        waves,
        entry_map,
        administrative_codes,
    )

    for row in ordered_map.itertuples(index=False):
        wave = int(row.wave)
        inw_source = registry["interview_indicator_template"].format(wave=wave)
        inw_value, inw_tag = _sidecar_columns(source, inw_source)
        observed = observed_from_source(inw_value, inw_tag, inw_source)

        interview_status_source = interview_status_template.format(wave=wave)
        interview_status_value, interview_status_tag = _sidecar_columns(
            source,
            interview_status_source,
        )
        sample_cohort_source = sample_cohort_template.format(wave=wave)
        sample_cohort_value, sample_cohort_tag = _sidecar_columns(
            source,
            sample_cohort_source,
        )
        scope_status = _scope_status_for_wave(
            cohort=cohort,
            wave=wave,
            interview_value=interview_status_value,
            interview_tag=interview_status_tag,
            sample_value=sample_cohort_value,
            sample_tag=sample_cohort_tag,
            sample_entry_wave=sample_entry_wave,
            status_map=status_map,
            administrative_codes=administrative_codes,
            administrative_tags=administrative_tags,
            interview_variable=interview_status_source,
        )

        frame = pd.DataFrame(
            {
                "cohort": pd.Series(cohort, index=source.index, dtype="string"),
                "person_id": person_id,
                "wave": wave,
                "wave_order": int(row.wave_order),
                "nominal_year": int(row.nominal_year),
                "predecessor_wave": (
                    pd.NA if pd.isna(row.predecessor_wave) else int(row.predecessor_wave)
                ),
                "successor_wave": (
                    pd.NA if pd.isna(row.successor_wave) else int(row.successor_wave)
                ),
                "observed_in_wave": observed,
                "observation_status": observed_status(observed, inw_tag),
                "person_wave_scope_status": scope_status,
                "scope_source_variable": interview_status_source,
                "scope_raw_value": pd.to_numeric(interview_status_value, errors="coerce"),
                "scope_missing_tag": interview_status_tag,
                "sample_cohort_source_variable": sample_cohort_source,
                "sample_cohort_raw_value": pd.to_numeric(
                    sample_cohort_value,
                    errors="coerce",
                ),
                "sample_cohort_missing_tag": sample_cohort_tag,
                "sample_entry_wave": sample_entry_wave,
                "sample_entry_wave_multiple_candidates": (
                    sample_entry_wave_multiple_candidates
                ),
                "interview_indicator_source_variable": inw_source,
                "interview_indicator_raw_value": pd.to_numeric(inw_value, errors="coerce"),
                "interview_indicator_missing_tag": inw_tag,
                "interview_year": pd.Series(pd.NA, index=source.index, dtype="Int64"),
                "interview_month": pd.Series(pd.NA, index=source.index, dtype="Int64"),
                "interview_date": pd.Series(pd.NaT, index=source.index, dtype="datetime64[ns]"),
                "interview_year_source_variable": pd.Series(pd.NA, index=source.index, dtype="string"),
                "interview_month_source_variable": pd.Series(pd.NA, index=source.index, dtype="string"),
                "interview_date_source_variable": pd.Series(pd.NA, index=source.index, dtype="string"),
                "interview_year_missing_tag": pd.Series(pd.NA, index=source.index, dtype="string"),
                "interview_month_missing_tag": pd.Series(pd.NA, index=source.index, dtype="string"),
                "interview_date_missing_tag": pd.Series(pd.NA, index=source.index, dtype="string"),
                "source_dataset_version": registry["dataset_version_expected"],
                "spine_schema_version": "spine_v2",
            }
        )

        date_template = registry.get("interview_date_template", "")
        if date_template:
            date_source = date_template.format(wave=wave)
            raw_date = pd.to_numeric(source[f"{date_source}__value"], errors="coerce")
            date_tag = source[f"{date_source}__missing_tag"].astype("string")
            converted = pd.to_datetime(
                raw_date,
                unit="D",
                origin="1960-01-01",
                errors="coerce",
            )
            frame["interview_date"] = converted
            frame["interview_year"] = converted.dt.year.astype("Int64")
            frame["interview_month"] = converted.dt.month.astype("Int64")
            frame["interview_date_source_variable"] = date_source
            frame["interview_year_source_variable"] = date_source
            frame["interview_month_source_variable"] = date_source
            frame["interview_date_missing_tag"] = date_tag
            frame["interview_year_missing_tag"] = date_tag
            frame["interview_month_missing_tag"] = date_tag
        else:
            year_source = registry["interview_year_template"].format(wave=wave)
            month_source = registry["interview_month_template"].format(wave=wave)
            year_value = pd.to_numeric(source[f"{year_source}__value"], errors="coerce")
            month_value = pd.to_numeric(source[f"{month_source}__value"], errors="coerce")
            frame["interview_year"] = year_value.round().astype("Int64")
            frame["interview_month"] = month_value.round().astype("Int64")
            frame["interview_year_source_variable"] = year_source
            frame["interview_month_source_variable"] = month_source
            frame["interview_year_missing_tag"] = source[
                f"{year_source}__missing_tag"
            ].astype("string")
            frame["interview_month_missing_tag"] = source[
                f"{month_source}__missing_tag"
            ].astype("string")

        pieces.append(frame)

    spine = pd.concat(pieces, ignore_index=True)
    spine["wave"] = spine["wave"].astype("int16")
    spine["wave_order"] = spine["wave_order"].astype("int16")
    spine["nominal_year"] = spine["nominal_year"].astype("int16")
    spine["predecessor_wave"] = spine["predecessor_wave"].astype("Int16")
    spine["successor_wave"] = spine["successor_wave"].astype("Int16")
    spine["sample_entry_wave"] = spine["sample_entry_wave"].astype("Int16")
    text_columns = [
        "cohort",
        "person_id",
        "observation_status",
        "person_wave_scope_status",
        "scope_source_variable",
        "scope_missing_tag",
        "sample_cohort_source_variable",
        "sample_cohort_missing_tag",
        "interview_indicator_source_variable",
        "interview_indicator_missing_tag",
        "interview_year_source_variable",
        "interview_month_source_variable",
        "interview_date_source_variable",
        "interview_year_missing_tag",
        "interview_month_missing_tag",
        "interview_date_missing_tag",
        "source_dataset_version",
        "spine_schema_version",
    ]
    for column in text_columns:
        spine[column] = spine[column].astype("string")
    spine = spine.sort_values(["person_id", "wave_order"], kind="stable").reset_index(drop=True)

    if spine.duplicated(["cohort", "person_id", "wave"]).any():
        raise ValueError(f"{cohort}: spine 主键不唯一")
    expected_rows = source.shape[0] * ordered_map.shape[0]
    if len(spine) != expected_rows:
        raise AssertionError(f"{cohort}: spine 行数守恒失败")
    return spine


def build_candidate_intervals(spine: pd.DataFrame) -> pd.DataFrame:
    """枚举全部 cohort-global 相邻 t-1 -> t 区间，不预筛个人 eligibility。"""
    required = {"cohort", "person_id", "wave", "wave_order", "nominal_year", "observed_in_wave"}
    missing = required.difference(spine.columns)
    if missing:
        raise KeyError(f"spine 缺少字段：{sorted(missing)}")
    cohorts = spine["cohort"].dropna().unique().tolist()
    if len(cohorts) != 1:
        raise ValueError("candidate intervals 每次只能处理一个 cohort")
    cohort = str(cohorts[0])
    if spine.duplicated(["cohort", "person_id", "wave"]).any():
        raise ValueError(f"{cohort}: spine 主键不唯一")

    wave_rows = (
        spine[["wave", "wave_order", "nominal_year"]]
        .drop_duplicates()
        .sort_values("wave_order")
        .reset_index(drop=True)
    )
    expected_orders = list(range(1, len(wave_rows) + 1))
    if wave_rows["wave_order"].astype(int).tolist() != expected_orders:
        raise ValueError(f"{cohort}: wave_order 不是连续 cohort-global 顺序")

    observed = spine.pivot(index="person_id", columns="wave", values="observed_in_wave")
    observed = observed.astype("boolean").sort_index()
    person_ids = observed.index.astype("string")
    nominal_by_wave = dict(zip(wave_rows["wave"].astype(int), wave_rows["nominal_year"].astype(int)))
    waves = wave_rows["wave"].astype(int).tolist()

    pieces: list[pd.DataFrame] = []
    for position in range(1, len(waves)):
        tminus1 = waves[position - 1]
        t = waves[position]
        tplus1 = waves[position + 1] if position + 1 < len(waves) else None
        tplus2 = waves[position + 2] if position + 2 < len(waves) else None
        tplus2_or_later_waves = waves[position + 2 :]

        observed_pre = observed[tminus1].reset_index(drop=True).astype("boolean")
        observed_t = observed[t].reset_index(drop=True).astype("boolean")
        if tplus1 is None:
            observed_tplus1 = pd.Series(pd.NA, index=range(len(observed)), dtype="boolean")
        else:
            observed_tplus1 = observed[tplus1].reset_index(drop=True).astype("boolean")

        if tplus2 is None:
            observed_tplus2 = pd.Series(pd.NA, index=range(len(observed)), dtype="boolean")
        else:
            observed_tplus2 = observed[tplus2].reset_index(drop=True).astype("boolean")

        if tplus2_or_later_waves:
            observed_later = _nullable_any(observed[tplus2_or_later_waves]).reset_index(
                drop=True
            )
        else:
            observed_later = pd.Series(pd.NA, index=range(len(observed)), dtype="boolean")

        frame = pd.DataFrame(
            {
                "cohort": pd.Series(cohort, index=range(len(observed)), dtype="string"),
                "person_id": pd.Series(person_ids, dtype="string").reset_index(drop=True),
                "candidate_interval_order": position,
                "tminus1_wave": tminus1,
                "candidate_shock_wave": t,
                "tplus1_wave": pd.NA if tplus1 is None else tplus1,
                "tplus2_wave": pd.NA if tplus2 is None else tplus2,
                "tminus1_nominal_year": nominal_by_wave[tminus1],
                "shock_wave_nominal_year": nominal_by_wave[t],
                "tplus1_nominal_year": pd.NA if tplus1 is None else nominal_by_wave[tplus1],
                "tplus2_nominal_year": pd.NA if tplus2 is None else nominal_by_wave[tplus2],
                "observed_tminus1": observed_pre,
                "observed_t": observed_t,
                "interval_both_observed": observed_pre & observed_t,
                "observed_tplus1": observed_tplus1,
                "observed_tplus2": observed_tplus2,
                "observed_tplus2_or_later": observed_later,
            }
        )
        pieces.append(frame)

    result = pd.concat(pieces, ignore_index=True)
    int16_columns = [
        "candidate_interval_order",
        "tminus1_wave",
        "candidate_shock_wave",
        "tplus1_wave",
        "tplus2_wave",
        "tminus1_nominal_year",
        "shock_wave_nominal_year",
        "tplus1_nominal_year",
        "tplus2_nominal_year",
    ]
    for column in int16_columns:
        result[column] = result[column].astype("Int16")
    for column in [
        "observed_tminus1",
        "observed_t",
        "interval_both_observed",
        "observed_tplus1",
        "observed_tplus2",
        "observed_tplus2_or_later",
    ]:
        result[column] = result[column].astype("boolean")

    result = result.loc[:, list(CANDIDATE_INTERVAL_COLUMNS)]
    result = result.sort_values(
        ["person_id", "candidate_interval_order"], kind="stable"
    ).reset_index(drop=True)
    validate_candidate_intervals(result)
    expected = observed.shape[0] * (len(waves) - 1)
    if len(result) != expected:
        raise AssertionError(f"{cohort}: candidate interval 行数守恒失败")
    return result


def build_age_long(
    source: pd.DataFrame,
    registry: dict[str, str],
    wave_map: pd.DataFrame,
    eligibility_mapping: pd.DataFrame,
) -> pd.DataFrame:
    """把最小年龄 source slice 转为 person-wave 年龄来源表。"""
    cohort = registry["cohort"]
    id_template = _single_mapping_template(
        eligibility_mapping,
        cohort,
        "eligibility_base",
        "person_id",
    )
    age_template = _single_mapping_template(
        eligibility_mapping,
        cohort,
        "eligibility_base",
        "age_in_years",
    )
    id_value, id_tag = _sidecar_columns(source, id_template)
    if id_tag.notna().any():
        raise ValueError(f"{cohort}: eligibility source 的 person ID 含缺失")
    person_id = normalise_person_id(id_value, cohort)
    if person_id.duplicated().any():
        raise ValueError(f"{cohort}: eligibility source 的 person ID 不唯一")

    pieces: list[pd.DataFrame] = []
    for row in wave_map.sort_values("wave_order").itertuples(index=False):
        wave = int(row.wave)
        variable = age_template.format(wave=wave)
        value, tag = _sidecar_columns(source, variable)
        pieces.append(
            pd.DataFrame(
                {
                    "cohort": pd.Series(cohort, index=source.index, dtype="string"),
                    "person_id": person_id,
                    "wave": wave,
                    "age_value": pd.to_numeric(value, errors="coerce").astype("Float64"),
                    "age_missing_tag": tag,
                    "age_source_variable": pd.Series(
                        variable,
                        index=source.index,
                        dtype="string",
                    ),
                }
            )
        )
    result = pd.concat(pieces, ignore_index=True)
    result["wave"] = result["wave"].astype("Int16")
    if result.duplicated(["cohort", "person_id", "wave"]).any():
        raise ValueError(f"{cohort}: 年龄来源表主键不唯一")
    return result


def _three_component_status(
    first: pd.Series,
    second: pd.Series,
    third: pd.Series,
) -> pd.Series:
    """按 fail 优先、全 pass 才 pass 的规则组合三个 assessment。"""
    result = pd.Series("undetermined", index=first.index, dtype="string")
    any_fail = first.eq("fail") | second.eq("fail") | third.eq("fail")
    all_pass = first.eq("pass") & second.eq("pass") & third.eq("pass")
    result.loc[any_fail] = "fail"
    result.loc[all_pass] = "pass"
    return result


def derive_structural_ineligibility_reason_primary(
    eligibility: pd.DataFrame,
) -> pd.Series:
    """从三个组件状态派生仅供人数流程使用的主要原因。"""
    required = {
        "scope_assessment_status",
        "pair_observation_status",
        "age_assessment_status",
        "structural_eligibility_status",
    }
    missing = sorted(required.difference(eligibility.columns))
    if missing:
        raise KeyError(f"eligibility base 缺少原因派生字段：{missing}")

    reason = pd.Series(pd.NA, index=eligibility.index, dtype="string")
    reason.loc[eligibility["structural_eligibility_status"].eq("undetermined")] = (
        "assessment_undetermined"
    )
    reason.loc[eligibility["age_assessment_status"].eq("fail")] = (
        "age_below_threshold"
    )
    reason.loc[eligibility["pair_observation_status"].eq("fail")] = (
        "interval_not_double_observed"
    )
    reason.loc[eligibility["scope_assessment_status"].eq("fail")] = (
        "out_of_person_scope"
    )
    return reason


def build_eligibility_base(
    intervals: pd.DataFrame,
    spine: pd.DataFrame,
    age_long: pd.DataFrame,
    *,
    age_threshold: float = 60.0,
) -> pd.DataFrame:
    """生成不删行的 pre-substantive structural eligibility ledger。"""
    key = ["cohort", "person_id", "candidate_shock_wave"]
    if intervals.duplicated(key).any():
        raise ValueError("candidate intervals 主键不唯一")
    if spine.duplicated(["cohort", "person_id", "wave"]).any():
        raise ValueError("spine 主键不唯一")
    if age_long.duplicated(["cohort", "person_id", "wave"]).any():
        raise ValueError("年龄来源表主键不唯一")

    result = intervals[
        [
            "cohort",
            "person_id",
            "candidate_shock_wave",
            "tminus1_wave",
            "observed_tminus1",
            "observed_t",
        ]
    ].copy()

    scope_left = spine[
        ["cohort", "person_id", "wave", "person_wave_scope_status"]
    ].rename(
        columns={
            "wave": "tminus1_wave",
            "person_wave_scope_status": "scope_status_tminus1",
        }
    )
    scope_right = spine[
        ["cohort", "person_id", "wave", "person_wave_scope_status"]
    ].rename(
        columns={
            "wave": "candidate_shock_wave",
            "person_wave_scope_status": "scope_status_t",
        }
    )
    result = result.merge(
        scope_left,
        on=["cohort", "person_id", "tminus1_wave"],
        how="left",
        validate="many_to_one",
    ).merge(
        scope_right,
        on=["cohort", "person_id", "candidate_shock_wave"],
        how="left",
        validate="many_to_one",
    )

    age_pre = age_long.rename(
        columns={
            "wave": "tminus1_wave",
            "age_value": "age_tminus1_value",
            "age_missing_tag": "age_tminus1_missing_tag",
            "age_source_variable": "age_tminus1_source_variable",
        }
    )
    result = result.merge(
        age_pre,
        on=["cohort", "person_id", "tminus1_wave"],
        how="left",
        validate="many_to_one",
    )
    if len(result) != len(intervals):
        raise AssertionError("eligibility base 合并后行数不守恒")
    if result[["scope_status_tminus1", "scope_status_t"]].isna().any().any():
        raise ValueError("eligibility base 存在无法连接到 spine 的 scope 状态")

    out_of_scope = {
        "pre_entry",
        "post_death",
        "post_permanent_exit",
        "administratively_unavailable",
    }
    scope_fail = result["scope_status_tminus1"].isin(out_of_scope) | result[
        "scope_status_t"
    ].isin(out_of_scope)
    scope_pass = result["scope_status_tminus1"].eq("in_scope") & result[
        "scope_status_t"
    ].eq("in_scope")
    result["scope_assessment_status"] = pd.Series(
        "undetermined",
        index=result.index,
        dtype="string",
    )
    result.loc[scope_fail, "scope_assessment_status"] = "fail"
    result.loc[scope_pass, "scope_assessment_status"] = "pass"

    observed_pre = result["observed_tminus1"].astype("boolean")
    observed_t = result["observed_t"].astype("boolean")
    pair_pass = observed_pre.eq(True).fillna(False) & observed_t.eq(True).fillna(False)
    pair_fail = observed_pre.eq(False).fillna(False) | observed_t.eq(False).fillna(False)
    result["pair_observation_status"] = pd.Series(
        "undetermined",
        index=result.index,
        dtype="string",
    )
    result.loc[pair_fail, "pair_observation_status"] = "fail"
    result.loc[pair_pass, "pair_observation_status"] = "pass"

    age_value = pd.to_numeric(result["age_tminus1_value"], errors="coerce")
    age_tag = result["age_tminus1_missing_tag"].astype("string")
    age_valid = age_tag.isna() & age_value.notna() & age_value.between(0, 120)
    result["age_tminus1_valid"] = age_valid.astype("boolean")
    result["age_tminus1_ge60"] = pd.Series(pd.NA, index=result.index, dtype="boolean")
    result.loc[age_valid, "age_tminus1_ge60"] = age_value.loc[age_valid].ge(
        age_threshold
    )
    result["age_assessment_status"] = pd.Series(
        "undetermined",
        index=result.index,
        dtype="string",
    )
    result.loc[age_valid & age_value.ge(age_threshold), "age_assessment_status"] = "pass"
    result.loc[age_valid & age_value.lt(age_threshold), "age_assessment_status"] = "fail"

    result["structural_eligibility_status"] = _three_component_status(
        result["scope_assessment_status"],
        result["pair_observation_status"],
        result["age_assessment_status"],
    )

    result["cohort"] = result["cohort"].astype("string")
    result["person_id"] = result["person_id"].astype("string")
    result["candidate_shock_wave"] = result["candidate_shock_wave"].astype("Int16")

    result = result.loc[:, list(ELIGIBILITY_BASE_COLUMNS)].sort_values(
        ["person_id", "candidate_shock_wave"],
        kind="stable",
    ).reset_index(drop=True)
    validate_eligibility_base(result)
    return result


def freeze_first_shock(intervals: pd.DataFrame) -> pd.DataFrame:
    """未来 exposure 阶段使用的纯函数：只按 shock 状态冻结首次冲击。"""
    required = {"cohort", "person_id", "candidate_shock_wave", "shock_status"}
    missing = required.difference(intervals.columns)
    if missing:
        raise KeyError(f"shock interval 缺少字段：{sorted(missing)}")
    true_rows = intervals.loc[intervals["shock_status"].eq("true")].copy()
    true_rows = true_rows.sort_values(["cohort", "person_id", "candidate_shock_wave"])
    first = true_rows.drop_duplicates(["cohort", "person_id"], keep="first")
    return first[["cohort", "person_id", "candidate_shock_wave"]].rename(
        columns={"candidate_shock_wave": "index_shock_wave"}
    )


def dataframe_manifest(data: pd.DataFrame, key: list[str]) -> dict[str, Any]:
    """生成不含个体记录的 schema 和键检查。"""
    duplicates = int(data.duplicated(key).sum())
    return {
        "row_count": int(len(data)),
        "column_count": int(data.shape[1]),
        "key": key,
        "duplicate_key_rows": duplicates,
        "key_unique": duplicates == 0,
        "schema": {column: str(dtype) for column, dtype in data.dtypes.items()},
    }
