"""三队列 Outcome Components v2 构建、审计与冻结。"""

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
from typing import Any, Callable, Iterable

import numpy as np
import pandas as pd
from pandas.io.stata import StataMissingValue

from panel_core import normalise_person_id, resolve_cohort_dataset_path, sha256_file


ROOT = Path(__file__).resolve().parents[2]
ADMIN = ROOT / "00_admin"
REGISTRY_PATH = ADMIN / "cohort_registry.csv"
CODEBOOK_MANIFEST_PATH = ADMIN / "codebooks" / "codebook_manifest.csv"
MANIFEST_ROOT = ADMIN / "panel_manifests"
OUTCOME_MANIFEST_ROOT = MANIFEST_ROOT / "outcome_components"
DERIVED_ROOT = ROOT / "01_data" / "derived" / "panel_components"
VARIABLE_MAP_PATH = ADMIN / "variable_maps" / "fi_component_map.csv"
RELEASE_VERSION = "v2"
REVIEW_PATH = ROOT / "04_collaboration" / f"outcome_components_review_{RELEASE_VERSION}.xlsx"
DICTIONARY_PATH = (
    OUTCOME_MANIFEST_ROOT / f"outcome_components_data_dictionary_{RELEASE_VERSION}.md"
)
FREEZE_PATH = OUTCOME_MANIFEST_ROOT / f"outcome_components_{RELEASE_VERSION}_freeze.json"

COHORTS = ("HRS", "ELSA", "KLoSA")
BUILDER_VERSION = "outcome_components_builder_v2.0.0"
MAP_SCHEMA_VERSION = "outcome_components_map_v2"
KEY_COLUMNS = ("cohort", "person_id", "wave")

TIAN_URL = "https://pmc.ncbi.nlm.nih.gov/articles/PMC13266220/"


COMMON_22 = (
    "hypertension",
    "diabetes",
    "cancer",
    "lung_disease",
    "heart_disease",
    "stroke",
    "arthritis",
    "vision",
    "hearing",
    "self_rated_health",
    "adl_dressing",
    "adl_bathing",
    "adl_eating",
    "adl_bed_transfer",
    "adl_toilet",
    "iadl_telephone",
    "iadl_finances",
    "iadl_medications",
    "iadl_shopping",
    "iadl_meal_preparation",
    "mental_illness",
    "depression",
)

HRS_ELSA_EXTENSIONS = (
    "mobility_walk_short",
    "mobility_climb_one_flight",
    "mobility_climb_several_flights",
    "mobility_rise_from_chair",
    "mobility_stoop_kneel_crouch",
    "mobility_lift_heavy",
    "mobility_pick_up_small_object",
    "mobility_raise_arms",
)

KLOSA_EXTENSIONS = (
    "underweight",
    "adl_brushing_teeth",
    "adl_continence",
    "iadl_housework",
    "iadl_grooming",
    "iadl_laundry",
    "iadl_going_out_short_distance",
    "iadl_transportation",
)

PRIMARY_ITEMS = {
    "HRS": COMMON_22 + HRS_ELSA_EXTENSIONS,
    "ELSA": COMMON_22 + HRS_ELSA_EXTENSIONS,
    "KLoSA": COMMON_22 + KLOSA_EXTENSIONS,
}


@dataclass(frozen=True)
class ComponentSpec:
    """一个队列内一个候选组件的完整可执行映射。"""

    item_id: str
    domain: str
    rule: str
    sources: tuple[tuple[int, str], ...]
    source_summary: str
    codebook_ref: str
    measurement_note: str = ""
    attention_waves: tuple[int, ...] = ()

    def source_for_wave(self, wave: int) -> str | None:
        """返回指定波次的精确来源变量。"""
        return dict(self.sources).get(wave)


DOMAIN_BY_ITEM = {
    **{
        item: "Chronic disease"
        for item in (
            "hypertension",
            "diabetes",
            "cancer",
            "lung_disease",
            "heart_disease",
            "stroke",
            "arthritis",
        )
    },
    "underweight": "Nutrition and body composition",
    "vision": "Sensory and general health",
    "hearing": "Sensory and general health",
    "self_rated_health": "Sensory and general health",
    **{item: "ADL" for item in COMMON_22 if item.startswith("adl_")},
    "adl_brushing_teeth": "ADL",
    "adl_continence": "ADL",
    **{item: "IADL" for item in COMMON_22 if item.startswith("iadl_")},
    **{item: "IADL" for item in KLOSA_EXTENSIONS if item.startswith("iadl_")},
    "mental_illness": "Mental health",
    "depression": "Mental health",
    **{item: "Mobility and physical function" for item in HRS_ELSA_EXTENSIONS},
}


HRS_BASES = {
    "hypertension": "hibpe",
    "diabetes": "diabe",
    "cancer": "cancre",
    "lung_disease": "lunge",
    "heart_disease": "hearte",
    "stroke": "stroke",
    "arthritis": "arthre",
    "vision": "eyert",
    "hearing": "earrt",
    "self_rated_health": "shlt",
    "adl_dressing": "dressa",
    "adl_bathing": "batha",
    "adl_eating": "eata",
    "adl_bed_transfer": "beda",
    "adl_toilet": "toilta",
    "iadl_telephone": "phonea",
    "iadl_finances": "moneya",
    "iadl_medications": "medsa",
    "iadl_shopping": "shopa",
    "iadl_meal_preparation": "mealsa",
    "mental_illness": "psyche",
    "depression": "cesd",
    "mobility_walk_short": "walk1a",
    "mobility_climb_one_flight": "clim1a",
    "mobility_climb_several_flights": "climsa",
    "mobility_rise_from_chair": "chaira",
    "mobility_stoop_kneel_crouch": "stoopa",
    "mobility_lift_heavy": "lifta",
    "mobility_pick_up_small_object": "dimea",
    "mobility_raise_arms": "armsa",
}

ELSA_BASES = {
    **{
        item: base
        for item, base in HRS_BASES.items()
        if item not in {"vision", "hearing", "mobility_walk_short"}
    },
    "vision": "sight",
    "hearing": "hearing",
    "mobility_walk_short": "walk100a",
}

KLOSA_BASES = {
    "hypertension": "hibpe",
    "diabetes": "diabe",
    "cancer": "cancre",
    "lung_disease": "lunge",
    "heart_disease": "hearte",
    "stroke": "stroke",
    "arthritis": "arthre",
    "vision": "sighta",
    "hearing": "hearinga",
    "self_rated_health": "shlt",
    "adl_dressing": "dressb",
    "adl_bathing": "bathb",
    "adl_eating": "eatb",
    "adl_bed_transfer": "bedb_k",
    "adl_toilet": "toiltb",
    "iadl_telephone": "phoneb",
    "iadl_finances": "moneyb",
    "iadl_medications": "medsb",
    "iadl_shopping": "shopb",
    "iadl_meal_preparation": "mealsb",
    "mental_illness": "psyche",
    "underweight": "bmicat",
    "adl_brushing_teeth": "brushb",
    "adl_continence": "urinb",
    "iadl_housework": "housewkb",
    "iadl_grooming": "groomb",
    "iadl_laundry": "laundryb",
    "iadl_going_out_short_distance": "gooutb",
    "iadl_transportation": "transb",
}


def _wave_sources(base: str, first: int, last: int) -> tuple[tuple[int, str], ...]:
    """展开常规 r{wave}{base} 来源变量。"""
    return tuple((wave, f"r{wave}{base}") for wave in range(first, last + 1))


def _summary(base: str, first: int, last: int) -> str:
    """生成人类可读但不复制逐波列表的来源摘要。"""
    if first == last:
        return f"r{first}{base}"
    return f"r{first}{base}–r{last}{base}"


def _rule_for(cohort: str, item_id: str) -> str:
    """返回组件的严格计分规则标识。"""
    if cohort == "KLoSA" and item_id == "underweight":
        return "underweight_bmicat"
    if item_id == "depression":
        return "continuous_30" if cohort == "KLoSA" else "continuous_8"
    if item_id == "vision":
        if cohort == "HRS":
            return "ordinal_poor_to_excellent"
        return "ordinal_better_to_worse_blind"
    if item_id in {"hearing", "self_rated_health"}:
        return (
            "ordinal_poor_to_excellent"
            if cohort == "HRS" and item_id == "hearing"
            else "ordinal_better_to_worse"
        )
    return "binary_01"


CODEBOOK_REFS = {
    "HRS": {
        "Chronic disease": "RAND HRS 2022 V1 codebook, pp. 401–448",
        "Sensory and general health": "RAND HRS 2022 V1 codebook, pp. 247, 557–561",
        "ADL": "RAND HRS 2022 V1 codebook, pp. 916–919",
        "IADL": "RAND HRS 2022 V1 codebook, pp. 981–985",
        "Mental health": "RAND HRS 2022 V1 codebook, pp. 335–340, 401–448",
        "Mobility and physical function": "RAND HRS 2022 V1 codebook, pp. 1045–1055",
    },
    "ELSA": {
        "Chronic disease": "Gateway Harmonized ELSA H codebook, pp. 240–334",
        "Sensory and general health": "Gateway Harmonized ELSA H codebook, pp. 141–143, 349, 356",
        "ADL": "Gateway Harmonized ELSA H codebook, pp. 150–161",
        "IADL": "Gateway Harmonized ELSA H codebook, pp. 150–161",
        "Mental health": "Gateway Harmonized ELSA H codebook, pp. 240–334, 1368",
        "Mobility and physical function": "Gateway Harmonized ELSA H codebook, pp. 172–181",
    },
    "KLoSA": {
        "Chronic disease": "Gateway Harmonized KLoSA F codebook, pp. 127–135",
        "Sensory and general health": "Gateway Harmonized KLoSA F codebook, pp. 81, 186, 197",
        "ADL": "Gateway Harmonized KLoSA F codebook, pp. 88–103",
        "IADL": "Gateway Harmonized KLoSA F codebook, pp. 88–103",
        "Mental health": "Gateway Harmonized KLoSA F codebook, pp. 127–135, 1111–1112",
        "Nutrition and body composition": "Gateway Harmonized KLoSA F codebook, pp. 178–180",
    },
}


def _measurement_note(cohort: str, item_id: str) -> str:
    """记录只影响解释、不改变数值方向的实质测量差异。"""
    if cohort == "HRS" and item_id.startswith(("adl_", "iadl_")):
        return "Any difficulty; HRS .x (‘don't do’) remains NA."
    if cohort == "ELSA" and item_id.startswith(("adl_", "iadl_")):
        return "Any difficulty; negative proxy/DK/refusal codes remain NA."
    if cohort == "KLoSA" and item_id.startswith(("adl_", "iadl_")):
        if item_id == "adl_bed_transfer":
            return "Needs help; source combines bed transfer and walking across a room."
        if item_id == "iadl_going_out_short_distance":
            return "Needs help going out a short distance; not equivalent to Tian's 1-km walking item."
        return "Needs help rather than any difficulty."
    if cohort == "ELSA" and item_id == "self_rated_health":
        return "Wave 3 uses the European r3shlta scale; retained with wave-specific mapping."
    if cohort == "KLoSA" and item_id == "hearing":
        return "Observed values are 1–5; the copied code-6 ‘Blind’ label is not accepted for hearing."
    if cohort == "KLoSA" and item_id == "underweight":
        return (
            "Derived from self-reported height and weight; official KLoSA category 1 is "
            "BMI <18.5 kg/m². Categories and source are stable in W1–W9."
        )
    if item_id == "depression":
        return (
            "CESD10 divided by 30; source changes from cesd10a to cesd10b at wave 5."
            if cohort == "KLoSA"
            else "Eight-item harmonized CESD score divided by 8."
        )
    if item_id == "mobility_walk_short":
        return (
            "Difficulty walking one block; not the Tian 1-km item."
            if cohort == "HRS"
            else "Difficulty walking 100 yards; not the Tian 1-km item."
        )
    return ""


def _build_specs() -> dict[str, tuple[ComponentSpec, ...]]:
    """构造唯一版本化 mapping；不从旧项目读取派生数据。"""
    result: dict[str, tuple[ComponentSpec, ...]] = {}

    hrs_specs: list[ComponentSpec] = []
    hrs_wave2_items = {
        item
        for item in PRIMARY_ITEMS["HRS"]
        if item.startswith(("adl_", "iadl_", "mobility_")) or item == "depression"
    }
    for item_id in PRIMARY_ITEMS["HRS"]:
        first = 2 if item_id in hrs_wave2_items else 1
        base = HRS_BASES[item_id]
        domain = DOMAIN_BY_ITEM[item_id]
        hrs_specs.append(
            ComponentSpec(
                item_id=item_id,
                domain=domain,
                rule=_rule_for("HRS", item_id),
                sources=_wave_sources(base, first, 16),
                source_summary=_summary(base, first, 16),
                codebook_ref=CODEBOOK_REFS["HRS"][domain],
                measurement_note=_measurement_note("HRS", item_id),
                attention_waves=tuple(range(2, 17)) if item_id == "depression" else (),
            )
        )
    result["HRS"] = tuple(hrs_specs)

    elsa_specs: list[ComponentSpec] = []
    for item_id in PRIMARY_ITEMS["ELSA"]:
        base = ELSA_BASES[item_id]
        domain = DOMAIN_BY_ITEM[item_id]
        sources = _wave_sources(base, 1, 10)
        source_summary = _summary(base, 1, 10)
        attention: tuple[int, ...] = ()
        if item_id == "self_rated_health":
            sources = tuple(
                (wave, "r3shlta" if wave == 3 else f"r{wave}shlt")
                for wave in range(1, 11)
            )
            source_summary = "r1shlt–r2shlt; r3shlta; r4shlt–r10shlt"
            attention = (3,)
        elsa_specs.append(
            ComponentSpec(
                item_id=item_id,
                domain=domain,
                rule=_rule_for("ELSA", item_id),
                sources=sources,
                source_summary=source_summary,
                codebook_ref=CODEBOOK_REFS["ELSA"][domain],
                measurement_note=_measurement_note("ELSA", item_id),
                attention_waves=attention,
            )
        )
    result["ELSA"] = tuple(elsa_specs)

    klosa_specs: list[ComponentSpec] = []
    for item_id in PRIMARY_ITEMS["KLoSA"]:
        domain = DOMAIN_BY_ITEM[item_id]
        attention: tuple[int, ...] = ()
        if item_id == "depression":
            sources = tuple(
                (wave, f"r{wave}cesd10a" if wave <= 4 else f"r{wave}cesd10b")
                for wave in range(1, 10)
            )
            source_summary = "r1cesd10a–r4cesd10a; r5cesd10b–r9cesd10b"
            attention = (5,)
        else:
            base = KLOSA_BASES[item_id]
            sources = _wave_sources(base, 1, 9)
            source_summary = _summary(base, 1, 9)
        klosa_specs.append(
            ComponentSpec(
                item_id=item_id,
                domain=domain,
                rule=_rule_for("KLoSA", item_id),
                sources=sources,
                source_summary=source_summary,
                codebook_ref=CODEBOOK_REFS["KLoSA"][domain],
                measurement_note=_measurement_note("KLoSA", item_id),
                attention_waves=attention,
            )
        )
    result["KLoSA"] = tuple(klosa_specs)
    return result


COMPONENT_SPECS = _build_specs()


def mapping_records() -> list[dict[str, Any]]:
    """返回可哈希、可测试的完整 mapping 记录。"""
    records: list[dict[str, Any]] = []
    for cohort in COHORTS:
        for spec in COMPONENT_SPECS[cohort]:
            records.append(
                {
                    "cohort": cohort,
                    "item_id": spec.item_id,
                    "domain": spec.domain,
                    "rule": spec.rule,
                    "sources": [[wave, source] for wave, source in spec.sources],
                    "source_summary": spec.source_summary,
                    "codebook_ref": spec.codebook_ref,
                    "measurement_note": spec.measurement_note,
                    "attention_waves": list(spec.attention_waves),
                    "primary_candidate": True,
                    "common_22": spec.item_id in COMMON_22,
                }
            )
    return records


def map_version() -> str:
    """以可执行配置与登记 map 的内容哈希绑定唯一 map version。"""
    payload = json.dumps(
        {
            "mapping_records": mapping_records(),
            "registered_variable_map_sha256": sha256_file(VARIABLE_MAP_PATH),
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()[:16]
    return f"{MAP_SCHEMA_VERSION}+{digest}"


def verify_variable_map_contract() -> None:
    """核对登记在 variable_maps 的候选集合与 v2 可执行映射。"""
    rows = _read_csv_rows(VARIABLE_MAP_PATH)
    for cohort in COHORTS:
        registered = {
            row["deficit_id"]
            for row in rows
            if row["cohort"] == cohort
            and row["primary_candidate"].strip().upper() == "TRUE"
        }
        if registered != set(PRIMARY_ITEMS[cohort]):
            raise RuntimeError(
                f"{cohort}: fi_component_map.csv 与 v2 候选集合不一致"
            )

    underweight_rows = [
        row
        for row in rows
        if row["cohort"] == "KLoSA" and row["deficit_id"] == "underweight"
    ]
    if len(underweight_rows) != 1:
        raise RuntimeError("KLoSA underweight 在 fi_component_map.csv 中必须恰有一行")
    underweight = underweight_rows[0]
    if (
        underweight["source_variable_pattern"] != "r{wave}bmicat"
        or underweight["scoring_rule_id"] != "underweight_bmicat"
        or underweight["common22_member"].strip().upper() != "FALSE"
    ):
        raise RuntimeError("KLoSA underweight 登记映射不符合 v2 契约")
    if any(
        row["cohort"] == "KLoSA"
        and row["deficit_id"] == "liver_disease"
        and row["primary_candidate"].strip().upper() == "TRUE"
        for row in rows
    ):
        raise RuntimeError("KLoSA liver_disease 仍被登记为 v2 primary candidate")


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    """以 UTF-8 读取 CSV。"""
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _repo_path(value: str | Path) -> Path:
    """把 freeze 中的相对路径解析到仓库根目录。"""
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _relative(path: Path) -> str:
    """返回用于审计文件的 POSIX 风格仓库相对路径。"""
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def _json_write(path: Path, payload: dict[str, Any]) -> None:
    """稳定写入 UTF-8 JSON。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _manifest_path(cohort: str) -> Path:
    """返回队列 v2 manifest 的固定路径。"""
    return (
        OUTCOME_MANIFEST_ROOT
        / f"{cohort.lower()}_outcome_components_{RELEASE_VERSION}.json"
    )


def _verify_release_absent() -> None:
    """冻结后禁止覆盖；无 freeze 时也拒绝覆盖孤立的同名资产。"""
    expected = [
        REVIEW_PATH,
        DICTIONARY_PATH,
        FREEZE_PATH,
        *[
            DERIVED_ROOT
            / cohort
            / "04_outcome_components"
            / f"{cohort.lower()}_outcome_components_{RELEASE_VERSION}.parquet"
            for cohort in COHORTS
        ],
        *[
            _manifest_path(cohort)
            for cohort in COHORTS
        ],
    ]
    existing = [str(path) for path in expected if path.exists()]
    if FREEZE_PATH.exists():
        raise RuntimeError(
            f"Outcome Components {RELEASE_VERSION} 已冻结，禁止原地覆盖"
        )
    if existing:
        raise RuntimeError(f"发现未冻结的同名 Outcome Components 资产，拒绝覆盖：{existing}")


def _freeze_manifest_reference(
    panel_name: str,
    record: dict[str, Any],
) -> tuple[Path, str]:
    """统一读取三种 freeze record 中的活动 manifest 引用。"""
    if panel_name == "spine":
        return (
            _repo_path(record["build_manifest_path"]),
            str(record["build_manifest_sha256"]),
        )
    return (
        _repo_path(record["active_manifest_path"]),
        str(record["active_manifest_sha256"]),
    )


def verify_prepanels(
    load_spine: Callable[[str], tuple[pd.DataFrame, Path, dict[str, Any]]],
    load_candidate_intervals: Callable[
        [str], tuple[pd.DataFrame, Path, dict[str, Any]]
    ],
    load_eligibility_base: Callable[
        [str], tuple[pd.DataFrame, Path, dict[str, Any]]
    ],
) -> dict[str, dict[str, dict[str, Any]]]:
    """复用现有 loader，并核对三个 freeze 的文件、manifest、schema 与主键。"""
    definitions = {
        "spine": (
            MANIFEST_ROOT / "spine" / "spine_v2_freeze.json",
            load_spine,
            list(KEY_COLUMNS),
        ),
        "candidate_intervals": (
            MANIFEST_ROOT
            / "candidate_intervals"
            / "candidate_intervals_v3_freeze.json",
            load_candidate_intervals,
            ["cohort", "person_id", "candidate_shock_wave"],
        ),
        "eligibility_base": (
            MANIFEST_ROOT
            / "eligibility_base"
            / "eligibility_base_v2_freeze.json",
            load_eligibility_base,
            ["cohort", "person_id", "candidate_shock_wave"],
        ),
    }
    snapshot: dict[str, dict[str, dict[str, Any]]] = {}
    loaded_keys: dict[str, dict[str, pd.DataFrame]] = {
        cohort: {} for cohort in COHORTS
    }

    for panel_name, (freeze_path, loader, expected_key) in definitions.items():
        freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
        if freeze.get("freeze_status") != "FROZEN":
            raise RuntimeError(f"{panel_name}: freeze_status 不是 FROZEN")
        snapshot[panel_name] = {}
        for cohort in COHORTS:
            data, path, _manifest = loader(cohort)
            record = freeze["cohorts"][cohort]
            frozen_path = _repo_path(record["path"])
            if path.resolve() != frozen_path.resolve():
                raise RuntimeError(f"{cohort}: {panel_name} 活动路径与 freeze 不一致")
            actual_sha = sha256_file(path)
            if actual_sha != record["sha256"]:
                raise RuntimeError(f"{cohort}: {panel_name} 文件哈希与 freeze 不一致")
            if len(data) != int(record["row_count"]):
                raise RuntimeError(f"{cohort}: {panel_name} 行数与 freeze 不一致")
            if list(record["key"]) != expected_key:
                raise RuntimeError(f"{cohort}: {panel_name} freeze 主键不一致")
            if data.duplicated(expected_key).any():
                raise RuntimeError(f"{cohort}: {panel_name} 主键不唯一")
            if "column_count" in record and len(data.columns) != int(
                record["column_count"]
            ):
                raise RuntimeError(f"{cohort}: {panel_name} 列数与 freeze 不一致")

            manifest_path, expected_manifest_sha = _freeze_manifest_reference(
                panel_name, record
            )
            actual_manifest_sha = sha256_file(manifest_path)
            if actual_manifest_sha != expected_manifest_sha:
                raise RuntimeError(f"{cohort}: {panel_name} manifest 哈希与 freeze 不一致")

            snapshot[panel_name][cohort] = {
                "path": str(path.resolve()),
                "sha256": actual_sha,
                "manifest_path": str(manifest_path.resolve()),
                "manifest_sha256": actual_manifest_sha,
                "row_count": len(data),
                "column_count": len(data.columns),
                "dtypes": {column: str(data[column].dtype) for column in data.columns},
            }
            loaded_keys[cohort][panel_name] = data.loc[:, expected_key].copy()

    for cohort in COHORTS:
        pd.testing.assert_frame_equal(
            loaded_keys[cohort]["candidate_intervals"],
            loaded_keys[cohort]["eligibility_base"],
            check_dtype=True,
            check_exact=True,
        )
    return snapshot


def _verify_source_and_codebook(cohort: str) -> tuple[Path, str]:
    """验证当前 raw、匹配 codebook 和已冻结 Spine 的来源哈希。"""
    registry_rows = [
        row for row in _read_csv_rows(REGISTRY_PATH) if row["cohort"] == cohort
    ]
    codebook_rows = [
        row
        for row in _read_csv_rows(CODEBOOK_MANIFEST_PATH)
        if row["cohort"] == cohort
    ]
    if len(registry_rows) != 1 or len(codebook_rows) != 1:
        raise RuntimeError(f"{cohort}: registry/codebook manifest 必须恰有一行")
    registry = registry_rows[0]
    codebook = codebook_rows[0]
    if registry["version_pair_verified"].upper() != "TRUE":
        raise RuntimeError(f"{cohort}: 数据与 codebook 版本尚未配对验证")
    if codebook["usage_status"] != "authoritative":
        raise RuntimeError(f"{cohort}: 当前 codebook 不是 authoritative")
    if registry["dataset_version_expected"] != codebook["dataset_version"]:
        raise RuntimeError(f"{cohort}: registry 与 codebook 数据版本不一致")

    source_path = resolve_cohort_dataset_path(cohort)
    codebook_path = _repo_path(codebook["local_path"])
    if not source_path.is_file() or not codebook_path.is_file():
        raise FileNotFoundError(f"{cohort}: raw 或 codebook 文件缺失")
    if sha256_file(codebook_path) != codebook["sha256"]:
        raise RuntimeError(f"{cohort}: codebook 哈希不一致")

    spine_freeze = json.loads(
        (MANIFEST_ROOT / "spine" / "spine_v2_freeze.json").read_text(
            encoding="utf-8"
        )
    )
    expected_source_sha = spine_freeze["cohorts"][cohort]["source_dataset_sha256"]
    source_sha = sha256_file(source_path)
    if source_sha != expected_source_sha:
        raise RuntimeError(f"{cohort}: raw 哈希与冻结 Spine 来源哈希不一致")
    return source_path, source_sha


def required_source_variables(
    cohort: str,
    component_ids: Iterable[str] | None = None,
) -> list[str]:
    """返回指定队列构建所需的最小 raw 列集合。"""
    ids = set(component_ids) if component_ids is not None else None
    specs = [
        spec
        for spec in COMPONENT_SPECS[cohort]
        if ids is None or spec.item_id in ids
    ]
    variables = [source for spec in specs for _, source in spec.sources]
    return list(dict.fromkeys(variables))


def read_raw_source(
    source_path: Path,
    cohort: str,
    component_ids: Iterable[str] | None = None,
) -> pd.DataFrame:
    """从只读 DTA 直接读取最小来源列，不创建长期 source slice。"""
    registry = next(
        row for row in _read_csv_rows(REGISTRY_PATH) if row["cohort"] == cohort
    )
    columns = [registry["person_id_source"], *required_source_variables(cohort, component_ids)]
    return pd.read_stata(
        source_path,
        columns=columns,
        convert_categoricals=False,
        convert_missing=True,
        preserve_dtypes=True,
    )


def _is_stata_missing(value: Any) -> bool:
    """识别包括 .x 在内的 Stata tagged/system missing。"""
    return isinstance(value, StataMissingValue)


def score_series(
    values: pd.Series,
    rule: str,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """把原始值转换为 0–1，并返回 tagged/非法值诊断。"""
    tag_mask = values.map(_is_stata_missing).astype(bool)
    tags = pd.Series(pd.NA, index=values.index, dtype="string")
    if tag_mask.any():
        tags.loc[tag_mask] = values.loc[tag_mask].map(str).astype("string")
    numeric = pd.to_numeric(values.mask(tag_mask), errors="coerce")
    scored = pd.Series(pd.NA, index=values.index, dtype="Float64")

    if rule == "binary_01":
        valid = numeric.isin([0, 1])
        scored.loc[valid] = numeric.loc[valid].astype(float)
    elif rule == "ordinal_poor_to_excellent":
        mapping = {1: 1.0, 2: 0.75, 3: 0.5, 4: 0.25, 5: 0.0}
        scored = numeric.map(mapping).astype("Float64")
    elif rule == "ordinal_better_to_worse":
        mapping = {1: 0.0, 2: 0.25, 3: 0.5, 4: 0.75, 5: 1.0}
        scored = numeric.map(mapping).astype("Float64")
    elif rule == "ordinal_better_to_worse_blind":
        mapping = {1: 0.0, 2: 0.25, 3: 0.5, 4: 0.75, 5: 1.0, 6: 1.0}
        scored = numeric.map(mapping).astype("Float64")
    elif rule == "continuous_8":
        valid = numeric.between(0, 8, inclusive="both")
        scored.loc[valid] = numeric.loc[valid] / 8.0
    elif rule == "continuous_30":
        valid = numeric.between(0, 30, inclusive="both")
        scored.loc[valid] = numeric.loc[valid] / 30.0
    elif rule == "underweight_bmicat":
        valid = numeric.isin([1, 2, 3, 4, 5, 6])
        scored.loc[valid] = numeric.loc[valid].eq(1).astype(float)
    else:
        raise ValueError(f"未知计分规则：{rule}")

    invalid_numeric = numeric.notna() & scored.isna()
    return scored, tags, invalid_numeric, numeric


def _diagnostic_counts(values: pd.Series) -> dict[str, int]:
    """把有限类别频数转换为稳定 JSON 字典。"""
    return {
        str(value): int(count)
        for value, count in values.value_counts(dropna=False).sort_index().items()
    }


def build_components_from_raw(
    raw: pd.DataFrame,
    spine: pd.DataFrame,
    cohort: str,
    component_ids: Iterable[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """以 Full Spine 全量键和顺序为骨架生成候选组件宽表。"""
    if cohort not in COHORTS:
        raise ValueError(f"未知队列：{cohort}")
    if list(spine.columns[:3]) != list(KEY_COLUMNS):
        raise ValueError("Full Spine 前三列必须是 cohort/person_id/wave")
    if spine.duplicated(list(KEY_COLUMNS)).any():
        raise ValueError(f"{cohort}: Full Spine 主键不唯一")
    if set(spine["cohort"].astype(str).unique()) != {cohort}:
        raise ValueError(f"{cohort}: Full Spine cohort 值不一致")

    requested = tuple(component_ids) if component_ids is not None else PRIMARY_ITEMS[cohort]
    if len(set(requested)) != len(requested):
        raise ValueError("component_ids 含重复项")
    spec_by_id = {spec.item_id: spec for spec in COMPONENT_SPECS[cohort]}
    missing_specs = sorted(set(requested).difference(spec_by_id))
    if missing_specs:
        raise ValueError(f"{cohort}: 未映射组件 {missing_specs}")
    specs = [spec_by_id[item_id] for item_id in requested]

    id_source = next(
        row["person_id_source"]
        for row in _read_csv_rows(REGISTRY_PATH)
        if row["cohort"] == cohort
    )
    required = [id_source, *required_source_variables(cohort, requested)]
    missing_columns = [column for column in required if column not in raw.columns]
    if missing_columns:
        raise KeyError(f"{cohort}: raw 缺少来源列 {missing_columns}")

    raw_work = raw.loc[:, required].copy()
    raw_work["_person_id"] = normalise_person_id(raw_work[id_source], cohort)
    if raw_work["_person_id"].duplicated().any():
        raise ValueError(f"{cohort}: raw person ID 不唯一")
    raw_work = raw_work.set_index("_person_id")
    if not raw_work.index.is_unique:
        raise ValueError(f"{cohort}: raw person ID 索引不唯一")
    spine_persons = set(spine["person_id"].astype(str).unique())
    raw_persons = set(raw_work.index.astype(str))
    if spine_persons != raw_persons:
        raise ValueError(
            f"{cohort}: raw 与 Spine 人员集合不一致，"
            f"raw_only={len(raw_persons - spine_persons)}, "
            f"spine_only={len(spine_persons - raw_persons)}"
        )

    output = spine.loc[:, list(KEY_COLUMNS)].copy()
    output["cohort"] = output["cohort"].astype("string")
    output["person_id"] = output["person_id"].astype("string")
    output["wave"] = output["wave"].astype("int16")
    buffers = {
        spec.item_id: pd.Series(pd.NA, index=output.index, dtype="Float64")
        for spec in specs
    }
    diagnostics: list[dict[str, Any]] = []

    for wave in sorted(output["wave"].unique().tolist()):
        positions = np.flatnonzero(output["wave"].to_numpy() == wave)
        wave_spine = spine.iloc[positions]
        person_ids = wave_spine["person_id"].astype("string")
        observed = wave_spine["observed_in_wave"].eq(True).fillna(False).to_numpy()
        denominator = int(observed.sum())

        for spec in specs:
            source_variable = spec.source_for_wave(int(wave))
            if source_variable is None:
                diagnostics.append(
                    {
                        "cohort": cohort,
                        "component": spec.item_id,
                        "wave": int(wave),
                        "denominator": denominator,
                        "valid_count": 0,
                        "valid_rate": 0.0 if denominator else None,
                        "source_variable": None,
                        "supported": False,
                        "tag_counts": {},
                        "invalid_counts": {},
                        "tracking_note": "Structural non-support: source unavailable.",
                    }
                )
                continue

            aligned = raw_work[source_variable].reindex(person_ids.to_numpy())
            aligned.index = range(len(aligned))
            scored, tags, invalid_numeric, numeric = score_series(aligned, spec.rule)
            observed_series = pd.Series(observed, index=scored.index)
            valid_count = int((observed_series & scored.notna()).sum())
            tagged_observed = tags.loc[observed_series & tags.notna()]
            invalid_observed = numeric.loc[observed_series & invalid_numeric]
            scored.loc[~observed_series] = pd.NA
            buffers[spec.item_id].iloc[positions] = scored.array

            tracking_note = ""
            if int(wave) in spec.attention_waves:
                if cohort == "ELSA" and spec.item_id == "self_rated_health":
                    tracking_note = "Source/scale switch: r3shlta European scale."
                elif cohort == "KLoSA" and spec.item_id == "depression":
                    tracking_note = "Source switch: CESD10A to CESD10B."
                elif cohort == "HRS" and spec.item_id == "depression":
                    tracking_note = "Tracked HRS depression wave."

            diagnostics.append(
                {
                    "cohort": cohort,
                    "component": spec.item_id,
                    "wave": int(wave),
                    "denominator": denominator,
                    "valid_count": valid_count,
                    "valid_rate": valid_count / denominator if denominator else None,
                    "source_variable": source_variable,
                    "supported": True,
                    "tag_counts": _diagnostic_counts(tagged_observed),
                    "invalid_counts": _diagnostic_counts(invalid_observed),
                    "tracking_note": tracking_note,
                }
            )

    for item_id in requested:
        output[item_id] = buffers[item_id]

    component_columns = list(requested)
    interviewed = spine["observed_in_wave"].eq(True).fillna(False)
    if output.loc[~interviewed, component_columns].notna().any(axis=None):
        raise AssertionError(f"{cohort}: 非访谈 Spine 行出现组件值")
    values = output[component_columns].stack()
    if not values.between(0, 1, inclusive="both").all():
        raise AssertionError(f"{cohort}: 组件值超出 [0,1]")
    if not output.loc[:, list(KEY_COLUMNS)].equals(
        spine.loc[:, list(KEY_COLUMNS)].astype(
            {"cohort": "string", "person_id": "string", "wave": "int16"}
        )
    ):
        raise AssertionError(f"{cohort}: Outcome 键或行序与 Spine 不一致")
    return output, pd.DataFrame(diagnostics)


def validate_outcome_frame(
    frame: pd.DataFrame,
    spine: pd.DataFrame,
    cohort: str,
) -> None:
    """验证持久化 Outcome 表的 schema、主键、行序和取值范围。"""
    expected_columns = [*KEY_COLUMNS, *PRIMARY_ITEMS[cohort]]
    if list(frame.columns) != expected_columns:
        raise ValueError(f"{cohort}: Outcome 列顺序不一致")
    expected_dtypes = {
        "cohort": "string",
        "person_id": "string",
        "wave": "int16",
        **{item: "Float64" for item in PRIMARY_ITEMS[cohort]},
    }
    actual_dtypes = {column: str(frame[column].dtype) for column in frame.columns}
    if actual_dtypes != expected_dtypes:
        raise TypeError(
            f"{cohort}: Outcome dtype 不一致：actual={actual_dtypes}, expected={expected_dtypes}"
        )
    if len(frame) != len(spine) or frame.duplicated(list(KEY_COLUMNS)).any():
        raise ValueError(f"{cohort}: Outcome 行数或主键不正确")
    expected_key = spine.loc[:, list(KEY_COLUMNS)].astype(
        {"cohort": "string", "person_id": "string", "wave": "int16"}
    )
    pd.testing.assert_frame_equal(
        frame.loc[:, list(KEY_COLUMNS)],
        expected_key,
        check_dtype=True,
        check_exact=True,
    )
    values = frame.loc[:, list(PRIMARY_ITEMS[cohort])].stack()
    if not values.between(0, 1, inclusive="both").all():
        raise ValueError(f"{cohort}: Outcome 存在超出 [0,1] 的值")


def _schema(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """生成最小 manifest 所需的有序 schema。"""
    return [
        {
            "name": column,
            "dtype": str(frame[column].dtype),
            "nullable": bool(frame[column].isna().any()),
        }
        for column in frame.columns
    ]


def minimal_manifest(
    frame: pd.DataFrame,
    source_hash: str,
    spine_hash: str,
    output_hash: str,
) -> dict[str, Any]:
    """生成严格收敛的队列机器 manifest。"""
    return {
        "schema": _schema(frame),
        "row_count": int(len(frame)),
        "key": list(KEY_COLUMNS),
        "source_hash": source_hash,
        "spine_hash": spine_hash,
        "output_hash": output_hash,
        "builder_version": BUILDER_VERSION,
        "map_version": map_version(),
    }


def _coding_text(item_id: str) -> str:
    """生成 Component Map 的精简计分说明。"""
    if item_id == "underweight":
        return "KLoSA BMI category 1 (<18.5 kg/m²)=1; categories 2–6=0"
    if item_id == "depression":
        return "HRS/ELSA CESD ÷ 8; KLoSA CESD10 ÷ 30"
    if item_id in {"vision", "hearing", "self_rated_health"}:
        return "Ordered categories → 0, .25, .50, .75, 1; higher=worse"
    if item_id.startswith(("adl_", "iadl_")):
        return "0/1; HRS/ELSA any difficulty, KLoSA needs help"
    return "0=no; 1=yes"


def _joined_notes(specs: list[ComponentSpec]) -> str:
    """合并 codebook 定位和必要差异，避免重复空文本。"""
    parts: list[str] = []
    for spec in specs:
        for text in (spec.codebook_ref, spec.measurement_note):
            if text and text not in parts:
                parts.append(text)
    return " | ".join(parts)


def component_map_rows() -> list[dict[str, Any]]:
    """从同一 mapping 生成 review workbook 的组件总表。"""
    order = (*COMMON_22, *HRS_ELSA_EXTENSIONS, *KLOSA_EXTENSIONS)
    rows: list[dict[str, Any]] = []
    for item_id in order:
        by_cohort: dict[str, ComponentSpec] = {}
        for cohort in COHORTS:
            match = next(
                (
                    spec
                    for spec in COMPONENT_SPECS[cohort]
                    if spec.item_id == item_id
                ),
                None,
            )
            if match is not None:
                by_cohort[cohort] = match
        rows.append(
            {
                "Item": item_id,
                "Domain": DOMAIN_BY_ITEM[item_id],
                "HRS source": by_cohort.get("HRS").source_summary
                if "HRS" in by_cohort
                else "",
                "ELSA source": by_cohort.get("ELSA").source_summary
                if "ELSA" in by_cohort
                else "",
                "KLoSA source": by_cohort.get("KLoSA").source_summary
                if "KLoSA" in by_cohort
                else "",
                "HRS waves": _wave_range(by_cohort.get("HRS")),
                "ELSA waves": _wave_range(by_cohort.get("ELSA")),
                "KLoSA waves": _wave_range(by_cohort.get("KLoSA")),
                "Coding": _coding_text(item_id),
                "Primary FI": "Candidate: " + ", ".join(by_cohort),
                "Common-22": "Yes" if item_id in COMMON_22 else "No",
                "Note": _joined_notes(list(by_cohort.values())),
            }
        )
    return rows


def _wave_range(spec: ComponentSpec | None) -> str:
    """把连续波次压缩为简短范围。"""
    if spec is None:
        return ""
    waves = [wave for wave, _ in spec.sources]
    return f"W{min(waves)}–W{max(waves)}"


def _format_issue_counts(cohort: str, counts: dict[str, int]) -> str:
    """将特殊缺失码翻译为简洁人工提示。"""
    labels = {
        ".x": "don't do (.x)",
        ".s": "proxy-related (.s)",
        ".d": "DK (.d)",
        ".r": "refuse (.r)",
        ".m": "other missing (.m)",
        ".": "system missing",
        "-16": "proxy (-16)",
        "-4": "DK (-4)",
        "-18": "refuse (-18)",
    }
    parts = [
        f"{labels.get(str(code), str(code))}={count:,}"
        for code, count in sorted(counts.items(), key=lambda pair: (-pair[1], pair[0]))
        if count > 0
    ]
    return ", ".join(parts[:4])


def wave_support_rows(diagnostics: pd.DataFrame) -> list[dict[str, Any]]:
    """只输出人工需要关注的逐波支持记录。"""
    rows: list[dict[str, Any]] = []
    for row in diagnostics.itertuples(index=False):
        tag_counts = dict(row.tag_counts)
        invalid_counts = dict(row.invalid_counts)
        important_special = (
            tag_counts.get(".x", 0) > 0
            or tag_counts.get(".s", 0) > 0
            or invalid_counts.get("-16", 0) > 0
        )
        attention = (
            not bool(row.supported)
            or row.valid_rate is None
            or float(row.valid_rate) < 0.95
            or bool(row.tracking_note)
            or important_special
        )
        if not attention:
            continue
        issue_parts: list[str] = []
        if row.tracking_note:
            issue_parts.append(str(row.tracking_note))
        if row.valid_rate is not None and float(row.valid_rate) < 0.95:
            issue_parts.append(
                f"Valid {int(row.valid_count):,}/{int(row.denominator):,}."
            )
        special_text = _format_issue_counts(
            str(row.cohort), {**tag_counts, **invalid_counts}
        )
        if special_text:
            issue_parts.append(special_text + ".")
        rows.append(
            {
                "Cohort": str(row.cohort),
                "Component": str(row.component),
                "Wave": int(row.wave),
                "Valid %": float(row.valid_rate) if row.valid_rate is not None else None,
                "Main issue": " ".join(issue_parts).strip(),
            }
        )
    return rows


def adaptation_rows() -> list[dict[str, str]]:
    """列出影响跨队列解释或与 Tian A1 对照的实质差异。"""
    return [
        {
            "Item": "ADL/IADL items",
            "Cohort": "KLoSA",
            "Difference": "Needs help is a more severe threshold than HRS/ELSA any difficulty.",
            "Decision": "Retain as a documented cohort adaptation; do not claim identical severity.",
        },
        {
            "Item": "adl_bed_transfer",
            "Cohort": "KLoSA",
            "Difference": "bedb_k combines getting in/out of bed with walking across a room.",
            "Decision": "Retain under the common concept with the broader source explicitly documented.",
        },
        {
            "Item": "walking",
            "Cohort": "All",
            "Difference": "HRS uses one block, ELSA 100 yards, and KLoSA uses help going out a short distance; Tian specifies 1 km.",
            "Decision": "Use exact component names; do not describe any source as equivalent to 1 km.",
        },
        {
            "Item": "self_rated_health",
            "Cohort": "ELSA W3",
            "Difference": "r3shlta uses the European very-good-to-very-bad scale.",
            "Decision": "Retain with a wave-specific ordinal mapping; flag W3 in Wave Support.",
        },
        {
            "Item": "vision",
            "Cohort": "ELSA/KLoSA",
            "Difference": "A separate Blind category exists; HRS has a five-category poor-to-excellent direction.",
            "Decision": "Map Blind to 1 and reverse categories only where the official direction requires it.",
        },
        {
            "Item": "hearing",
            "Cohort": "KLoSA",
            "Difference": "A copied code-6 ‘Blind’ label appears in metadata but is not observed and is not a valid hearing category.",
            "Decision": "Accept only official observed categories 1–5; any future code 6 remains NA pending review.",
        },
        {
            "Item": "depression",
            "Cohort": "All",
            "Difference": "HRS/ELSA use an 0–8 harmonized CESD score; KLoSA uses 0–30 and switches source at W5.",
            "Decision": "Preserve continuous information using ÷8 and ÷30; do not copy Tian A1's inconsistent 0–10 statement.",
        },
        {
            "Item": "iadl_telephone",
            "Cohort": "All",
            "Difference": "It belongs to the historical common-22 but not Tian's 23-item standard.",
            "Decision": "Keep in common-22; do not silently replace it with Tian walking or cognition.",
        },
        {
            "Item": "memory/cognition",
            "Cohort": "All",
            "Difference": "Tian A1 includes memory-related disease and cognition, but the cross-cohort formula and full-wave support are incomplete.",
            "Decision": f"Comparison evidence only; no v2 component. Tian source: {TIAN_URL}",
        },
        {
            "Item": "KLoSA extensions",
            "Cohort": "KLoSA",
            "Difference": "Seven K-ADL/K-IADL items plus underweight replace HRS/ELSA mobility items in the cohort-specific candidate 30.",
            "Decision": "Retain brushing teeth, continence, housework, grooming, laundry, going out, transportation and underweight; exclude from common-22.",
        },
        {
            "Item": "underweight",
            "Cohort": "KLoSA",
            "Difference": "KLoSA-specific nutritional/body-composition deficit; BMI is derived from self-reported height and weight and is not a mobility item.",
            "Decision": "Use official rWbmicat category 1 (<18.5 kg/m²) as deficit; replace liver disease only, with the other seven extensions unchanged.",
        },
        {
            "Item": "special missing",
            "Cohort": "All",
            "Difference": "Proxy, DK, refusal, not asked and HRS .x (‘don't do’) are not evidence of no deficit.",
            "Decision": "Keep all such values as NA; never recode them to zero.",
        },
    ]


def _workbook_payload(diagnostics: pd.DataFrame) -> dict[str, Any]:
    """生成 artifact-tool 使用的三表 payload。"""
    return {
        "component_map": component_map_rows(),
        "wave_support": wave_support_rows(diagnostics),
        "adaptations": adaptation_rows(),
    }


WORKBOOK_BUILDER = r'''
import fs from "node:fs/promises";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const [payloadPath, outputPath, previewDir] = process.argv.slice(2);
const payload = JSON.parse(await fs.readFile(payloadPath, "utf8"));
const workbook = Workbook.create();
const fontFamily = "Arial";

function addTableSheet(name, rows, headers, tableName, widths, percentColumns = []) {
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
  header.format.rowHeightPx = 36;
  for (let i = 0; i < widths.length; i += 1) {
    sheet.getRangeByIndexes(0, i, matrix.length, 1).format.columnWidthPx = widths[i];
  }
  for (const columnIndex of percentColumns) {
    if (matrix.length > 1) {
      sheet.getRangeByIndexes(1, columnIndex, matrix.length - 1, 1).format.numberFormat = "0.0%";
    }
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
  return { sheet, rowCount: matrix.length };
}

const componentHeaders = ["Item", "Domain", "HRS source", "ELSA source", "KLoSA source", "HRS waves", "ELSA waves", "KLoSA waves", "Coding", "Primary FI", "Common-22", "Note"];
const supportHeaders = ["Cohort", "Component", "Wave", "Valid %", "Main issue"];
const adaptationHeaders = ["Item", "Cohort", "Difference", "Decision"];

const component = addTableSheet("1_Component Map", payload.component_map, componentHeaders, "OutcomeComponentMap", [165, 145, 185, 185, 205, 82, 82, 88, 275, 145, 82, 500]);
const support = addTableSheet("2_Wave Support", payload.wave_support, supportHeaders, "OutcomeWaveSupport", [90, 215, 62, 82, 520], [3]);
const adaptations = addTableSheet("3_Adaptations", payload.adaptations, adaptationHeaders, "OutcomeAdaptations", [225, 95, 465, 500]);

for (const target of [component, support, adaptations]) {
  const maxRows = Math.min(target.rowCount, 40);
  const maxCols = target.sheet.name === "1_Component Map" ? 12 : target.sheet.name === "2_Wave Support" ? 5 : 4;
  const inspected = await workbook.inspect({
    kind: "table",
    range: `${target.sheet.name}!A1:${String.fromCharCode(64 + maxCols)}${maxRows}`,
    include: "values,formulas",
    tableMaxRows: 8,
    tableMaxCols: maxCols,
    maxChars: 6000,
  });
  console.log(inspected.ndjson);
}

const errors = await workbook.inspect({
  kind: "match",
  searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!|#SPILL!|#CALC!",
  options: { useRegex: true, maxResults: 100 },
  summary: "final formula error scan",
});
console.log(errors.ndjson);

await fs.mkdir(previewDir, { recursive: true });
for (const [sheetName, range] of [["1_Component Map", "A1:L40"], ["2_Wave Support", "A1:E40"], ["3_Adaptations", "A1:D13"]]) {
  const preview = await workbook.render({ sheetName, range, scale: 1, format: "png" });
  const filename = sheetName.replaceAll(" ", "_") + ".png";
  await fs.writeFile(`${previewDir}/${filename}`, new Uint8Array(await preview.arrayBuffer()));
}

const xlsx = await SpreadsheetFile.exportXlsx(workbook);
await xlsx.save(outputPath);
'''


def _runtime_paths() -> tuple[Path, Path]:
    """定位 load_workspace_dependencies 提供的 Node 和 node_modules。"""
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
    """在系统临时目录创建 artifact-tool 所需的 node_modules 目录链接。"""
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
    diagnostics: pd.DataFrame,
    output_path: Path,
    preview_dir: Path,
) -> None:
    """用 artifact-tool 从同一 mapping/诊断对象生成三表 workbook。"""
    node, node_modules = _runtime_paths()
    with tempfile.TemporaryDirectory(prefix="outcome_components_xlsx_") as tmp_name:
        tmp = Path(tmp_name)
        payload_path = tmp / "payload.json"
        builder_path = tmp / "build_review.mjs"
        payload_path.write_text(
            json.dumps(_workbook_payload(diagnostics), ensure_ascii=False),
            encoding="utf-8",
        )
        builder_path.write_text(WORKBOOK_BUILDER, encoding="utf-8")
        _create_node_modules_link(tmp / "node_modules", node_modules)
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
        if completed.stdout:
            # Windows 控制台可能不是 UTF-8；仅转义日志中的不可编码字符。
            print(completed.stdout.encode("ascii", errors="backslashreplace").decode("ascii"))
        if not output_path.is_file():
            raise RuntimeError("artifact-tool 未生成 review workbook")


DICTIONARY_TEXT = """# Outcome Components v2 数据字典

## 数据粒度与主键

每一行表示一个队列中的一个人次波次（person-wave）。主键固定为 `cohort + person_id + wave`，行集合和行顺序与对应的冻结 Full Spine `spine_v2` 完全一致。Candidate Intervals 与 Eligibility Base 不参与筛行。

## 列与数值含义

- `cohort`：`HRS`、`ELSA` 或 `KLoSA`。
- `person_id`：队列内稳定的字符串型参与者标识。
- `wave`：队列内登记波次。
- 其余列：经核实的候选健康缺陷组件。组件值在 0–1 之间，数值越高表示缺陷越重。

二元组件使用 0/1；有序感官或自评健康类别映射为 0、0.25、0.50、0.75、1；HRS/ELSA 的 harmonized CESD 除以 8，KLoSA CESD10 除以 30。KLoSA `underweight` 使用官方 `rWbmicat`：类别 1（BMI <18.5 kg/m²）为 1，类别 2–6 为 0。逐项来源、波次、编码和适配见 `outcome_components_review_v2.xlsx`。

## 缺失值

`NA` 表示该波未测、来源不可用、不适用、个体缺失或特殊缺失，不能解释为“无缺陷”。Proxy、don't know、refusal、not asked、HRS `.x`（don't do）及其他 tagged missing 均保持 `NA`，不会编码为 0。

Full Spine 中 `observed_in_wave != TRUE` 的行是结构骨架，而不是当次有效访谈；这些行的全部组件值均为 `NA`。本层不利用未来信息填补过去值，不做 LOCF 或插补。

## 与正式 FI 的边界

本层只保存候选组件值，不包含 FI 总分、FI 分类、完整性阈值、分母、最终资格、wealth、recovery、权重或模型结果。HRS/ELSA 各保留 30 个旧主候选组件；KLoSA 保留 common-22 加八个有来源支持的队列特异组件（7 个 K-ADL/K-IADL 项与 underweight）。v2 仅以 underweight 替换 v1 的 liver disease；其余条目不变。共同 22 项仅是后续测量敏感性的候选集合，不等于 Tian 等人的 23 项。

正式主 FI 与 common-22 FI 必须在后续单独获批的步骤中按明确名单和完整性规则生成。
"""


def _final_output_path(cohort: str) -> Path:
    """返回队列 Outcome Parquet 的固定 v2 路径。"""
    return (
        DERIVED_ROOT
        / cohort
        / "04_outcome_components"
        / f"{cohort.lower()}_outcome_components_{RELEASE_VERSION}.parquet"
    )


def _build_freeze(
    staged_outputs: dict[str, Path],
    staged_manifests: dict[str, Path],
    staged_review: Path,
    staged_dictionary: Path,
) -> dict[str, Any]:
    """生成唯一跨队列 Outcome Components freeze record。"""
    return {
        "freeze_record_schema_version": "outcome_components_freeze_record_v2",
        "freeze_status": "FROZEN",
        "frozen_at_utc": datetime.now(UTC).isoformat(),
        "builder_version": BUILDER_VERSION,
        "map_version": map_version(),
        "common_key": list(KEY_COLUMNS),
        "component_range": [0.0, 1.0],
        "non_interview_policy": "All component values are NA when observed_in_wave is not TRUE.",
        "overwrite_policy": "Do not overwrite Outcome Components v2 or this freeze record; create a new version.",
        "supersedes": "outcome_components_v1",
        "change_summary": "KLoSA underweight replaces liver_disease; all other candidate items are unchanged.",
        "registered_variable_map": {
            "path": _relative(VARIABLE_MAP_PATH),
            "sha256": sha256_file(VARIABLE_MAP_PATH),
        },
        "cohorts": {
            cohort: {
                "parquet_path": _relative(_final_output_path(cohort)),
                "parquet_sha256": sha256_file(staged_outputs[cohort]),
                "manifest_path": _relative(
                    _manifest_path(cohort)
                ),
                "manifest_sha256": sha256_file(staged_manifests[cohort]),
                "candidate_item_count": len(PRIMARY_ITEMS[cohort]),
            }
            for cohort in COHORTS
        },
        "review_workbook": {
            "path": _relative(REVIEW_PATH),
            "sha256": sha256_file(staged_review),
            "sheets": ["1_Component Map", "2_Wave Support", "3_Adaptations"],
        },
        "data_dictionary": {
            "path": _relative(DICTIONARY_PATH),
            "sha256": sha256_file(staged_dictionary),
        },
    }


def _publish(staged: Path, final: Path) -> None:
    """验证完成后发布单个资产，不覆盖已有文件。"""
    if final.exists():
        raise RuntimeError(f"拒绝覆盖已有文件：{final}")
    final.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(staged), str(final))


def run_outcome_components(
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
    """一次性构建三队列精简 Outcome Components v2 并冻结。"""
    if tuple(cohorts) != COHORTS:
        raise RuntimeError(
            "outcome-components v2 必须使用 --cohort all 一次性生成三队列共同 review/freeze"
        )
    verify_variable_map_contract()
    _verify_release_absent()
    pre_snapshot = verify_prepanels(
        load_spine, load_candidate_intervals, load_eligibility_base
    )

    source_info = {
        cohort: _verify_source_and_codebook(cohort) for cohort in COHORTS
    }
    all_diagnostics: list[pd.DataFrame] = []
    staged_outputs: dict[str, Path] = {}
    staged_manifests: dict[str, Path] = {}

    qa_preview_override = os.environ.get("OUTCOME_COMPONENTS_PREVIEW_DIR")
    staging_context = tempfile.TemporaryDirectory(prefix="outcome_components_v2_")
    try:
        staging = Path(staging_context.name)
        preview_dir = (
            Path(qa_preview_override)
            if qa_preview_override
            else staging / "workbook_previews"
        )

        for cohort in COHORTS:
            print(f"{cohort}: reading selected Outcome Components columns")
            source_path, source_hash = source_info[cohort]
            raw = read_raw_source(source_path, cohort)
            spine, _spine_path, _spine_manifest = load_spine(cohort)
            outcome, diagnostics = build_components_from_raw(raw, spine, cohort)
            del raw
            staged_output = (
                staging
                / f"{cohort.lower()}_outcome_components_{RELEASE_VERSION}.parquet"
            )
            outcome.to_parquet(staged_output, index=False, compression="zstd")
            persisted = pd.read_parquet(staged_output)
            validate_outcome_frame(persisted, spine, cohort)
            pd.testing.assert_frame_equal(
                persisted,
                outcome,
                check_dtype=True,
                check_exact=True,
            )
            staged_outputs[cohort] = staged_output
            all_diagnostics.append(diagnostics)

            manifest = minimal_manifest(
                persisted,
                source_hash,
                pre_snapshot["spine"][cohort]["sha256"],
                sha256_file(staged_output),
            )
            staged_manifest = staging / _manifest_path(cohort).name
            _json_write(staged_manifest, manifest)
            staged_manifests[cohort] = staged_manifest
            print(
                f"{cohort}: Outcome Components PASS | rows={len(persisted):,} "
                f"components={len(PRIMARY_ITEMS[cohort])}"
            )
            del outcome, persisted, spine

        diagnostics = pd.concat(all_diagnostics, ignore_index=True)
        staged_review = staging / REVIEW_PATH.name
        create_review_workbook(diagnostics, staged_review, preview_dir)
        staged_dictionary = staging / DICTIONARY_PATH.name
        staged_dictionary.write_text(DICTIONARY_TEXT, encoding="utf-8")
        staged_freeze = staging / FREEZE_PATH.name
        _json_write(
            staged_freeze,
            _build_freeze(
                staged_outputs,
                staged_manifests,
                staged_review,
                staged_dictionary,
            ),
        )

        post_snapshot = verify_prepanels(
            load_spine, load_candidate_intervals, load_eligibility_base
        )
        if post_snapshot != pre_snapshot:
            raise RuntimeError("Outcome 构建前后三张 pre-panel 状态发生变化")

        for cohort in COHORTS:
            _publish(staged_outputs[cohort], _final_output_path(cohort))
        _publish(staged_review, REVIEW_PATH)
        _publish(staged_dictionary, DICTIONARY_PATH)
        for cohort in COHORTS:
            _publish(
                staged_manifests[cohort],
                _manifest_path(cohort),
            )
        _publish(staged_freeze, FREEZE_PATH)
    finally:
        staging_context.cleanup()

    return {
        "status": "FROZEN",
        "builder_version": BUILDER_VERSION,
        "map_version": map_version(),
        "freeze_path": str(FREEZE_PATH),
        "review_path": str(REVIEW_PATH),
        "preview_dir": qa_preview_override,
    }
