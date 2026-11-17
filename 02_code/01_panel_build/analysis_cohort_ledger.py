"""构建 Analysis Cohort Ledger v1 人工复核候选。"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import subprocess
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from panel_core import RUNNER_VERSION, sha256_file


ROOT = Path(__file__).resolve().parents[2]
MANIFEST_ROOT = ROOT / "00_admin" / "panel_manifests"
LEDGER_MANIFEST_ROOT = MANIFEST_ROOT / "analysis_cohort_ledger"
DESIGN_REGISTER_PATH = ROOT / "00_admin" / "panel_design_register.csv"
REVIEW_PATH = ROOT / "04_collaboration" / "analysis_cohort_ledger_review_v1.csv"
DICTIONARY_PATH = LEDGER_MANIFEST_ROOT / "analysis_cohort_ledger_data_dictionary.md"
RELEASE_CANDIDATE_PATH = (
    LEDGER_MANIFEST_ROOT / "analysis_cohort_ledger_v1_release_candidate.json"
)
FORMAL_FREEZE_PATH = LEDGER_MANIFEST_ROOT / "analysis_cohort_ledger_v1_freeze.json"
EXPECTED_APPROVED_CANDIDATE_SHA256 = (
    "be7468c322d003fab00e16178d5bc0fb1c929e1d6b4894cbd61ca77847403fcb"
)

COHORTS = ("HRS", "ELSA", "KLoSA")
RELEASE_VERSION = "v1"
SCHEMA_VERSION = "analysis_cohort_ledger_v1"
BUILDER_VERSION = "analysis_cohort_ledger_builder_v1.0.0"

FREEZE_PATHS = {
    "spine_v2": MANIFEST_ROOT / "spine" / "spine_v2_freeze.json",
    "fi_panel_v2": MANIFEST_ROOT / "fi_panel" / "fi_panel_v2_freeze.json",
    "exposure_panel_v1": (
        MANIFEST_ROOT / "exposure_panel" / "exposure_panel_v1_freeze.json"
    ),
    "covariate_panel_v1": (
        MANIFEST_ROOT / "covariate_panel" / "covariate_panel_v1_freeze.json"
    ),
    "weight_design_panel_v1": (
        MANIFEST_ROOT
        / "weight_design_panel"
        / "weight_design_panel_v1_freeze.json"
    ),
}

LEDGER_COLUMNS = (
    "cohort",
    "person_id",
    "tminus1_wave",
    "index_shock_wave",
    "landmark_wave",
    "recovery_status",
    "recovery_classifiable",
    "fi_t_available",
    "landmark_fi_available",
    "later_fi_count",
    "nine_covariates_complete",
    "missing_adjustment_variable_count",
    "missing_adjustment_variables",
    "longitudinal_support_complete",
    "complete_case_support_complete",
    "landmark_weight_status",
    "weighted_complete_case_support_complete",
    "first_failed_gate",
)

LEDGER_DTYPES = {
    "cohort": "string",
    "person_id": "string",
    "tminus1_wave": "Int16",
    "index_shock_wave": "Int16",
    "landmark_wave": "Int16",
    "recovery_status": "string",
    "recovery_classifiable": "boolean",
    "fi_t_available": "boolean",
    "landmark_fi_available": "boolean",
    "later_fi_count": "Int16",
    "nine_covariates_complete": "boolean",
    "missing_adjustment_variable_count": "Int8",
    "missing_adjustment_variables": "string",
    "longitudinal_support_complete": "boolean",
    "complete_case_support_complete": "boolean",
    "landmark_weight_status": "string",
    "weighted_complete_case_support_complete": "boolean",
    "first_failed_gate": "string",
}

NULLABLE_COLUMNS = {"recovery_status", "missing_adjustment_variables"}
RECOVERY_VALUES = {
    "substantial_recovery",
    "persistent_or_limited_recovery",
}
WEIGHT_STATUS_VALUES = {"positive", "zero", "missing"}
FIRST_FAILED_GATE_VALUES = {
    "recovery_unavailable",
    "landmark_fi_missing",
    "no_later_fi",
    "fi_t_missing",
    "prespecified_covariate_missing",
    "none",
}

T_COVARIATES = (
    "age",
    "sex",
    "education",
    "marital_status",
    "household_size",
    "labour_force_status",
    "household_income_rank",
    "current_smoking",
)
ADJUSTMENT_VARIABLES = ("fi_t", *T_COVARIATES, "wealth_quintile")

EXPECTED_FLOW = {
    "HRS": {
        "shock_experienced_cohort": 8204,
        "longitudinal_support_complete": 5222,
        "complete_case_support_complete": 5114,
        "weighted_complete_case_support_complete": 5097,
    },
    "ELSA": {
        "shock_experienced_cohort": 1284,
        "longitudinal_support_complete": 772,
        "complete_case_support_complete": 699,
        "weighted_complete_case_support_complete": 676,
    },
    "KLoSA": {
        "shock_experienced_cohort": 3125,
        "longitudinal_support_complete": 2507,
        "complete_case_support_complete": 2502,
        "weighted_complete_case_support_complete": 2502,
    },
}

EXPECTED_FIRST_FAILED = {
    "HRS": {
        "recovery_unavailable": 1657,
        "landmark_fi_missing": 212,
        "no_later_fi": 1113,
        "fi_t_missing": 72,
        "prespecified_covariate_missing": 36,
        "none": 5114,
    },
    "ELSA": {
        "recovery_unavailable": 316,
        "landmark_fi_missing": 0,
        "no_later_fi": 196,
        "fi_t_missing": 0,
        "prespecified_covariate_missing": 73,
        "none": 699,
    },
    "KLoSA": {
        "recovery_unavailable": 370,
        "landmark_fi_missing": 0,
        "no_later_fi": 248,
        "fi_t_missing": 0,
        "prespecified_covariate_missing": 5,
        "none": 2502,
    },
}

EXPECTED_RECOVERY_COUNTS = {
    "HRS": {
        "shock_experienced_cohort": {
            "substantial_recovery": 2088,
            "persistent_or_limited_recovery": 4459,
            "unavailable": 1657,
        },
        "longitudinal_support_complete": {
            "substantial_recovery": 1733,
            "persistent_or_limited_recovery": 3489,
        },
        "complete_case_support_complete": {
            "substantial_recovery": 1703,
            "persistent_or_limited_recovery": 3411,
        },
    },
    "ELSA": {
        "shock_experienced_cohort": {
            "substantial_recovery": 362,
            "persistent_or_limited_recovery": 606,
            "unavailable": 316,
        },
        "longitudinal_support_complete": {
            "substantial_recovery": 294,
            "persistent_or_limited_recovery": 478,
        },
        "complete_case_support_complete": {
            "substantial_recovery": 271,
            "persistent_or_limited_recovery": 428,
        },
    },
    "KLoSA": {
        "shock_experienced_cohort": {
            "substantial_recovery": 1024,
            "persistent_or_limited_recovery": 1731,
            "unavailable": 370,
        },
        "longitudinal_support_complete": {
            "substantial_recovery": 944,
            "persistent_or_limited_recovery": 1563,
        },
        "complete_case_support_complete": {
            "substantial_recovery": 943,
            "persistent_or_limited_recovery": 1559,
        },
    },
}


DICTIONARY_TEXT = """# Analysis Cohort Ledger v1 data dictionary

## 粒度与作用

每行表示一名已由冻结 Exposure Panel v1 识别出 analytic index shock 的参与者，主键为 `cohort + person_id`。Ledger 全部行即 shock-experienced cohort。该表只保存测量、调整变量和权重支持状态，不是最终模型长表，也不保存 FI、协变量、财富或原始权重数值。

已获作者批准的层级为：不加权完整案例主分析，exact-`t+1` landmark 加权及协变量多重插补作为敏感性分析。本候选不修改 design register。`complete_case_support_complete` 在完成最终访谈时间和模型输入检查前仍是支持候选，而非最终 fitted sample。

## 列定义

| Column | Type | Nullable | Definition |
|---|---|---:|---|
| `cohort` | string | no | HRS、ELSA 或 KLoSA。 |
| `person_id` | string | no | 队列内参与者标识；不得出现在公开汇总中。 |
| `tminus1_wave` | Int16 | no | Index shock 前一相邻波次。 |
| `index_shock_wave` | Int16 | no | 队列特异搜索窗口内首次 qualifying major shock 的波次。 |
| `landmark_wave` | Int16 | no | Index shock 后下一预定波次，即 `t+1`。 |
| `recovery_status` | string | yes | `substantial_recovery`、`persistent_or_limited_recovery`或NA。 |
| `recovery_classifiable` | boolean | no | `recovery_status`属于两个正式主分类。 |
| `fi_t_available` | boolean | no | Shock interview `t`的primary FI有效。 |
| `landmark_fi_available` | boolean | no | Landmark `t+1`的primary FI有效。 |
| `later_fi_count` | Int16 | no | Landmark之后有效primary FI的次数；不要求连续访谈。 |
| `nine_covariates_complete` | boolean | no | 八项`t`协变量及`t-1` wealth quintile均有效。 |
| `missing_adjustment_variable_count` | Int8 | no | `FI_t`加九项协变量中缺失的数量，范围0--10。 |
| `missing_adjustment_variables` | string | yes | 全部缺失调整变量的分号分隔名称；计数为0时NA。 |
| `longitudinal_support_complete` | boolean | no | Recovery可分类、landmark FI有效且至少一次later FI。 |
| `complete_case_support_complete` | boolean | no | 纵向支持完整，且`FI_t`与九项协变量完整。 |
| `landmark_weight_status` | string | no | Exact-`t+1` raw person weight的`positive`、`zero`或`missing`状态。 |
| `weighted_complete_case_support_complete` | boolean | no | 完整案例支持候选且landmark weight为正；只是敏感性支持分支。 |
| `first_failed_gate` | string | no | 按固定gate顺序记录第一个失败条件；通过全部完整案例支持gate时为`none`。 |

## 调整变量范围

- `t`：`age`, `sex`, `education`, `marital_status`, `household_size`, `labour_force_status`, `household_income_rank`, `current_smoking`。
- `t-1`：`wealth_quintile`。
- 另计：`fi_t`。

财富损失比例不进入缺失计数。每一行必须在冻结 Exposure Panel 中满足 `is_index_shock=TRUE`、`major_shock=TRUE`，并具有有限的 `wealth_loss_fraction >= 0.75`；允许损失比例大于1。

## Gate顺序与边界

`first_failed_gate`依次检查：`recovery_unavailable`、`landmark_fi_missing`、`no_later_fi`、`fi_t_missing`、`prespecified_covariate_missing`，全部通过为`none`。该字段是操作性流程原因；`no_later_fi`不等同于死亡或失访。`none`不表示权重为正，也不表示已经通过最终模型的访谈时间与输入检查。

这些flags仅描述测量与协变量支持。最终模型资格还须验证landmark及拟使用FI记录的实际访谈时间。一条记录时间不可用不自动排除整名参与者；不得在Ledger阶段填补年份、寻找替代波次或静默删人。`FI_t`是否允许在未来MI敏感性中插补，由独立MI实施规则决定。
"""


def _relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def _json_write(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def _first(mapping: dict[str, Any], names: tuple[str, ...]) -> Any:
    for name in names:
        value = mapping.get(name)
        if value not in (None, ""):
            return value
    return None


def _cohort_asset_entry(freeze: dict[str, Any], cohort: str) -> dict[str, Any]:
    entry = freeze["cohorts"][cohort]
    data_path = _first(entry, ("parquet_path", "path", "data_path"))
    data_sha = _first(entry, ("parquet_sha256", "sha256", "data_sha256"))
    manifest_path = _first(entry, ("manifest_path", "build_manifest_path"))
    manifest_sha = _first(entry, ("manifest_sha256", "build_manifest_sha256"))
    if None in (data_path, data_sha, manifest_path, manifest_sha):
        raise RuntimeError(f"{cohort}: freeze缺少数据或manifest路径/哈希")
    return {
        "data_path": ROOT / str(data_path),
        "data_sha256": str(data_sha),
        "manifest_path": ROOT / str(manifest_path),
        "manifest_sha256": str(manifest_sha),
        "row_count": int(entry["row_count"]),
    }


def protected_upstream_snapshot() -> dict[str, Any]:
    """核对五个冻结输入及其三队列数据和manifest。"""
    snapshot: dict[str, Any] = {}
    for panel_name, freeze_path in FREEZE_PATHS.items():
        if not freeze_path.is_file():
            raise RuntimeError(f"缺少冻结记录：{freeze_path}")
        freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
        if freeze.get("freeze_status") != "FROZEN":
            raise RuntimeError(f"{panel_name}: freeze_status不是FROZEN")
        cohort_records: dict[str, Any] = {}
        for cohort in COHORTS:
            entry = _cohort_asset_entry(freeze, cohort)
            if sha256_file(entry["data_path"]) != entry["data_sha256"]:
                raise RuntimeError(f"{panel_name}/{cohort}: Parquet哈希漂移")
            if sha256_file(entry["manifest_path"]) != entry["manifest_sha256"]:
                raise RuntimeError(f"{panel_name}/{cohort}: manifest哈希漂移")
            cohort_records[cohort] = {
                "data_path": _relative(entry["data_path"]),
                "data_sha256": entry["data_sha256"],
                "manifest_path": _relative(entry["manifest_path"]),
                "manifest_sha256": entry["manifest_sha256"],
                "row_count": entry["row_count"],
            }
        snapshot[panel_name] = {
            "freeze_path": _relative(freeze_path),
            "freeze_sha256": sha256_file(freeze_path),
            "cohorts": cohort_records,
        }
    return snapshot


def _load_frozen_panel(
    panel_name: str, cohort: str, columns: list[str] | None = None
) -> pd.DataFrame:
    freeze_path = FREEZE_PATHS[panel_name]
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    entry = _cohort_asset_entry(freeze, cohort)
    frame = pd.read_parquet(entry["data_path"], columns=columns)
    if len(frame) != entry["row_count"]:
        raise RuntimeError(f"{panel_name}/{cohort}: 行数与freeze不一致")
    return frame


def _final_output_path(cohort: str) -> Path:
    return (
        ROOT
        / "01_data"
        / "derived"
        / "panel_components"
        / cohort
        / "10_analysis_cohort_ledger"
        / f"{cohort.lower()}_analysis_cohort_ledger_v1.parquet"
    )


def _manifest_path(cohort: str) -> Path:
    return LEDGER_MANIFEST_ROOT / f"{cohort.lower()}_analysis_cohort_ledger_v1.json"


def _release_paths() -> list[Path]:
    return [
        *(_final_output_path(cohort) for cohort in COHORTS),
        *(_manifest_path(cohort) for cohort in COHORTS),
        REVIEW_PATH,
        DICTIONARY_PATH,
        RELEASE_CANDIDATE_PATH,
    ]


def _verify_release_absent() -> None:
    existing = [str(path) for path in _release_paths() if path.exists()]
    if existing:
        raise RuntimeError(f"发现同名Ledger v1候选，拒绝覆盖：{existing}")


def _later_fi_counts(
    index: pd.DataFrame, fi: pd.DataFrame, spine: pd.DataFrame, cohort: str
) -> pd.Series:
    landmark_order = spine[["cohort", "person_id", "wave", "wave_order"]].rename(
        columns={"wave": "landmark_wave", "wave_order": "landmark_wave_order"}
    )
    anchor = index[["cohort", "person_id", "landmark_wave"]].merge(
        landmark_order,
        on=["cohort", "person_id", "landmark_wave"],
        how="left",
        validate="one_to_one",
    )
    if anchor["landmark_wave_order"].isna().any():
        raise RuntimeError(f"{cohort}: landmark wave无法映射到Full Spine")
    future = spine[["cohort", "person_id", "wave", "wave_order"]].merge(
        fi,
        on=["cohort", "person_id", "wave"],
        how="left",
        validate="one_to_one",
    )
    future = future.merge(
        anchor[["cohort", "person_id", "landmark_wave_order"]],
        on=["cohort", "person_id"],
        how="inner",
        validate="many_to_one",
    )
    future = future.loc[future["wave_order"] > future["landmark_wave_order"]]
    counts = (
        future["fi_primary"]
        .notna()
        .groupby([future["cohort"], future["person_id"]])
        .sum()
        .astype("Int16")
    )
    counts.name = "later_fi_count"
    return counts


def _missing_adjustments(row: pd.Series) -> object:
    missing = [name for name in ADJUSTMENT_VARIABLES if pd.isna(row[name])]
    return ";".join(missing) if missing else pd.NA


def _first_failed_gate(frame: pd.DataFrame) -> pd.Series:
    reason = pd.Series("none", index=frame.index, dtype="string")
    reason.loc[~frame["recovery_classifiable"]] = "recovery_unavailable"
    reason.loc[
        frame["recovery_classifiable"] & ~frame["landmark_fi_available"]
    ] = "landmark_fi_missing"
    reason.loc[
        frame["recovery_classifiable"]
        & frame["landmark_fi_available"]
        & frame["later_fi_count"].eq(0)
    ] = "no_later_fi"
    longitudinal = frame["longitudinal_support_complete"]
    reason.loc[longitudinal & ~frame["fi_t_available"]] = "fi_t_missing"
    reason.loc[
        longitudinal
        & frame["fi_t_available"]
        & ~frame["nine_covariates_complete"]
    ] = "prespecified_covariate_missing"
    return reason


def _cast_ledger(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.loc[:, LEDGER_COLUMNS].copy()
    for column, dtype in LEDGER_DTYPES.items():
        output[column] = output[column].astype(dtype)
    return output


def validate_ledger(frame: pd.DataFrame, cohort: str) -> dict[str, Any]:
    if list(frame.columns) != list(LEDGER_COLUMNS):
        raise RuntimeError(f"{cohort}: Ledger列顺序不符合18列合同")
    for column, expected_dtype in LEDGER_DTYPES.items():
        if str(frame[column].dtype).lower() != expected_dtype.lower():
            raise RuntimeError(
                f"{cohort}/{column}: dtype={frame[column].dtype}，预期{expected_dtype}"
            )
    if frame.duplicated(["cohort", "person_id"]).any():
        raise RuntimeError(f"{cohort}: Ledger主键重复")
    if frame[["cohort", "person_id"]].isna().any().any():
        raise RuntimeError(f"{cohort}: Ledger主键缺失")
    for column in set(LEDGER_COLUMNS) - NULLABLE_COLUMNS:
        if frame[column].isna().any():
            raise RuntimeError(f"{cohort}/{column}: 非空列出现缺失")
    if not set(frame["recovery_status"].dropna().unique()).issubset(RECOVERY_VALUES):
        raise RuntimeError(f"{cohort}: recovery_status出现未知枚举")
    if set(frame["landmark_weight_status"].unique()) - WEIGHT_STATUS_VALUES:
        raise RuntimeError(f"{cohort}: landmark_weight_status出现未知枚举")
    if set(frame["first_failed_gate"].unique()) - FIRST_FAILED_GATE_VALUES:
        raise RuntimeError(f"{cohort}: first_failed_gate出现未知枚举")
    if not frame["recovery_classifiable"].eq(frame["recovery_status"].notna()).all():
        raise RuntimeError(f"{cohort}: recovery_classifiable与status不一致")
    if (frame["later_fi_count"] < 0).any():
        raise RuntimeError(f"{cohort}: later_fi_count出现负数")
    missing_names = frame["missing_adjustment_variables"].fillna("").map(
        lambda value: [] if value == "" else value.split(";")
    )
    if not missing_names.map(lambda values: set(values).issubset(ADJUSTMENT_VARIABLES)).all():
        raise RuntimeError(f"{cohort}: missing_adjustment_variables出现未知名称")
    if not frame["missing_adjustment_variable_count"].eq(
        missing_names.map(len).astype("Int8")
    ).all():
        raise RuntimeError(f"{cohort}: 调整变量缺失计数与列表不一致")
    if (frame["missing_adjustment_variable_count"] > 10).any():
        raise RuntimeError(f"{cohort}: 调整变量缺失计数超过10")

    expected_longitudinal = (
        frame["recovery_classifiable"]
        & frame["landmark_fi_available"]
        & frame["later_fi_count"].ge(1)
    )
    expected_complete = (
        expected_longitudinal
        & frame["fi_t_available"]
        & frame["nine_covariates_complete"]
    )
    expected_weighted = expected_complete & frame["landmark_weight_status"].eq(
        "positive"
    )
    if not frame["longitudinal_support_complete"].eq(expected_longitudinal).all():
        raise RuntimeError(f"{cohort}: longitudinal support逻辑不一致")
    if not frame["complete_case_support_complete"].eq(expected_complete).all():
        raise RuntimeError(f"{cohort}: complete-case support逻辑不一致")
    if not frame["weighted_complete_case_support_complete"].eq(expected_weighted).all():
        raise RuntimeError(f"{cohort}: weighted support逻辑不一致")
    if not frame["first_failed_gate"].eq(_first_failed_gate(frame)).all():
        raise RuntimeError(f"{cohort}: first_failed_gate优先级不一致")

    flow = {
        "shock_experienced_cohort": len(frame),
        "longitudinal_support_complete": int(
            frame["longitudinal_support_complete"].sum()
        ),
        "complete_case_support_complete": int(
            frame["complete_case_support_complete"].sum()
        ),
        "weighted_complete_case_support_complete": int(
            frame["weighted_complete_case_support_complete"].sum()
        ),
    }
    first_failed = {
        reason: int(frame["first_failed_gate"].eq(reason).sum())
        for reason in (
            "recovery_unavailable",
            "landmark_fi_missing",
            "no_later_fi",
            "fi_t_missing",
            "prespecified_covariate_missing",
            "none",
        )
    }
    if flow != EXPECTED_FLOW[cohort]:
        raise RuntimeError(f"{cohort}: 支持人数未复现：{flow} != {EXPECTED_FLOW[cohort]}")
    if first_failed != EXPECTED_FIRST_FAILED[cohort]:
        raise RuntimeError(
            f"{cohort}: 排除原因未复现：{first_failed} != {EXPECTED_FIRST_FAILED[cohort]}"
        )
    if sum(first_failed.values()) != len(frame):
        raise RuntimeError(f"{cohort}: first_failed_gate不守恒")
    if not (
        flow["weighted_complete_case_support_complete"]
        <= flow["complete_case_support_complete"]
        <= flow["longitudinal_support_complete"]
        <= flow["shock_experienced_cohort"]
    ):
        raise RuntimeError(f"{cohort}: 支持层级不嵌套")
    return {"flow": flow, "first_failed_gate": first_failed}


def build_ledger(cohort: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    exposure = _load_frozen_panel(
        "exposure_panel_v1",
        cohort,
        [
            "cohort",
            "person_id",
            "tminus1_wave",
            "candidate_shock_wave",
            "tplus1_wave",
            "wealth_loss_fraction",
            "major_shock",
            "is_index_shock",
            "recovery_status",
        ],
    )
    fi = _load_frozen_panel(
        "fi_panel_v2", cohort, ["cohort", "person_id", "wave", "fi_primary"]
    )
    covariate = _load_frozen_panel("covariate_panel_v1", cohort)
    weight = _load_frozen_panel(
        "weight_design_panel_v1",
        cohort,
        ["cohort", "person_id", "wave", "person_weight_raw"],
    )
    spine = _load_frozen_panel(
        "spine_v2", cohort, ["cohort", "person_id", "wave", "wave_order"]
    )

    index = exposure.loc[exposure["is_index_shock"].fillna(False)].copy()
    if index.duplicated(["cohort", "person_id"]).any():
        raise RuntimeError(f"{cohort}: 每人最多只能有一个index shock")
    loss = pd.to_numeric(index["wealth_loss_fraction"], errors="coerce")
    if (
        not index["major_shock"].eq(True).all()
        or not np.isfinite(loss).all()
        or not loss.ge(0.75).all()
    ):
        raise RuntimeError(f"{cohort}: index shock不满足有限L>=0.75及major_shock合同")
    if index["tplus1_wave"].isna().any():
        raise RuntimeError(f"{cohort}: index shock缺少landmark wave")

    ledger = index[
        [
            "cohort",
            "person_id",
            "tminus1_wave",
            "candidate_shock_wave",
            "tplus1_wave",
            "recovery_status",
        ]
    ].rename(
        columns={
            "candidate_shock_wave": "index_shock_wave",
            "tplus1_wave": "landmark_wave",
        }
    )
    ledger["recovery_status"] = ledger["recovery_status"].where(
        ledger["recovery_status"].isin(RECOVERY_VALUES), pd.NA
    )
    ledger["recovery_classifiable"] = ledger["recovery_status"].notna()

    fi_t = fi.rename(columns={"wave": "index_shock_wave", "fi_primary": "fi_t"})
    fi_landmark = fi.rename(
        columns={"wave": "landmark_wave", "fi_primary": "fi_landmark"}
    )
    ledger = ledger.merge(
        fi_t,
        on=["cohort", "person_id", "index_shock_wave"],
        how="left",
        validate="one_to_one",
    ).merge(
        fi_landmark,
        on=["cohort", "person_id", "landmark_wave"],
        how="left",
        validate="one_to_one",
    )
    ledger["fi_t_available"] = ledger["fi_t"].notna()
    ledger["landmark_fi_available"] = ledger["fi_landmark"].notna()

    later_counts = _later_fi_counts(ledger, fi, spine, cohort)
    ledger = ledger.merge(
        later_counts.reset_index(),
        on=["cohort", "person_id"],
        how="left",
        validate="one_to_one",
    )
    ledger["later_fi_count"] = ledger["later_fi_count"].fillna(0).astype("Int16")

    cov_t = covariate[["cohort", "person_id", "wave", *T_COVARIATES]].rename(
        columns={"wave": "index_shock_wave"}
    )
    cov_tminus1 = covariate[
        ["cohort", "person_id", "wave", "wealth_quintile"]
    ].rename(columns={"wave": "tminus1_wave"})
    ledger = ledger.merge(
        cov_t,
        on=["cohort", "person_id", "index_shock_wave"],
        how="left",
        validate="one_to_one",
    ).merge(
        cov_tminus1,
        on=["cohort", "person_id", "tminus1_wave"],
        how="left",
        validate="one_to_one",
    )
    nine_covariates = [*T_COVARIATES, "wealth_quintile"]
    ledger["nine_covariates_complete"] = ledger[nine_covariates].notna().all(axis=1)
    ledger["missing_adjustment_variables"] = ledger.apply(
        _missing_adjustments, axis=1
    )
    ledger["missing_adjustment_variable_count"] = (
        ledger["missing_adjustment_variables"]
        .fillna("")
        .map(lambda value: 0 if value == "" else len(value.split(";")))
        .astype("Int8")
    )

    landmark_weight = weight.rename(
        columns={"wave": "landmark_wave", "person_weight_raw": "landmark_weight"}
    )
    ledger = ledger.merge(
        landmark_weight,
        on=["cohort", "person_id", "landmark_wave"],
        how="left",
        validate="one_to_one",
    )
    if (ledger["landmark_weight"].dropna() < 0).any():
        raise RuntimeError(f"{cohort}: exact-t+1出现负权重")
    ledger["landmark_weight_status"] = pd.Series(
        np.select(
            [
                ledger["landmark_weight"].isna(),
                ledger["landmark_weight"].eq(0),
                ledger["landmark_weight"].gt(0),
            ],
            ["missing", "zero", "positive"],
            default="invalid",
        ),
        index=ledger.index,
        dtype="string",
    )
    if ledger["landmark_weight_status"].eq("invalid").any():
        raise RuntimeError(f"{cohort}: landmark weight状态无法分类")

    ledger["longitudinal_support_complete"] = (
        ledger["recovery_classifiable"]
        & ledger["landmark_fi_available"]
        & ledger["later_fi_count"].ge(1)
    )
    ledger["complete_case_support_complete"] = (
        ledger["longitudinal_support_complete"]
        & ledger["fi_t_available"]
        & ledger["nine_covariates_complete"]
    )
    ledger["weighted_complete_case_support_complete"] = (
        ledger["complete_case_support_complete"]
        & ledger["landmark_weight_status"].eq("positive")
    )
    ledger["first_failed_gate"] = _first_failed_gate(ledger)

    output = _cast_ledger(ledger)
    validation = validate_ledger(output, cohort)
    return output, validation


def _recovery_counts(frame: pd.DataFrame) -> dict[str, dict[str, int]]:
    stages = {
        "shock_experienced_cohort": pd.Series(True, index=frame.index),
        "longitudinal_support_complete": frame["longitudinal_support_complete"],
        "complete_case_support_complete": frame["complete_case_support_complete"],
    }
    result: dict[str, dict[str, int]] = {}
    for stage, mask in stages.items():
        subset = frame.loc[mask]
        counts = {
            status: int(subset["recovery_status"].eq(status).sum())
            for status in RECOVERY_VALUES
        }
        if stage == "shock_experienced_cohort":
            counts["unavailable"] = int(subset["recovery_status"].isna().sum())
        result[stage] = counts
    return result


def build_review_summary(panels: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    def add(
        section: str,
        cohort: str,
        stage_or_reason: str,
        recovery_status: str,
        n: int,
        denominator: int,
    ) -> None:
        rows.append(
            {
                "section": section,
                "cohort": cohort,
                "stage_or_reason": stage_or_reason,
                "recovery_status": recovery_status,
                "n": n,
                "denominator": denominator,
                "percent": 100.0 * n / denominator if denominator else math.nan,
            }
        )

    for cohort in COHORTS:
        frame = panels[cohort]
        flow = EXPECTED_FLOW[cohort]
        for stage in (
            "shock_experienced_cohort",
            "longitudinal_support_complete",
            "complete_case_support_complete",
            "weighted_complete_case_support_complete",
        ):
            add("nested_flow", cohort, stage, "ALL", flow[stage], len(frame))

        for reason, n in EXPECTED_FIRST_FAILED[cohort].items():
            add("first_failed_gate", cohort, reason, "ALL", n, len(frame))

        recovery_counts = _recovery_counts(frame)
        if recovery_counts != EXPECTED_RECOVERY_COUNTS[cohort]:
            raise RuntimeError(
                f"{cohort}: recovery分组人数未复现：{recovery_counts} != "
                f"{EXPECTED_RECOVERY_COUNTS[cohort]}"
            )
        for stage, counts in recovery_counts.items():
            denominator = int(
                len(frame)
                if stage == "shock_experienced_cohort"
                else frame[stage].sum()
            )
            for recovery_status, n in counts.items():
                add(
                    "recovery_group_counts",
                    cohort,
                    stage,
                    recovery_status,
                    n,
                    denominator,
                )

        complete = frame.loc[frame["complete_case_support_complete"]]
        for status in ("positive", "zero", "missing"):
            n = int(complete["landmark_weight_status"].eq(status).sum())
            add(
                "complete_case_landmark_weight_status",
                cohort,
                status,
                "ALL",
                n,
                len(complete),
            )

    summary = pd.DataFrame(rows)
    if "person_id" in summary.columns:
        raise RuntimeError("Review summary不得包含person_id")
    return summary


def _schema_contract(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return [
        {
            "name": column,
            "dtype": str(frame[column].dtype),
            "nullable": column in NULLABLE_COLUMNS,
        }
        for column in LEDGER_COLUMNS
    ]


def git_provenance() -> dict[str, Any]:
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        capture_output=True,
        check=True,
    ).stdout.decode("ascii").strip()
    diff = subprocess.run(
        ["git", "diff", "--binary", "HEAD"],
        cwd=ROOT,
        capture_output=True,
        check=True,
    ).stdout
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=ROOT,
        capture_output=True,
        check=True,
    ).stdout.decode("utf-8", errors="replace").splitlines()
    return {
        "git_commit": head,
        "git_dirty": bool(status),
        "git_diff_sha256": hashlib.sha256(diff).hexdigest(),
        "git_status_entry_count": len(status),
    }


def _build_manifest(
    cohort: str,
    panel: pd.DataFrame,
    staged_output: Path,
    validation: dict[str, Any],
    upstream: dict[str, Any],
    design_sha256: str,
    provenance: dict[str, Any],
) -> dict[str, Any]:
    recovery_counts = _recovery_counts(panel)
    complete = panel.loc[panel["complete_case_support_complete"]]
    return {
        "manifest_schema_version": "analysis_cohort_ledger_manifest_v1",
        "release_status": "READY_FOR_REVIEW",
        "release_version": RELEASE_VERSION,
        "generated_utc": datetime.now(UTC).isoformat(),
        "cohort": cohort,
        "builder_version": BUILDER_VERSION,
        "builder_sha256": sha256_file(Path(__file__)),
        "runner_version": RUNNER_VERSION,
        **provenance,
        "design_register": {
            "path": _relative(DESIGN_REGISTER_PATH),
            "sha256": design_sha256,
            "modified_by_this_task": False,
        },
        "author_approved_analysis_hierarchy": {
            "primary": "unweighted complete-case analysis",
            "sensitivities": [
                "exact-tplus1 landmark-weighted analysis",
                "covariate multiple-imputation analysis",
            ],
            "design_register_sync": "pending before Analysis Panel",
        },
        "row_unit": "one participant with one analytic index shock",
        "common_key": ["cohort", "person_id"],
        "physical_columns": list(LEDGER_COLUMNS),
        "schema": _schema_contract(panel),
        "row_count": len(panel),
        "unique_key": not panel.duplicated(["cohort", "person_id"]).any(),
        "flow": validation["flow"],
        "first_failed_gate": validation["first_failed_gate"],
        "recovery_group_counts": recovery_counts,
        "complete_case_landmark_weight_status": {
            status: int(complete["landmark_weight_status"].eq(status).sum())
            for status in ("positive", "zero", "missing")
        },
        "output": {
            "path": _relative(_final_output_path(cohort)),
            "sha256": sha256_file(staged_output),
            "compression": "zstd",
        },
        "upstream_frozen_inputs": {
            panel_name: snapshot["cohorts"][cohort]
            for panel_name, snapshot in upstream.items()
        },
        "invariants": {
            "all_rows_are_analytic_index_shocks": True,
            "finite_wealth_loss_fraction_ge_0_75_verified_upstream": True,
            "wealth_loss_fraction_may_exceed_one": True,
            "later_fi_is_strictly_after_landmark": True,
            "exact_tplus1_weight_without_fallback": True,
            "support_layers_nested": True,
            "first_failed_gate_mutually_exclusive_and_conserved": True,
            "no_raw_fi_covariate_wealth_or_weight_values_stored": True,
            "final_interview_timing_not_applied": True,
        },
        "interpretation": {
            "complete_case_support_is_candidate": True,
            "weighted_support_is_sensitivity_branch_not_fourth_layer": True,
            "no_later_fi_is_operational_not_causal": True,
            "none_does_not_imply_positive_weight_or_final_model_eligibility": True,
        },
    }


def _build_release_candidate(
    staged_outputs: dict[str, Path],
    staged_manifests: dict[str, Path],
    staged_review: Path,
    staged_dictionary: Path,
    validations: dict[str, dict[str, Any]],
    upstream: dict[str, Any],
    design_sha256: str,
    provenance: dict[str, Any],
) -> dict[str, Any]:
    return {
        "release_candidate_schema_version": "analysis_cohort_ledger_release_candidate_v1",
        "release_status": "READY_FOR_REVIEW",
        "release_version": RELEASE_VERSION,
        "generated_utc": datetime.now(UTC).isoformat(),
        "candidate_label": "ANALYSIS COHORT LEDGER V1 READY FOR REVIEW",
        "formal_freeze_created": False,
        "builder_version": BUILDER_VERSION,
        "builder_sha256": sha256_file(Path(__file__)),
        "runner_version": RUNNER_VERSION,
        **provenance,
        "design_register": {
            "path": _relative(DESIGN_REGISTER_PATH),
            "sha256": design_sha256,
            "modified_by_this_task": False,
            "sync_required_before_analysis_panel": True,
        },
        "row_unit": "one participant with one analytic index shock",
        "common_key": ["cohort", "person_id"],
        "physical_columns": list(LEDGER_COLUMNS),
        "cohorts": {
            cohort: {
                "parquet_path": _relative(_final_output_path(cohort)),
                "parquet_sha256": sha256_file(staged_outputs[cohort]),
                "manifest_path": _relative(_manifest_path(cohort)),
                "manifest_sha256": sha256_file(staged_manifests[cohort]),
                "row_count": EXPECTED_FLOW[cohort]["shock_experienced_cohort"],
                **validations[cohort],
            }
            for cohort in COHORTS
        },
        "review_summary": {
            "path": _relative(REVIEW_PATH),
            "sha256": sha256_file(staged_review),
            "contains_participant_ids": False,
        },
        "data_dictionary": {
            "path": _relative(DICTIONARY_PATH),
            "sha256": sha256_file(staged_dictionary),
        },
        "upstream_frozen_snapshot": upstream,
        "release_integrity": {
            "upstream_before_after_must_match": True,
            "design_register_before_after_must_match": True,
            "three_cohorts_publish_as_one_transaction": True,
            "release_reload_required": True,
            "same_known_repository_failure_only": (
                "test_supplementary_table_s3_reflects_v2_boundary"
            ),
        },
        "scope_boundaries": {
            "analysis_panel_built": False,
            "eligibility_final_built": False,
            "model_run": False,
            "design_register_updated": False,
            "manuscript_or_supplement_modified": False,
        },
        "next_step": (
            "Human review of the summary counts; unchanged candidate hashes may then "
            "be referenced by a separate freeze approval."
        ),
    }


def _publish_candidate_transaction(staged_by_final: list[tuple[Path, Path]]) -> None:
    published: list[Path] = []
    try:
        for staged, final in staged_by_final:
            if final.exists():
                raise RuntimeError(f"发布目标已存在，拒绝覆盖：{final}")
            final.parent.mkdir(parents=True, exist_ok=True)
            os.replace(staged, final)
            published.append(final)
    except Exception:
        for path in reversed(published):
            path.unlink(missing_ok=True)
        raise


def run_analysis_cohort_ledger_release(cohorts: list[str]) -> dict[str, Any]:
    """一次性构建三队列Analysis Cohort Ledger v1候选。"""
    if tuple(cohorts) != COHORTS:
        raise RuntimeError("landmark-ledger v1必须使用--cohort all共同构建")
    _verify_release_absent()
    upstream_before = protected_upstream_snapshot()
    design_before = sha256_file(DESIGN_REGISTER_PATH)
    provenance = git_provenance()

    panels: dict[str, pd.DataFrame] = {}
    validations: dict[str, dict[str, Any]] = {}
    for cohort in COHORTS:
        panels[cohort], validations[cohort] = build_ledger(cohort)
    review = build_review_summary(panels)

    staging_parent = ROOT / "tmp"
    staging_parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="analysis_cohort_ledger_v1_", dir=staging_parent
    ) as tmp_name:
        staging = Path(tmp_name)
        staged_outputs: dict[str, Path] = {}
        for cohort in COHORTS:
            staged_output = staging / _final_output_path(cohort).name
            panels[cohort].to_parquet(staged_output, index=False, compression="zstd")
            reloaded = pd.read_parquet(staged_output)
            validate_ledger(reloaded, cohort)
            pd.testing.assert_frame_equal(
                reloaded, panels[cohort], check_dtype=True, check_exact=True
            )
            staged_outputs[cohort] = staged_output

        staged_review = staging / REVIEW_PATH.name
        review.to_csv(staged_review, index=False, encoding="utf-8-sig")
        reloaded_review = pd.read_csv(staged_review)
        pd.testing.assert_frame_equal(
            reloaded_review,
            review,
            check_dtype=False,
            check_exact=False,
            rtol=1e-12,
            atol=1e-12,
        )
        if "person_id" in reloaded_review.columns:
            raise RuntimeError("Review CSV不得包含person_id")

        staged_dictionary = staging / DICTIONARY_PATH.name
        staged_dictionary.write_text(DICTIONARY_TEXT, encoding="utf-8")

        staged_manifests: dict[str, Path] = {}
        for cohort in COHORTS:
            manifest = _build_manifest(
                cohort,
                panels[cohort],
                staged_outputs[cohort],
                validations[cohort],
                upstream_before,
                design_before,
                provenance,
            )
            staged_manifest = staging / _manifest_path(cohort).name
            _json_write(staged_manifest, manifest)
            staged_manifests[cohort] = staged_manifest

        upstream_after = protected_upstream_snapshot()
        design_after = sha256_file(DESIGN_REGISTER_PATH)
        if upstream_before != upstream_after:
            raise RuntimeError("Ledger构建期间冻结上游发生变化")
        if design_before != design_after:
            raise RuntimeError("Ledger构建期间design register发生变化")

        staged_candidate = staging / RELEASE_CANDIDATE_PATH.name
        _json_write(
            staged_candidate,
            _build_release_candidate(
                staged_outputs,
                staged_manifests,
                staged_review,
                staged_dictionary,
                validations,
                upstream_before,
                design_before,
                provenance,
            ),
        )
        staged_by_final = [
            *((staged_outputs[cohort], _final_output_path(cohort)) for cohort in COHORTS),
            (staged_review, REVIEW_PATH),
            (staged_dictionary, DICTIONARY_PATH),
            *((staged_manifests[cohort], _manifest_path(cohort)) for cohort in COHORTS),
            (staged_candidate, RELEASE_CANDIDATE_PATH),
        ]
        _publish_candidate_transaction(staged_by_final)

    # 发布后重新载入并核对候选资产，避免只验证staging副本。
    release = json.loads(RELEASE_CANDIDATE_PATH.read_text(encoding="utf-8"))
    for cohort in COHORTS:
        record = release["cohorts"][cohort]
        if sha256_file(_final_output_path(cohort)) != record["parquet_sha256"]:
            raise RuntimeError(f"{cohort}: 发布后Parquet哈希不一致")
        if sha256_file(_manifest_path(cohort)) != record["manifest_sha256"]:
            raise RuntimeError(f"{cohort}: 发布后manifest哈希不一致")
        validate_ledger(pd.read_parquet(_final_output_path(cohort)), cohort)
    if sha256_file(REVIEW_PATH) != release["review_summary"]["sha256"]:
        raise RuntimeError("发布后review CSV哈希不一致")
    if sha256_file(DICTIONARY_PATH) != release["data_dictionary"]["sha256"]:
        raise RuntimeError("发布后数据字典哈希不一致")
    if protected_upstream_snapshot() != upstream_before:
        raise RuntimeError("发布后冻结上游发生变化")
    if sha256_file(DESIGN_REGISTER_PATH) != design_before:
        raise RuntimeError("发布后design register发生变化")

    return {
        "release_status": "READY_FOR_REVIEW",
        "candidate_label": "ANALYSIS COHORT LEDGER V1 READY FOR REVIEW",
        "release_candidate_path": str(RELEASE_CANDIDATE_PATH),
        "review_summary_path": str(REVIEW_PATH),
        "formal_freeze_created": False,
        "cohorts": {
            cohort: validations[cohort]["flow"] for cohort in COHORTS
        },
    }


def verify_analysis_cohort_ledger_candidate_for_freeze() -> dict[str, Any]:
    """冻结前重载候选及全部引用资产，不修改候选文件。"""
    if FORMAL_FREEZE_PATH.exists():
        raise RuntimeError("Analysis Cohort Ledger v1 freeze已存在，拒绝重复冻结")
    if not RELEASE_CANDIDATE_PATH.is_file():
        raise RuntimeError("缺少Analysis Cohort Ledger v1 release candidate")
    candidate_sha256 = sha256_file(RELEASE_CANDIDATE_PATH)
    if candidate_sha256 != EXPECTED_APPROVED_CANDIDATE_SHA256:
        raise RuntimeError(
            "Analysis Cohort Ledger候选哈希漂移："
            f"{candidate_sha256} != {EXPECTED_APPROVED_CANDIDATE_SHA256}"
        )
    release = json.loads(RELEASE_CANDIDATE_PATH.read_text(encoding="utf-8"))
    if release.get("release_status") != "READY_FOR_REVIEW":
        raise RuntimeError("Ledger候选状态不是READY_FOR_REVIEW")
    if release.get("formal_freeze_created") is not False:
        raise RuntimeError("Ledger候选的formal_freeze_created必须为false")
    if release.get("physical_columns") != list(LEDGER_COLUMNS):
        raise RuntimeError("Ledger候选18列合同漂移")
    if release["design_register"]["sha256"] != sha256_file(DESIGN_REGISTER_PATH):
        raise RuntimeError("Ledger候选记录的design register哈希漂移")

    upstream = protected_upstream_snapshot()
    if release.get("upstream_frozen_snapshot") != upstream:
        raise RuntimeError("Ledger候选记录的冻结上游发生漂移")

    review_record = release["review_summary"]
    if sha256_file(ROOT / review_record["path"]) != review_record["sha256"]:
        raise RuntimeError("Ledger review CSV哈希漂移")
    if review_record.get("contains_participant_ids") is not False:
        raise RuntimeError("Ledger review CSV隐私状态不符合合同")
    dictionary_record = release["data_dictionary"]
    if sha256_file(ROOT / dictionary_record["path"]) != dictionary_record["sha256"]:
        raise RuntimeError("Ledger数据字典哈希漂移")

    panels: dict[str, pd.DataFrame] = {}
    for cohort in COHORTS:
        record = release["cohorts"][cohort]
        parquet_path = ROOT / record["parquet_path"]
        manifest_path = ROOT / record["manifest_path"]
        if sha256_file(parquet_path) != record["parquet_sha256"]:
            raise RuntimeError(f"{cohort}: Ledger Parquet哈希漂移")
        if sha256_file(manifest_path) != record["manifest_sha256"]:
            raise RuntimeError(f"{cohort}: Ledger manifest哈希漂移")
        panel = pd.read_parquet(parquet_path)
        validation = validate_ledger(panel, cohort)
        if validation != {
            "flow": record["flow"],
            "first_failed_gate": record["first_failed_gate"],
        }:
            raise RuntimeError(f"{cohort}: Ledger内容与release candidate不一致")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("flow") != record["flow"]:
            raise RuntimeError(f"{cohort}: cohort manifest flow不一致")
        if manifest.get("first_failed_gate") != record["first_failed_gate"]:
            raise RuntimeError(f"{cohort}: cohort manifest gate不一致")
        if manifest.get("recovery_group_counts") != _recovery_counts(panel):
            raise RuntimeError(f"{cohort}: cohort manifest recovery人数不一致")
        panels[cohort] = panel

    expected_review = build_review_summary(panels)
    observed_review = pd.read_csv(ROOT / review_record["path"], encoding="utf-8-sig")
    pd.testing.assert_frame_equal(
        observed_review,
        expected_review,
        check_dtype=False,
        check_exact=False,
        rtol=1e-12,
        atol=1e-12,
    )
    if "person_id" in observed_review.columns:
        raise RuntimeError("Ledger review CSV不得包含person_id")
    return release


def verify_analysis_cohort_ledger_freeze() -> dict[str, Any]:
    """验证正式freeze及其全部引用，不把支持事实解释为最终资格。"""
    if not FORMAL_FREEZE_PATH.is_file():
        raise RuntimeError("Analysis Cohort Ledger v1 freeze不存在")
    freeze = json.loads(FORMAL_FREEZE_PATH.read_text(encoding="utf-8"))
    if freeze.get("freeze_status") != "FROZEN":
        raise RuntimeError("Ledger freeze_status不是FROZEN")
    if freeze.get("support_ledger_status") != "SUPPORT_FACTS_ONLY":
        raise RuntimeError("Ledger support_ledger_status不符合合同")
    if freeze.get("final_model_eligibility_status") != "NOT_LOCKED":
        raise RuntimeError("Ledger不得提前锁定最终模型资格")
    source = freeze["source_release_candidate"]
    if source.get("path") != _relative(RELEASE_CANDIDATE_PATH):
        raise RuntimeError("Ledger freeze引用的候选路径不一致")
    if source.get("sha256") != EXPECTED_APPROVED_CANDIDATE_SHA256:
        raise RuntimeError("Ledger freeze引用的获准候选哈希不一致")
    if sha256_file(RELEASE_CANDIDATE_PATH) != source["sha256"]:
        raise RuntimeError("Ledger freeze之后候选发生漂移")

    release = json.loads(RELEASE_CANDIDATE_PATH.read_text(encoding="utf-8"))
    if freeze.get("cohorts") != release.get("cohorts"):
        raise RuntimeError("Ledger freeze与候选的三队列记录不一致")
    if freeze.get("review_summary") != release.get("review_summary"):
        raise RuntimeError("Ledger freeze与候选的review记录不一致")
    if freeze.get("data_dictionary") != release.get("data_dictionary"):
        raise RuntimeError("Ledger freeze与候选的数据字典记录不一致")
    if freeze.get("upstream_frozen_snapshot") != protected_upstream_snapshot():
        raise RuntimeError("Ledger freeze记录的上游冻结资产发生漂移")

    for cohort in COHORTS:
        record = freeze["cohorts"][cohort]
        if sha256_file(ROOT / record["parquet_path"]) != record["parquet_sha256"]:
            raise RuntimeError(f"{cohort}: frozen Ledger Parquet哈希漂移")
        if sha256_file(ROOT / record["manifest_path"]) != record["manifest_sha256"]:
            raise RuntimeError(f"{cohort}: frozen Ledger manifest哈希漂移")
    if sha256_file(ROOT / freeze["review_summary"]["path"]) != freeze[
        "review_summary"
    ]["sha256"]:
        raise RuntimeError("Frozen Ledger review CSV哈希漂移")
    if sha256_file(ROOT / freeze["data_dictionary"]["path"]) != freeze[
        "data_dictionary"
    ]["sha256"]:
        raise RuntimeError("Frozen Ledger数据字典哈希漂移")
    return freeze


def freeze_analysis_cohort_ledger_v1(
    *, pre_freeze_validation: dict[str, Any]
) -> dict[str, Any]:
    """按作者批准的现有候选哈希建立一次性support-facts freeze。"""
    release = verify_analysis_cohort_ledger_candidate_for_freeze()
    provenance = git_provenance()
    freeze = {
        "freeze_record_schema_version": "analysis_cohort_ledger_freeze_record_v1",
        "freeze_status": "FROZEN",
        "support_ledger_status": "SUPPORT_FACTS_ONLY",
        "final_model_eligibility_status": "NOT_LOCKED",
        "release_version": RELEASE_VERSION,
        "generated_utc": datetime.now(UTC).isoformat(),
        "source_release_candidate": {
            "path": _relative(RELEASE_CANDIDATE_PATH),
            "sha256": EXPECTED_APPROVED_CANDIDATE_SHA256,
            "release_status_at_build": release["release_status"],
            "candidate_label": release["candidate_label"],
        },
        **provenance,
        "candidate_builder_version": release["builder_version"],
        "candidate_builder_sha256": release["builder_sha256"],
        "post_review_builder_version": BUILDER_VERSION,
        "post_review_builder_sha256": sha256_file(Path(__file__)),
        "runner_version": RUNNER_VERSION,
        "row_unit": release["row_unit"],
        "common_key": release["common_key"],
        "physical_columns": release["physical_columns"],
        "cohorts": release["cohorts"],
        "review_summary": release["review_summary"],
        "data_dictionary": release["data_dictionary"],
        "upstream_frozen_snapshot": release["upstream_frozen_snapshot"],
        "human_review": {
            "authorization": (
                "User approved Analysis Cohort Ledger v1 after review of the "
                "three support layers, mutually exclusive gates, recovery groups, "
                "and landmark-weight support branch."
            ),
            "review_outcome": (
                "Support facts approved for freezing; final model eligibility was "
                "explicitly not locked."
            ),
        },
        "frozen_support_facts": [
            "analytic index-shock membership",
            "recovery classification availability",
            "landmark and later FI measurement support",
            "FI_t and nine-covariate completeness support",
            "exact-tplus1 landmark-weight status",
            "mutually exclusive first failed gate",
        ],
        "interpretation_boundary": {
            "complete_case_support_remains_candidate_for_final_model": True,
            "positive_landmark_weight_is_sensitivity_support_not_primary_eligibility": True,
            "actual_record_level_model_input_validation_pending": True,
            "eligibility_final_built": False,
            "analysis_panel_built": False,
            "real_data_models_run": False,
        },
        "release_integrity": {
            "candidate_exact_author_approved_sha256_verified": True,
            "all_parquet_manifest_dictionary_and_review_hashes_verified": True,
            "all_support_counts_and_gate_conservation_recomputed": True,
            "protected_upstream_freezes_verified": True,
            "parquet_rebuilt_for_freeze": False,
            "cohort_manifest_rebuilt_for_freeze": False,
            "review_csv_rebuilt_for_freeze": False,
            "downstream_assets_created_by_freeze": False,
        },
        "pre_freeze_validation": pre_freeze_validation,
        "overwrite_policy": (
            "This v1 support ledger and all hash-bound assets are immutable after "
            "freezing; any schema, gate-order, definition, or count change requires "
            "a new version."
        ),
        "next_stage_status": (
            "FIGURE_S1_AUTHOR_REVIEW_ALLOWED; FINAL_MODEL_ELIGIBILITY_NOT_LOCKED"
        ),
    }

    FORMAL_FREEZE_PATH.parent.mkdir(parents=True, exist_ok=True)
    if FORMAL_FREEZE_PATH.exists():
        raise RuntimeError("Analysis Cohort Ledger v1 freeze已存在，拒绝覆盖")
    with tempfile.TemporaryDirectory(
        prefix="analysis_cohort_ledger_freeze_", dir=ROOT / "tmp"
    ) as tmp_name:
        staged = Path(tmp_name) / FORMAL_FREEZE_PATH.name
        _json_write(staged, freeze)
        os.replace(staged, FORMAL_FREEZE_PATH)
    return verify_analysis_cohort_ledger_freeze()


__all__ = [
    "ADJUSTMENT_VARIABLES",
    "COHORTS",
    "EXPECTED_FIRST_FAILED",
    "EXPECTED_FLOW",
    "EXPECTED_RECOVERY_COUNTS",
    "EXPECTED_APPROVED_CANDIDATE_SHA256",
    "FORMAL_FREEZE_PATH",
    "LEDGER_COLUMNS",
    "LEDGER_DTYPES",
    "RELEASE_CANDIDATE_PATH",
    "build_ledger",
    "build_review_summary",
    "freeze_analysis_cohort_ledger_v1",
    "run_analysis_cohort_ledger_release",
    "validate_ledger",
    "verify_analysis_cohort_ledger_candidate_for_freeze",
    "verify_analysis_cohort_ledger_freeze",
]
