"""三队列 raw-only panel 的阶段 runner。"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from panel_core import (
    CANDIDATE_INTERVAL_COLUMNS,
    CANDIDATE_INTERVAL_DTYPES,
    CANDIDATE_INTERVAL_NULLABLE,
    CANDIDATE_SCHEMA_VERSION,
    ELIGIBILITY_BASE_COLUMNS,
    ELIGIBILITY_BASE_DTYPES,
    ELIGIBILITY_BASE_NULLABLE,
    ELIGIBILITY_SCHEMA_VERSION,
    RUNNER_VERSION,
    build_age_long,
    build_candidate_intervals,
    build_eligibility_base,
    build_spine,
    dataframe_manifest,
    derive_structural_ineligibility_reason_primary,
    resolve_cohort_codebook_path,
    resolve_cohort_dataset_path,
    select_spine_core,
    sha256_file,
    validate_candidate_intervals,
    validate_eligibility_base,
)


ROOT = Path(__file__).resolve().parents[2]
ADMIN = ROOT / "00_admin"
REGISTRY_PATH = ADMIN / "cohort_registry.csv"
CODEBOOK_MANIFEST_PATH = ADMIN / "codebooks" / "codebook_manifest.csv"
SPINE_MAPPING_PATH = ADMIN / "variable_maps" / "spine_variable_map.csv"
ELIGIBILITY_MAPPING_PATH = ADMIN / "variable_maps" / "eligibility_base_variable_map.csv"
SCOPE_RULES_PATH = ADMIN / "variable_maps" / "scope_status_rules.csv"
WAVE_MAP_PATH = ADMIN / "wave_maps.csv"
DESIGN_REGISTER_PATH = ADMIN / "panel_design_register.csv"
MANIFEST_ROOT = ADMIN / "panel_manifests"
DERIVED_ROOT = ROOT / "01_data" / "derived" / "panel_components"
CANDIDATE_FREEZE_PATH = (
    MANIFEST_ROOT / "candidate_intervals" / "candidate_intervals_v3_freeze.json"
)
ELIGIBILITY_FREEZE_PATH = (
    MANIFEST_ROOT / "eligibility_base" / "eligibility_base_v2_freeze.json"
)
RUN_R = ADMIN / "run_r.ps1"
R_EXTRACT_SPINE = Path(__file__).parent / "r" / "extract_spine_source.R"
R_TAGGED_FIXTURE = Path(__file__).parent / "r" / "tagged_missing_roundtrip.R"

COHORTS = ("HRS", "ELSA", "KLoSA")
IMPLEMENTED_REAL_STAGES = (
    "preflight",
    "spine",
    "candidate-intervals",
    "eligibility-base",
    "outcome-components",
    "fi-panel",
    "wealth",
    "exposure",
    "covariate",
    "weights",
    "landmark-ledger",
    "eligibility-final",
    "analysis-panel",
)
ALL_STAGES = (
    "preflight",
    "spine",
    "candidate-intervals",
    "eligibility-base",
    "outcome-components",
    "fi-panel",
    "wealth",
    "exposure",
    "covariate",
    "weights",
    "landmark-ledger",
    "eligibility-final",
    "analysis-panel",
    "validate",
    "cohort-model",
    "meta",
)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    """以 UTF-8 读取 CSV。"""
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def registry_for(cohort: str) -> dict[str, str]:
    """读取指定队列的来源登记。"""
    matches = [row for row in read_csv_rows(REGISTRY_PATH) if row["cohort"] == cohort]
    if len(matches) != 1:
        raise ValueError(f"{cohort}: cohort_registry 必须恰有一行")
    record = matches[0].copy()
    record["dataset_path"] = str(resolve_cohort_dataset_path(cohort))
    return record


def codebook_for(cohort: str) -> dict[str, str]:
    """读取指定队列的 codebook 登记。"""
    matches = [
        row for row in read_csv_rows(CODEBOOK_MANIFEST_PATH) if row["cohort"] == cohort
    ]
    if len(matches) != 1:
        raise ValueError(f"{cohort}: codebook_manifest 必须恰有一行")
    return matches[0]


def mapping_rows_for(
    cohort: str,
    mapping_path: Path,
    domains: tuple[str, ...],
) -> list[dict[str, str]]:
    """读取并验证指定结构阶段 mapping。"""
    matches = [
        row
        for row in read_csv_rows(mapping_path)
        if row["cohort"] == cohort and row["domain"] in domains
    ]
    if not matches:
        raise ValueError(f"{cohort}: 缺少 {domains} mapping")
    blocked = [row for row in matches if row["verification_status"] != "verified"]
    if blocked:
        raise ValueError(f"{cohort}: {domains} mapping 含未 verified 项")
    return matches


def expanded_source_variables(
    cohort: str,
    mapping_path: Path,
    domains: tuple[str, ...],
) -> list[str]:
    """把 mapping 中的波次模板展开为精确变量名。"""
    result: list[str] = []
    for row in mapping_rows_for(cohort, mapping_path, domains):
        template = row["source_variable_template"]
        if "{wave}" in template:
            for wave in range(int(row["wave_min"]), int(row["wave_max"]) + 1):
                result.append(template.format(wave=wave))
        else:
            result.append(template)
    return list(dict.fromkeys(result))


def required_structural_variables(cohort: str) -> list[str]:
    """汇总本轮 preflight、spine 与 eligibility-base 所需变量。"""
    variables = expanded_source_variables(
        cohort,
        SPINE_MAPPING_PATH,
        ("spine", "scope"),
    )
    variables.extend(
        expanded_source_variables(
            cohort,
            ELIGIBILITY_MAPPING_PATH,
            ("eligibility_base",),
        )
    )
    return list(dict.fromkeys(variables))


def utc_mtime(path: Path) -> str:
    """返回 ISO-8601 UTC 修改时间。"""
    return datetime.fromtimestamp(path.stat().st_mtime, tz=UTC).isoformat()


def run_command(
    command: list[str],
    *,
    cwd: Path = ROOT,
    capture: bool = True,
) -> subprocess.CompletedProcess[str]:
    """运行子进程并在失败时保留完整输出。"""
    completed = subprocess.run(
        command,
        cwd=cwd,
        check=False,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=capture,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "子进程失败：\n"
            + " ".join(command)
            + "\nSTDOUT:\n"
            + (completed.stdout or "")
            + "\nSTDERR:\n"
            + (completed.stderr or "")
        )
    return completed


def powershell_executable() -> str:
    """定位 PowerShell；优先使用 Windows PowerShell 以调用项目 launcher。"""
    executable = shutil.which("powershell.exe") or shutil.which("powershell") or shutil.which("pwsh")
    if not executable:
        raise FileNotFoundError("找不到 PowerShell")
    return executable


def run_r_script(script: Path, arguments: list[str]) -> str:
    """跨平台调用研究者安装的Rscript，不使用私有启动器。"""
    executable = os.environ.get("WF_RSCRIPT") or shutil.which("Rscript")
    if not executable:
        raise FileNotFoundError("Mode B requires Rscript 4.5.1; install R or set WF_RSCRIPT.")
    completed = run_command([executable, str(script), *arguments])
    return ((completed.stdout or "") + (completed.stderr or "")).strip()


def git_output(arguments: list[str], *, binary: bool = False) -> bytes | str:
    """读取 Git 状态，不改变仓库。"""
    completed = subprocess.run(
        ["git", *arguments],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    if binary:
        return completed.stdout
    return completed.stdout.decode("utf-8", errors="replace").strip()


def git_provenance() -> dict[str, Any]:
    """记录 dirty worktree 的可复核代码指纹。"""
    commit = str(git_output(["rev-parse", "HEAD"]))
    status = str(git_output(["status", "--porcelain=v1"]))
    diff = bytes(git_output(["diff", "--binary", "HEAD"], binary=True))
    untracked_text = str(
        git_output(
            [
                "ls-files",
                "--others",
                "--exclude-standard",
                "--",
                "AGENTS.md",
                ".gitignore",
                "renv.lock",
                "00_admin",
                "02_code/01_panel_build",
            ]
        )
    )
    inventory: list[str] = []
    for relative in sorted(line for line in untracked_text.splitlines() if line):
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


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """以 UTF-8 原子写入 JSON。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def rate_record(
    numerator: int,
    denominator: int,
    denominator_definition: str,
) -> dict[str, Any]:
    """生成带明确分母定义的聚合比例。"""
    return {
        "numerator": int(numerator),
        "denominator": int(denominator),
        "denominator_definition": denominator_definition,
        "rate": None if denominator == 0 else float(numerator / denominator),
    }


def inspect_stata(path: Path) -> dict[str, Any]:
    """只读 Stata header 和 variable labels。"""
    with pd.io.stata.StataReader(path) as reader:
        reader._ensure_open()  # pandas 尚未提供等价的公共 metadata-only API
        variables = list(reader._varlist)
        labels = reader.variable_labels()
        return {
            "stata_release_observed": int(reader._format_version),
            "nobs_observed": int(reader._nobs),
            "nvars_observed": int(reader._nvar),
            "dataset_label_observed": str(reader._data_label or "").strip(),
            "variables": variables,
            "variable_labels": labels,
        }


def run_tagged_missing_fixture() -> dict[str, Any]:
    """在含中文的系统临时目录验证 DTA -> haven -> Arrow -> Python。"""
    with tempfile.TemporaryDirectory(prefix="wealth_recovery_缺失码_") as directory:
        output_dir = Path(directory)
        r_output = run_r_script(R_TAGGED_FIXTURE, [str(output_dir)])
        parquet = output_dir / "tagged_missing_roundtrip.parquet"
        if not parquet.is_file():
            raise FileNotFoundError("tagged missing fixture 未生成 Parquet")
        data = pd.read_parquet(parquet).sort_values("row_id").reset_index(drop=True)
        expected_tags: list[str | None] = [
            "d",
            "r",
            "m",
            "u",
            "v",
            "p",
            "system_na",
            None,
            None,
            None,
        ]
        observed_tags = [None if pd.isna(value) else str(value) for value in data["value__missing_tag"]]
        if observed_tags != expected_tags:
            raise AssertionError(
                f"tagged missing round-trip 失败：{observed_tags} != {expected_tags}"
            )
        expected_values = [None, None, None, None, None, None, None, 0.0, 1.0, -1.0]
        observed_values = [None if pd.isna(value) else float(value) for value in data["value__value"]]
        if observed_values != expected_values:
            raise AssertionError(
                f"tagged missing value round-trip 失败：{observed_values} != {expected_values}"
            )
        if "tagged_missing_roundtrip=PASS" not in r_output:
            raise AssertionError("R 未确认 value labels round-trip")
        remaining = list(output_dir.iterdir())
        return {
            "status": "PASS",
            "cases": len(data),
            "tags_verified": ["d", "r", "m", "u", "v", "p", "system_na"],
            "temporary_files_before_cleanup": [path.name for path in remaining],
            "temporary_directory_cleanup": "managed_by_TemporaryDirectory",
        }


def common_manifest(stage: str, cohort: str) -> dict[str, Any]:
    """构建所有阶段共享的 provenance。"""
    return {
        "manifest_schema_version": "panel_build_manifest_v1",
        "runner_version": RUNNER_VERSION,
        "stage": stage,
        "cohort": cohort,
        "created_at_utc": datetime.now(tz=UTC).isoformat(),
        **git_provenance(),
        "mapping_manifests_sha256": {
            "spine_variable_map": sha256_file(SPINE_MAPPING_PATH),
            "eligibility_base_variable_map": sha256_file(ELIGIBILITY_MAPPING_PATH),
            "scope_status_rules": sha256_file(SCOPE_RULES_PATH),
        },
        "design_register_sha256": sha256_file(DESIGN_REGISTER_PATH),
        "cohort_registry_sha256": sha256_file(REGISTRY_PATH),
        "wave_map_sha256": sha256_file(WAVE_MAP_PATH),
    }


def run_preflight(cohort: str, tagged_fixture: dict[str, Any]) -> dict[str, Any]:
    """验证 raw/codebook 版本配对并写入只读来源指纹。"""
    registry = registry_for(cohort)
    codebook = codebook_for(cohort)
    dataset = Path(registry["dataset_path"])
    source_codebook = resolve_cohort_codebook_path(cohort)
    local_codebook = source_codebook

    for role, path in (
        ("dataset", dataset),
        ("source codebook", source_codebook),
        ("local codebook", local_codebook),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{cohort}: {role} 不存在：{path}")

    metadata = inspect_stata(dataset)
    required_variables = required_structural_variables(cohort)
    missing_variables = sorted(set(required_variables).difference(metadata["variables"]))

    expected_codebook_hash = codebook["sha256"].lower()
    source_codebook_hash = sha256_file(source_codebook)
    local_codebook_hash = sha256_file(local_codebook)
    dataset_hash = sha256_file(dataset)

    checks = {
        "registry_version_pair_preverified": registry["version_pair_verified"].upper() == "TRUE",
        "dataset_filename_matches": dataset.name == registry["dataset_filename"],
        "stata_release_matches": metadata["stata_release_observed"]
        == int(registry["stata_release_expected"]),
        "nobs_matches": metadata["nobs_observed"] == int(registry["nobs_expected"]),
        "nvars_matches": metadata["nvars_observed"] == int(registry["nvars_expected"]),
        "dataset_label_matches": (
            not registry["dataset_label_expected"]
            or metadata["dataset_label_observed"] == registry["dataset_label_expected"]
        ),
        "dataset_version_registry_matches": registry["dataset_version_observed"]
        == registry["dataset_version_expected"],
        "codebook_version_registry_matches": registry["codebook_version_observed"]
        == registry["codebook_version_expected"],
        "codebook_dataset_version_matches": codebook["dataset_version"]
        == registry["dataset_version_expected"],
        "codebook_version_matches": codebook["codebook_version"]
        == registry["codebook_version_expected"],
        "source_codebook_hash_matches": source_codebook_hash == expected_codebook_hash,
        "local_codebook_hash_matches": local_codebook_hash == expected_codebook_hash,
        "copied_codebook_is_byte_identical": source_codebook_hash == local_codebook_hash,
        "required_structural_variables_present": not missing_variables,
        "tagged_missing_contract_passed": tagged_fixture["status"] == "PASS",
    }
    version_pair_verified = all(checks.values())
    manifest = {
        **common_manifest("preflight", cohort),
        "dataset": {
            "path": str(dataset),
            "bytes": dataset.stat().st_size,
            "modified_utc": utc_mtime(dataset),
            "sha256": dataset_hash,
            "dataset_version_expected": registry["dataset_version_expected"],
            "dataset_version_observed": registry["dataset_version_observed"],
            "dataset_label_expected": registry["dataset_label_expected"],
            "dataset_label_observed": metadata["dataset_label_observed"],
            "stata_release_expected": int(registry["stata_release_expected"]),
            "stata_release_observed": metadata["stata_release_observed"],
            "nobs_expected": int(registry["nobs_expected"]),
            "nobs_observed": metadata["nobs_observed"],
            "nvars_expected": int(registry["nvars_expected"]),
            "nvars_observed": metadata["nvars_observed"],
        },
        "codebook": {
            "source_path": str(source_codebook),
            "local_path": str(local_codebook),
            "version_expected": registry["codebook_version_expected"],
            "version_observed": registry["codebook_version_observed"],
            "sha256_expected": expected_codebook_hash,
            "source_sha256_observed": source_codebook_hash,
            "local_sha256_observed": local_codebook_hash,
        },
        "required_structural_variables": required_variables,
        "missing_structural_variables": missing_variables,
        "tagged_missing_fixture": tagged_fixture,
        "checks": checks,
        "version_pair_verified": version_pair_verified,
    }
    manifest_path = MANIFEST_ROOT / "preflight" / f"{cohort.lower()}_preflight.json"
    write_json(manifest_path, manifest)
    if not version_pair_verified:
        failed = [name for name, passed in checks.items() if not passed]
        raise RuntimeError(f"{cohort}: preflight 失败：{failed}")
    print(f"{cohort}: preflight PASS | raw sha256={dataset_hash}")
    return manifest


def load_preflight(cohort: str) -> dict[str, Any]:
    """确认最近 preflight 仍对应当前文件大小与修改时间。"""
    path = MANIFEST_ROOT / "preflight" / f"{cohort.lower()}_preflight.json"
    if not path.is_file():
        raise RuntimeError(f"{cohort}: 缺少 preflight manifest，请先运行 preflight")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if not manifest.get("version_pair_verified"):
        raise RuntimeError(f"{cohort}: version_pair_verified 不是 TRUE")
    expected_mapping_hashes = {
        "spine_variable_map": sha256_file(SPINE_MAPPING_PATH),
        "eligibility_base_variable_map": sha256_file(ELIGIBILITY_MAPPING_PATH),
        "scope_status_rules": sha256_file(SCOPE_RULES_PATH),
    }
    if manifest.get("mapping_manifests_sha256") != expected_mapping_hashes:
        raise RuntimeError(f"{cohort}: 结构 mapping 自 preflight 后改变，请重跑 preflight")
    dataset = Path(manifest["dataset"]["path"])
    if dataset.stat().st_size != manifest["dataset"]["bytes"]:
        raise RuntimeError(f"{cohort}: raw file size 自 preflight 后改变")
    if utc_mtime(dataset) != manifest["dataset"]["modified_utc"]:
        raise RuntimeError(f"{cohort}: raw file mtime 自 preflight 后改变")
    local_codebook = Path(manifest["codebook"]["local_path"])
    if sha256_file(local_codebook) != manifest["codebook"]["sha256_expected"]:
        raise RuntimeError(f"{cohort}: local codebook 自 preflight 后改变")
    return manifest


def run_spine(cohort: str) -> dict[str, Any]:
    """读取最小结构 slice 并生成 scope/observed 分离的 global spine。"""
    preflight = load_preflight(cohort)
    registry = registry_for(cohort)
    output_dir = DERIVED_ROOT / cohort / "01_spine"
    output_dir.mkdir(parents=True, exist_ok=True)
    source_path = output_dir / f"{cohort.lower()}_spine_source.parquet"
    spine_path = output_dir / f"{cohort.lower()}_spine.parquet"
    schema_path = MANIFEST_ROOT / "spine" / f"{cohort.lower()}_source_variables.csv"
    schema_path.parent.mkdir(parents=True, exist_ok=True)

    r_output = run_r_script(
        R_EXTRACT_SPINE,
        [
            registry["dataset_path"],
            str(SPINE_MAPPING_PATH),
            cohort,
            "spine;scope",
            str(source_path),
            str(schema_path),
        ],
    )
    source = pd.read_parquet(source_path)
    wave_map = pd.read_csv(WAVE_MAP_PATH, encoding="utf-8")
    wave_map = wave_map.loc[wave_map["cohort"] == cohort].copy()
    structural_mapping = pd.read_csv(SPINE_MAPPING_PATH, encoding="utf-8")
    scope_rules = pd.read_csv(SCOPE_RULES_PATH, encoding="utf-8")
    spine = build_spine(
        source,
        registry,
        wave_map,
        structural_mapping,
        scope_rules,
    )
    spine.to_parquet(spine_path, index=False, compression="zstd")

    wave_counts = []
    for wave, group in spine.groupby("wave", sort=True):
        observed = group["observed_in_wave"].astype("boolean")
        wave_counts.append(
            {
                "wave": int(wave),
                "n_person": int(len(group)),
                "n_observed": int(observed.eq(True).fillna(False).sum()),
                "n_nonrespondent": int(observed.eq(False).fillna(False).sum()),
                "n_observation_status_unknown": int(observed.isna().sum()),
                "scope_status_counts": {
                    str(status): int(count)
                    for status, count in group["person_wave_scope_status"]
                    .value_counts(dropna=False)
                    .sort_index()
                    .items()
                },
                "n_observed_with_scope_not_in_scope": int(
                    (
                        observed.eq(True).fillna(False)
                        & group["person_wave_scope_status"].ne("in_scope")
                    ).sum()
                ),
                "n_interview_year_available": int(group["interview_year"].notna().sum()),
            }
        )

    manifest = {
        **common_manifest("spine", cohort),
        "preflight_manifest_sha256": sha256_file(
            MANIFEST_ROOT / "preflight" / f"{cohort.lower()}_preflight.json"
        ),
        "input_dataset_sha256": preflight["dataset"]["sha256"],
        "source_slice": {
            "path": str(source_path),
            "sha256": sha256_file(source_path),
            "selected_variables": expanded_source_variables(
                cohort,
                SPINE_MAPPING_PATH,
                ("spine", "scope"),
            ),
            "selected_variable_count": len(
                expanded_source_variables(
                    cohort,
                    SPINE_MAPPING_PATH,
                    ("spine", "scope"),
                )
            ),
            "r_output": r_output,
        },
        "source_variable_manifest": {
            "path": str(schema_path),
            "sha256": sha256_file(schema_path),
        },
        "output": {
            "path": str(spine_path),
            "sha256": sha256_file(spine_path),
            **dataframe_manifest(spine, ["cohort", "person_id", "wave"]),
        },
        "wave_counts": wave_counts,
        "n_person_with_multiple_authoritative_entry_categories": int(
            spine.loc[
                spine["sample_entry_wave_multiple_candidates"].eq(True).fillna(False),
                "person_id",
            ].nunique()
        ),
        "n_observed_with_scope_not_in_scope": int(
            (
                spine["observed_in_wave"].eq(True).fillna(False)
                & spine["person_wave_scope_status"].ne("in_scope")
            ).sum()
        ),
        "contains_only_structural_fields": True,
        "scope_and_observation_are_separate": True,
    }
    manifest_path = MANIFEST_ROOT / "spine" / f"{cohort.lower()}_spine.json"
    write_json(manifest_path, manifest)
    print(
        f"{cohort}: spine PASS | persons={source.shape[0]} rows={len(spine)} "
        f"waves={wave_map.shape[0]}"
    )
    return manifest


def load_spine(cohort: str) -> tuple[pd.DataFrame, Path, dict[str, Any]]:
    """读取已经验证的 spine。"""
    manifest_path = MANIFEST_ROOT / "spine" / f"{cohort.lower()}_spine.json"
    if not manifest_path.is_file():
        raise RuntimeError(f"{cohort}: 缺少 spine manifest，请先运行 spine")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    path = Path(manifest["output"]["path"])
    if not path.is_file():
        raise FileNotFoundError(f"{cohort}: spine 文件不存在：{path}")
    if sha256_file(path) != manifest["output"]["sha256"]:
        raise RuntimeError(f"{cohort}: spine 哈希与 manifest 不一致")
    data = pd.read_parquet(path)
    if data.duplicated(["cohort", "person_id", "wave"]).any():
        raise RuntimeError(f"{cohort}: spine 主键不唯一")
    return data, path, manifest


def candidate_schema_contract() -> list[dict[str, Any]]:
    """返回 Candidate Interval v3 的有序数据类型与缺失契约。"""
    return [
        {
            "column": column,
            "pandas_dtype": CANDIDATE_INTERVAL_DTYPES[column],
            "nullable": CANDIDATE_INTERVAL_NULLABLE[column],
        }
        for column in CANDIDATE_INTERVAL_COLUMNS
    ]


def assert_candidate_output_writable(cohort: str, output_path: Path) -> None:
    """冻结记录存在时禁止覆盖 Candidate Interval v3。"""
    if not CANDIDATE_FREEZE_PATH.is_file():
        return
    freeze = json.loads(CANDIDATE_FREEZE_PATH.read_text(encoding="utf-8"))
    frozen = freeze.get("cohorts", {}).get(cohort)
    if frozen is None:
        raise RuntimeError(f"{cohort}: Candidate freeze 缺少队列记录，禁止写入")
    if Path(frozen["path"]) != output_path:
        raise RuntimeError(f"{cohort}: Candidate freeze 路径不一致，禁止写入")
    if not output_path.is_file():
        raise RuntimeError(f"{cohort}: 冻结 Candidate 文件缺失，禁止重建覆盖")
    if sha256_file(output_path) != frozen["sha256"]:
        raise RuntimeError(f"{cohort}: 冻结 Candidate 哈希不一致，禁止覆盖")
    raise RuntimeError(f"{cohort}: Candidate Interval v3 已冻结，禁止原地覆盖")


def run_candidate_intervals(cohort: str) -> dict[str, Any]:
    """生成不依赖个人 scope、t+1、wealth 或 outcome 的 global 相邻区间。"""
    full_spine, spine_path, spine_manifest = load_spine(cohort)
    spine = select_spine_core(full_spine)
    intervals = build_candidate_intervals(spine)
    output_dir = DERIVED_ROOT / cohort / "02_candidate_intervals"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{cohort.lower()}_candidate_intervals_v3.parquet"
    assert_candidate_output_writable(cohort, output_path)
    intervals.to_parquet(output_path, index=False, compression="zstd")

    observed_tplus1 = intervals["observed_tplus1"].astype("boolean")
    observed_tplus2 = intervals["observed_tplus2"].astype("boolean")
    observed_tplus2_or_later = intervals["observed_tplus2_or_later"].astype(
        "boolean"
    )
    pair_observed = intervals["interval_both_observed"].eq(True).fillna(False)
    has_tplus1 = intervals["tplus1_wave"].notna()
    has_tplus2 = intervals["tplus2_wave"].notna()
    has_tplus2_or_later = has_tplus2
    manifest = {
        **common_manifest("candidate-intervals", cohort),
        "active_schema": CANDIDATE_SCHEMA_VERSION,
        "schema_contract": candidate_schema_contract(),
        "input_spine_path": str(spine_path),
        "input_spine_sha256": spine_manifest["output"]["sha256"],
        "output": {
            "path": str(output_path),
            "sha256": sha256_file(output_path),
            **dataframe_manifest(
                intervals,
                ["cohort", "person_id", "candidate_shock_wave"],
            ),
        },
        "invariants": {
            "full_cohort_global_adjacent_interval_ledger_retained": True,
            "person_scope_filter_applied": False,
            "requires_tplus1": False,
            "uses_wealth": False,
            "uses_recovery": False,
            "uses_fi_or_outcome": False,
            "shock_identification_performed": False,
        },
        "counts": {
            "n_person": int(intervals["person_id"].nunique()),
            "n_global_intervals": int(len(intervals)),
            "n_interval_both_waves_observed": int(
                pair_observed.sum()
            ),
            "n_cohort_tplus1_wave_absent": int((~has_tplus1).sum()),
            "n_cohort_tplus2_wave_absent": int((~has_tplus2).sum()),
            "n_tplus1_wave_exists_not_observed": int(
                (has_tplus1 & observed_tplus1.eq(False).fillna(False)).sum()
            ),
            "n_tplus1_wave_exists_observation_unknown": int(
                (has_tplus1 & observed_tplus1.isna()).sum()
            ),
            "n_tplus2_wave_absent_observation_not_applicable": int(
                ((~has_tplus2) & observed_tplus2.isna()).sum()
            ),
        },
        "conditional_availability": {
            "tplus1_observed": rate_record(
                int((pair_observed & has_tplus1 & observed_tplus1.eq(True).fillna(False)).sum()),
                int((pair_observed & has_tplus1).sum()),
                "global intervals with both t-1 and t observed and a cohort t+1 wave",
            ),
            "exact_tplus2_observed": rate_record(
                int((pair_observed & has_tplus2 & observed_tplus2.eq(True).fillna(False)).sum()),
                int((pair_observed & has_tplus2).sum()),
                "global intervals with both t-1 and t observed and an exact cohort t+2 wave",
            ),
            "any_tplus2_or_later_observed": rate_record(
                int(
                    (
                        pair_observed
                        & has_tplus2_or_later
                        & observed_tplus2_or_later.eq(True).fillna(False)
                    ).sum()
                ),
                int((pair_observed & has_tplus2_or_later).sum()),
                "global intervals with both t-1 and t observed and at least one cohort t+2-or-later wave",
            ),
        },
    }
    manifest_path = (
        MANIFEST_ROOT
        / "candidate_intervals"
        / f"{cohort.lower()}_candidate_intervals.json"
    )
    write_json(manifest_path, manifest)
    print(
        f"{cohort}: candidate-intervals PASS | persons="
        f"{manifest['counts']['n_person']} intervals={len(intervals)}"
    )
    return manifest


def load_candidate_intervals(
    cohort: str,
) -> tuple[pd.DataFrame, Path, dict[str, Any]]:
    """读取已经验证的 Candidate Interval v3 ledger。"""
    manifest_path = (
        MANIFEST_ROOT
        / "candidate_intervals"
        / f"{cohort.lower()}_candidate_intervals.json"
    )
    if not manifest_path.is_file():
        raise RuntimeError(f"{cohort}: 缺少 candidate-intervals manifest")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("active_schema") != CANDIDATE_SCHEMA_VERSION:
        raise RuntimeError(
            f"{cohort}: Candidate manifest 不是 {CANDIDATE_SCHEMA_VERSION}"
        )
    if manifest.get("schema_contract") != candidate_schema_contract():
        raise RuntimeError(f"{cohort}: Candidate manifest schema contract 不一致")
    path = Path(manifest["output"]["path"])
    if not path.is_file():
        raise FileNotFoundError(f"{cohort}: candidate intervals 不存在：{path}")
    if sha256_file(path) != manifest["output"]["sha256"]:
        raise RuntimeError(f"{cohort}: candidate intervals 哈希与 manifest 不一致")
    data = pd.read_parquet(path)
    validate_candidate_intervals(data)
    return data, path, manifest


def eligibility_schema_contract() -> list[dict[str, Any]]:
    """返回 Eligibility Base v2 的有序数据类型与非空契约。"""
    return [
        {
            "column": column,
            "pandas_dtype": ELIGIBILITY_BASE_DTYPES[column],
            "nullable": ELIGIBILITY_BASE_NULLABLE[column],
        }
        for column in ELIGIBILITY_BASE_COLUMNS
    ]


def assert_eligibility_output_writable(cohort: str, output_path: Path) -> None:
    """冻结记录存在时禁止覆盖 Eligibility Base v2。"""
    if not ELIGIBILITY_FREEZE_PATH.is_file():
        return
    freeze = json.loads(ELIGIBILITY_FREEZE_PATH.read_text(encoding="utf-8"))
    frozen = freeze.get("cohorts", {}).get(cohort)
    if frozen is None:
        raise RuntimeError(f"{cohort}: Eligibility freeze 缺少队列记录，禁止写入")
    if Path(frozen["path"]) != output_path:
        raise RuntimeError(f"{cohort}: Eligibility freeze 路径不一致，禁止写入")
    if not output_path.is_file():
        raise RuntimeError(f"{cohort}: 冻结 Eligibility 文件缺失，禁止重建覆盖")
    if sha256_file(output_path) != frozen["sha256"]:
        raise RuntimeError(f"{cohort}: 冻结 Eligibility 哈希不一致，禁止覆盖")
    raise RuntimeError(f"{cohort}: Eligibility Base v2 已冻结，禁止原地覆盖")


def load_eligibility_base(
    cohort: str,
) -> tuple[pd.DataFrame, Path, dict[str, Any]]:
    """从活动 manifest 读取并验证 Eligibility Base v2。"""
    manifest_path = (
        MANIFEST_ROOT / "eligibility_base" / f"{cohort.lower()}_eligibility_base.json"
    )
    if not manifest_path.is_file():
        raise RuntimeError(f"{cohort}: 缺少 eligibility-base manifest")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("active_schema") != ELIGIBILITY_SCHEMA_VERSION:
        raise RuntimeError(
            f"{cohort}: Eligibility manifest 不是 {ELIGIBILITY_SCHEMA_VERSION}"
        )
    if manifest.get("schema_contract") != eligibility_schema_contract():
        raise RuntimeError(f"{cohort}: Eligibility manifest schema contract 不一致")
    path = Path(manifest["output"]["path"])
    if not path.is_file():
        raise FileNotFoundError(f"{cohort}: eligibility base 不存在：{path}")
    if sha256_file(path) != manifest["output"]["sha256"]:
        raise RuntimeError(f"{cohort}: eligibility base 哈希与 manifest 不一致")
    data = pd.read_parquet(path)
    validate_eligibility_base(data)
    return data, path, manifest


def locked_age_threshold() -> float:
    """确认 manuscript 中的 t-1 年龄门槛已写入 locked design register。"""
    rows = [
        row
        for row in read_csv_rows(DESIGN_REGISTER_PATH)
        if row["decision_id"] == "age_at_tminus1_rule"
    ]
    if len(rows) != 1:
        raise RuntimeError("design register 必须恰有一条 age_at_tminus1_rule")
    row = rows[0]
    if row["status"] != "locked" or row["selected_value"] != "age_ge_60":
        raise RuntimeError("age_at_tminus1_rule 尚未 locked 为 age_ge_60")
    return 60.0


def run_eligibility_base(cohort: str) -> dict[str, Any]:
    """构建 scope、双波观察和 t-1 年龄组成的结构纳入排除 ledger。"""
    age_threshold = locked_age_threshold()
    registry = registry_for(cohort)
    full_spine, spine_path, spine_manifest = load_spine(cohort)
    spine = select_spine_core(full_spine)
    intervals, intervals_path, intervals_manifest = load_candidate_intervals(cohort)

    output_dir = DERIVED_ROOT / cohort / "03_eligibility_base"
    output_dir.mkdir(parents=True, exist_ok=True)
    source_path = output_dir / f"{cohort.lower()}_eligibility_base_source.parquet"
    output_path = output_dir / f"{cohort.lower()}_eligibility_base_v2.parquet"
    assert_eligibility_output_writable(cohort, output_path)
    schema_path = (
        MANIFEST_ROOT / "eligibility_base" / f"{cohort.lower()}_source_variables.csv"
    )
    schema_path.parent.mkdir(parents=True, exist_ok=True)

    r_output = run_r_script(
        R_EXTRACT_SPINE,
        [
            registry["dataset_path"],
            str(ELIGIBILITY_MAPPING_PATH),
            cohort,
            "eligibility_base",
            str(source_path),
            str(schema_path),
        ],
    )
    source = pd.read_parquet(source_path)
    wave_map = pd.read_csv(WAVE_MAP_PATH, encoding="utf-8")
    wave_map = wave_map.loc[wave_map["cohort"] == cohort].copy()
    eligibility_mapping = pd.read_csv(ELIGIBILITY_MAPPING_PATH, encoding="utf-8")
    age_long = build_age_long(source, registry, wave_map, eligibility_mapping)
    eligibility = build_eligibility_base(
        intervals,
        spine,
        age_long,
        age_threshold=age_threshold,
    )
    eligibility.to_parquet(output_path, index=False, compression="zstd")
    persisted = pd.read_parquet(output_path)
    validate_eligibility_base(persisted)
    pd.testing.assert_frame_equal(
        persisted,
        eligibility,
        check_dtype=True,
        check_exact=True,
    )
    eligibility = persisted
    primary_reasons = derive_structural_ineligibility_reason_primary(eligibility)

    scope_pass = eligibility["scope_assessment_status"].eq("pass")
    pair_pass = eligibility["pair_observation_status"].eq("pass")
    age_pass = eligibility["age_assessment_status"].eq("pass")
    structural_pass = eligibility["structural_eligibility_status"].eq("pass")
    scope_pair = scope_pass & pair_pass

    manifest = {
        **common_manifest("eligibility-base", cohort),
        "active_schema": ELIGIBILITY_SCHEMA_VERSION,
        "schema_contract": eligibility_schema_contract(),
        "input_spine": {
            "path": str(spine_path),
            "sha256": spine_manifest["output"]["sha256"],
        },
        "input_candidate_intervals": {
            "path": str(intervals_path),
            "sha256": intervals_manifest["output"]["sha256"],
        },
        "source_slice": {
            "path": str(source_path),
            "sha256": sha256_file(source_path),
            "selected_variables": expanded_source_variables(
                cohort,
                ELIGIBILITY_MAPPING_PATH,
                ("eligibility_base",),
            ),
            "selected_variable_count": len(
                expanded_source_variables(
                    cohort,
                    ELIGIBILITY_MAPPING_PATH,
                    ("eligibility_base",),
                )
            ),
            "r_output": r_output,
        },
        "source_variable_manifest": {
            "path": str(schema_path),
            "sha256": sha256_file(schema_path),
        },
        "age_rule": {
            "design_decision_id": "age_at_tminus1_rule",
            "status": "locked",
            "selected_value": "age_ge_60",
            "threshold_years": age_threshold,
            "measurement_wave": "tminus1",
        },
        "output": {
            "path": str(output_path),
            "sha256": sha256_file(output_path),
            **dataframe_manifest(
                eligibility,
                ["cohort", "person_id", "candidate_shock_wave"],
            ),
        },
        "component_status_counts": {
            column: {
                str(status): int(count)
                for status, count in eligibility[column]
                .value_counts(dropna=False)
                .sort_index()
                .items()
            }
            for column in (
                "scope_assessment_status",
                "pair_observation_status",
                "age_assessment_status",
                "structural_eligibility_status",
            )
        },
        "primary_reason_counts": {
            str(reason): int(count)
            for reason, count in (
                primary_reasons.fillna("pass_no_reason")
                .value_counts()
                .sort_index()
                .items()
            )
        },
        "denominator_layers": {
            "level_1_global_interval_universe": int(len(eligibility)),
            "level_2_person_scope_interval_universe": int(scope_pass.sum()),
            "level_3_double_observed_within_person_scope": int(scope_pair.sum()),
            "structural_age_eligible_intervals": int(structural_pass.sum()),
        },
        "conditional_rates": {
            "person_scope_among_global": rate_record(
                int(scope_pass.sum()),
                int(len(eligibility)),
                "all person by cohort-global adjacent-wave intervals",
            ),
            "double_observed_among_person_scope": rate_record(
                int(scope_pair.sum()),
                int(scope_pass.sum()),
                "intervals with both endpoints explicitly in_scope",
            ),
            "age_ge60_among_scope_double_observed": rate_record(
                int((scope_pair & age_pass).sum()),
                int(scope_pair.sum()),
                "in-scope intervals with both t-1 and t observed; missing age remains in denominator",
            ),
            "structural_pass_among_global": rate_record(
                int(structural_pass.sum()),
                int(len(eligibility)),
                "all person by cohort-global adjacent-wave intervals",
            ),
        },
        "invariants": {
            "all_global_interval_rows_retained": len(eligibility) == len(intervals),
            "scope_and_observation_assessed_separately": True,
            "no_wealth_fi_covariate_or_weight_fields_read": True,
            "no_shock_or_recovery_classification_performed": True,
            "primary_reason_is_reporting_only": True,
        },
    }
    manifest_path = (
        MANIFEST_ROOT / "eligibility_base" / f"{cohort.lower()}_eligibility_base.json"
    )
    write_json(manifest_path, manifest)
    print(
        f"{cohort}: eligibility-base PASS | global={len(eligibility)} "
        f"in_scope={int(scope_pass.sum())} double_observed={int(scope_pair.sum())} "
        f"structural_pass={int(structural_pass.sum())}"
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Wealth Recovery and Frailty raw-only panel stage runner"
    )
    parser.add_argument("--cohort", required=True, choices=[*COHORTS, "all"])
    parser.add_argument("--stage", required=True, choices=ALL_STAGES)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.stage not in IMPLEMENTED_REAL_STAGES:
        raise SystemExit(
            f"错误：阶段 {args.stage} 超出当前获准的真实数据边界；"
            "本轮必须在 analysis-panel 后停止。"
        )
    cohorts = list(COHORTS) if args.cohort == "all" else [args.cohort]

    if args.stage == "preflight":
        fixture = run_tagged_missing_fixture()
        for cohort in cohorts:
            run_preflight(cohort, fixture)
    elif args.stage == "spine":
        for cohort in cohorts:
            run_spine(cohort)
    elif args.stage == "candidate-intervals":
        for cohort in cohorts:
            run_candidate_intervals(cohort)
    elif args.stage == "eligibility-base":
        for cohort in cohorts:
            run_eligibility_base(cohort)
    elif args.stage == "outcome-components":
        from outcome_components import run_outcome_components

        run_outcome_components(
            cohorts,
            load_spine=load_spine,
            load_candidate_intervals=load_candidate_intervals,
            load_eligibility_base=load_eligibility_base,
        )
    elif args.stage == "fi-panel":
        from fi_panel import run_fi_panel_release

        run_fi_panel_release(
            cohorts,
            load_spine=load_spine,
            load_candidate_intervals=load_candidate_intervals,
            load_eligibility_base=load_eligibility_base,
        )
    elif args.stage == "wealth":
        from wealth_panel import run_wealth_panel_release

        result = run_wealth_panel_release(
            cohorts,
            load_spine=load_spine,
            load_candidate_intervals=load_candidate_intervals,
            load_eligibility_base=load_eligibility_base,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
    elif args.stage == "exposure":
        from exposure_panel import run_exposure_panel_release

        result = run_exposure_panel_release(
            cohorts,
            load_candidate_intervals=load_candidate_intervals,
            load_eligibility_base=load_eligibility_base,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
    elif args.stage == "covariate":
        from covariate_panel import run_covariate_panel_release

        result = run_covariate_panel_release(
            cohorts,
            load_spine=load_spine,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
    elif args.stage == "weights":
        from weight_design_panel import run_weight_design_panel_release

        result = run_weight_design_panel_release(
            cohorts,
            load_spine=load_spine,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
    elif args.stage == "landmark-ledger":
        from analysis_cohort_ledger import run_analysis_cohort_ledger_release

        result = run_analysis_cohort_ledger_release(cohorts)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    elif args.stage == "eligibility-final":
        from final_eligibility import run_final_eligibility_release

        result = run_final_eligibility_release(cohorts)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    elif args.stage == "analysis-panel":
        from analysis_panel import run_analysis_panel_release

        result = run_analysis_panel_release(cohorts)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
