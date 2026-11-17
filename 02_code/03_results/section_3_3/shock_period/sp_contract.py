"""SP01科学合同、来源绑定和分阶段授权；不把规格批准当作拟合许可。"""
from __future__ import annotations

from datetime import UTC, datetime
import hashlib
import inspect
import json
from pathlib import Path
import re
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import scipy.optimize._optimize as bfgs_native
import scipy.optimize._lbfgsb_py as lbfgs_native
import statsmodels.base.optimizer as backend
import statsmodels.regression.mixed_linear_model as mixed_backend

from sp_audit import (ROOT, CODE, COHORTS, LEVELS, INTERPRETATION, FORMULA, sha, rel,
    require, clean, write_json, provenance, verify_records, load_cohort, reconstruct_primary, build_design)
from sp_fit import POLICY, REASONS

CONTRACT = ROOT / "00_admin/model_manifests/shock_period/shock_period_round1e_v1_contract.json"
STAGE0 = ROOT / "06_project_progress/round1e_shock_period_v1/Stage0/round1e_stage0_20261014_01"
STAGE0_SHA = "79d881ca321efab0c6f0bdf9b743d4ee99b040c771d505a1713e2686061c0f3c"
DESIGN_PATH = ROOT / "03_output/3.3 Robustness of the Main Findings/Shock Calendar Period/Audit/round1e_stage0_20261014_01/design_diagnostics.json"
UNITS = ["HRS/SP01", "ELSA/SP01", "KLoSA/SP01"]
NEW_FILES = ("sp_contract.py", "sp_fit.py", "run_shock_period.py", "test_stage12.py", "README_STAGE12.md")
PROGRESS = ROOT / "06_project_progress/round1e_shock_period_v1/Stage1"
PRIMARY = ROOT / "03_output/3.2 Wealth Recovery and Subsequent Frailty Progression/Primary Models"


class GateError(ValueError):
    def __init__(self, message, reason="SOURCE_MISMATCH"):
        super().__init__(message)
        self.reason_code = reason


def checked(condition, message, reason="SOURCE_MISMATCH"):
    if not bool(condition):
        raise GateError(message, reason)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def content_sha(value):
    data = json.dumps(clean(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def safe_id(value):
    checked(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,95}", value) is not None, "非法运行ID")
    return value


def code_hashes():
    return {rel(CODE / name): sha(CODE / name) for name in NEW_FILES}


def read_stage0():
    checked(sha(STAGE0 / "stage0_manifest.json") == STAGE0_SHA, "活动Stage0 manifest身份漂移")
    m = read_json(STAGE0 / "stage0_manifest.json")
    checked(m["status"] == "VALIDATED_FOR_STAGE1_REVIEW" and m["audit_id"] == STAGE0.name, "Stage0状态错误")
    verify_records(m["outputs"])
    return m, read_json(DESIGN_PATH)


def source_records():
    m, d = read_stage0()
    records = {**m["sources"], **m["outputs"], rel(STAGE0 / "stage0_manifest.json"): STAGE0_SHA}
    verify_records(records)
    records.update(code_hashes())
    for module in (backend, bfgs_native, lbfgs_native, mixed_backend):
        p = Path(inspect.getfile(module)).resolve()
        name = rel(p) if p.is_relative_to(ROOT.resolve()) else str(p)
        records[name] = sha(p)
    return records, d


def prepare(cohort, stage0_design):
    frame, lm, primary_x, beta, covariance, diag, anchor = load_cohort(cohort)
    try:
        x, design = build_design(frame, primary_x, cohort)
    except ValueError as error:
        raise GateError(str(error), "DESIGN_RANK_DEFICIENT" if "秩亏" in str(error) or "42/42" in str(error) else "SOURCE_MISMATCH") from error
    old = stage0_design["cohorts"][cohort]
    checked(anchor == old["anchor"], "Primary行集或物理输入漂移")
    checked(design["columns"] == old["columns"], "SP01设计列身份漂移")
    checked(design["rank"] == old["rank"], "SP01设计秩亏", "DESIGN_RANK_DEFICIENT")
    reconstruct = reconstruct_primary(frame, primary_x, beta, diag)
    start = np.r_[beta, np.zeros(x.shape[1] - 42)]
    design.update(anchor=anchor, primary_reconstruction=reconstruct,
        primary_random_intercept_variance=diag["random_intercept_variance"],
        primary_residual_variance=diag["residual_variance"], initial_fixed_effects=start.tolist(),
        latest_wave_support=old["latest_wave_support"], wave_group_counts=old["wave_group_counts"])
    return frame, x, design, beta, covariance, diag, start


def scientific_spec(designs):
    primary_contract = read_json(ROOT / "00_admin/model_manifests/primary_lmm/primary_lmm_v1_contract.json")
    constants = pd.read_csv(PRIMARY / "primary_lmm_standardization_constants.csv", encoding="utf-8", float_precision="round_trip")
    return clean({"schema": "shock_period_scientific_specification_v1", "spec_id": "SP01", "scope": UNITS,
        "reviewer_mapping": {"Editor_Precheck_7": "shock_wave_adjustment_and_context",
            "Editor_FirstDecision_5": "shock_wave_adjustment_only_no_nonshock_reference",
            "Editor_Precheck_1": "financial_crisis_overlap_is_descriptive_not_measurement_validation"},
        "formula_base": FORMULA, "formula_addition": "shock_wave + shock_wave:time_years",
        "columns_policy": "original42_then_nonreference_wave_mains_then_wave_time_numeric_order",
        "cohorts": {c: {k: designs[c][k] for k in ("columns", "primary_columns", "added_columns", "levels", "reference", "rank", "anchor")}
                    for c in COHORTS},
        "outcome": "fi_primary", "sample": "unchanged_primary_complete_case_eligible_all_rows",
        "time": "actual_elapsed_months_from_original_landmark_divided_by_12",
        "recovery_reference": "persistent_or_limited_recovery", "core_term": "recovery:time_years",
        "direction": "substantial_recovery_minus_persistent_or_limited_recovery", "units": "FI units/year",
        "interpretation": INTERPRETATION, "comparison_order": ["Frozen Primary", "SP01"],
        "primary_adjustment": primary_contract["adjustment"], "standardization_constants": constants.to_dict("records"),
        "standardization_constants_sha256": sha(PRIMARY / "primary_lmm_standardization_constants.csv"),
        "fit": {"reml": False, "weighting": "unweighted", "random_effects": "participant_random_intercept", "use_sqrt": True},
        "start": "Primary_fixed_beta_plus_zeros_and_Primary_tau2_over_sigma2; beta_profiled_by_MixedLM",
        "optimizer": {"sequence": ["bfgs", "lbfgs"], **POLICY, "lbfgs_start": "bfgs.params_object.copy()",
            "automatic_retry": False, "exact_native_terminal_logging_required": True},
        "failure_reason_priority": list(REASONS), "basic_statuses": ["PASS", "BOUNDARY_REVIEW", "HOLD_NOT_ESTIMABLE"],
        "output_definitions": ["fixed_effects", "full_fixed_covariance", "beta_SE_CI_P", "tau2_sigma2", "both_optimizer_candidates",
            "native_task_status_message_and_effective_arguments", "score_Hessian_spectra", "sample_design_sources"],
        "stage3_required": {"all_cohorts": ["all_wave_time_parameter_SE_CI", "independent_observed_information",
            "whitened_SVD_parameter_correlations_covariance_stability", "latest_wave_time_support_and_residuals",
            "fixed_covariance_individual_core_score_contributions_aggregate_by_wave_no_refit", "Primary_to_SP01_beta_SE_CI"],
            "ELSA_additional": ["W8_recovery19_reference45", "W10_2021_2023_FI_rows272", "manuscript_time_consistency"],
            "influence_caveat": "screening_not_deletion_refit_or_variance_reestimation",
            "late_support_caveat": "晚期观察敏感性尚未充分排除"},
        "scope_limits": {"recovery_wave_interaction": False, "triple_interaction": False, "period_search": False,
            "primary_refit": False, "RS_NL_LM_combined": False, "meta": False, "automatic_PASS_WITH_CAVEAT": False,
            "period_heterogeneity_or_causal_claim": False, "freeze_in_stage12": False}})


def junit_evidence(path):
    root = ET.parse(path).getroot()
    suites = [root] if root.tag == "testsuite" else root.findall(".//testsuite")
    counts = {k: sum(int(s.attrib.get(k, 0)) for s in suites) for k in ("tests", "failures", "errors", "skipped")}
    checked(counts["tests"] > 0 and all(counts[k] == 0 for k in ("failures", "errors", "skipped")), "Stage1必要测试未通过")
    return {"path": str(Path(path).resolve()), "sha256": sha(path), **counts}


def verify_approval(approval, preflight):
    checked(approval.get("schema") == "shock_period_scientific_author_approval_v1", "科学批准schema错误")
    checked(approval.get("author_explicit_approval") is True and approval.get("scope") == "SPECIFICATION_LOCK_AND_STAGE2_CODE_PREPARATION_ONLY", "没有明确科学规格批准")
    checked(approval.get("scientific_specification_sha256") == preflight["scientific_specification_sha256"], "批准规格SHA不符")
    checked(approval.get("preflight_manifest_sha256") == preflight["_actual_sha256"], "批准Preflight SHA不符")
    checked(approval.get("stage2_real_fit_authorized") is False, "Stage1不能提前授权真实拟合")
    for key in ("original_instruction", "instruction_source", "recorded_utc"):
        checked(isinstance(approval.get(key), str) and bool(approval[key].strip()), "缺少作者批准来源：" + key)


def verify_preflight(path):
    m = read_json(path)
    checked(m["status"] == "READY_FOR_SPECIFICATION_LOCK" and m["real_model_fit_calls"] == 0, "准备状态错误")
    verify_records(m["sources"])
    verify_records(m["outputs"])
    checked(m["runtime"] == provenance()["runtime"], "准备环境漂移")
    checked(m["code"] == code_hashes(), "准备代码漂移")
    checked(content_sha(m["scientific_specification"]) == m["scientific_specification_sha256"], "规格内容漂移")
    m["_actual_sha256"] = sha(path)
    return m


def lock_spec(preflight_path, approval_path):
    checked(not CONTRACT.exists(), "拒绝覆盖既有合同")
    m = verify_preflight(preflight_path)
    a = read_json(approval_path)
    verify_approval(a, m)
    contract = {"schema": "shock_period_round1e_v1_contract", "status": "LOCKED",
        "scope": UNITS, "scientific_specification": m["scientific_specification"],
        "scientific_specification_sha256": m["scientific_specification_sha256"],
        "stage0": {"path": rel(STAGE0 / "stage0_manifest.json"), "sha256": STAGE0_SHA},
        "preflight": {"path": rel(preflight_path), "manifest_sha256": m["_actual_sha256"]},
        "sources": m["sources"], "code": m["code"], "runtime": m["runtime"],
        "author_specification_approval": {"path": rel(approval_path), "sha256": sha(approval_path)},
        "real_execution_authorization": "NOT_GRANTED_BY_STAGE1_REQUEST",
        "actual_result_approval": "NOT_GRANTED", "freeze_approval": "NOT_GRANTED", **provenance()}
    write_json(CONTRACT, contract)
    return {"status": "READY_FOR_STAGE2_AUTHORIZATION", "contract": rel(CONTRACT), "contract_sha256": sha(CONTRACT),
            "preflight_manifest_sha256": m["_actual_sha256"], "real_model_fit_calls": 0}


def verify_contract():
    c = read_json(CONTRACT)
    checked(c["status"] == "LOCKED" and c["scope"] == UNITS, "合同范围错误")
    checked(c["real_execution_authorization"] == "NOT_GRANTED_BY_STAGE1_REQUEST", "历史授权字段被改写")
    checked(c["code"] == code_hashes() and c["runtime"] == provenance()["runtime"], "代码或环境漂移")
    m = verify_preflight(ROOT / c["preflight"]["path"])
    checked(m["_actual_sha256"] == c["preflight"]["manifest_sha256"], "合同Preflight绑定变化")
    checked(c["scientific_specification"] == m["scientific_specification"], "合同规格变化")
    checked(sha(ROOT / c["author_specification_approval"]["path"]) == c["author_specification_approval"]["sha256"], "批准记录变化")
    verify_approval(read_json(ROOT / c["author_specification_approval"]["path"]), m)
    return c


def verify_authorization(contract, path):
    checked(path is not None, "真实拟合未授权，拒绝创建运行目录")
    a = read_json(path)
    checked(a.get("schema") == "shock_period_real_execution_authorization_v1" and a.get("authorization") == "GRANTED_FOR_THREE_SP01_MODELS", "执行授权schema或状态错误")
    checked(a.get("scope") == UNITS, "只能授权三个SP01单元")
    checked(a.get("contract_sha256") == sha(CONTRACT), "执行授权合同SHA不符")
    checked(a.get("preflight_manifest_sha256") == contract["preflight"]["manifest_sha256"], "执行授权Preflight SHA不符")
    for key in ("original_instruction", "instruction_source", "recorded_utc"):
        checked(isinstance(a.get(key), str) and bool(a[key].strip()), "缺少真实作者执行指令：" + key)
    return a
