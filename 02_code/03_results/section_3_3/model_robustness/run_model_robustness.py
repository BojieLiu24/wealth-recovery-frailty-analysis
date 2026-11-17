"""分阶段入口；不提供all、freeze、release、present或meta命令。"""
from __future__ import annotations

import argparse
import json
import re
import traceback
from pathlib import Path

import numpy as np
import patsy

from common import (ROOT, CODE, PROGRESS, CONTRACT_PATH, RUNS, PANEL_FREEZE, COHORTS, SPECS,
                    read_json, write_json, write_csv, sha, rel, require, provenance, code_hashes,
                    check_protected, repo_path)
from specification import scientific_spec, KNOTS
from prepare_inputs import (verify_sources, load_cohort, model_design, reconstruct_primary, time_support,
                            classify_history, PRIMARY, CLOSEOUT)
from time_basis import scale_equivalence_checks, rcs_basis
from contrasts import prediction_vectors, contrast_vector_set, evaluate_all


def preflight(output):
    """真实输入只作核验与冻结参数重构；模型拟合模块不会被导入。"""
    require(output.resolve().is_relative_to((PROGRESS / "Stage0").resolve()), "Preflight输出必须位于Stage0")
    output.mkdir(parents=True, exist_ok=False)
    result = {"stage": 0, "status": "FAILED", "model_fit_calls": 0, "meta_calls": 0,
              "created_model_results": False, "scientific_specification": scientific_spec(), **provenance()}
    error = None
    try:
        source = verify_sources()
        source["inputs"][rel(PROGRESS / "Stage0/implementation_before.json")] = sha(PROGRESS / "Stage0/implementation_before.json")
        write_json(output / "historical_exceptions.json", source["historical_exceptions"])
        write_csv(output / "historical_exceptions.csv", [
            {**r, "current_candidates": json.dumps(r["current_candidates"], ensure_ascii=False)}
            for r in source["historical_exceptions"]])
        scale = scale_equivalence_checks()
        write_json(output / "rs_scale_equivalence.json", scale)
        cohorts = []
        primary_contrasts = []
        for cohort in COHORTS:
            frame, lm, x, beta, covariance, diag, anchor = load_cohort(cohort)
            designs = {}
            for spec in SPECS:
                _, designs[spec] = model_design(frame, x, cohort, spec)
            knots = KNOTS[cohort]
            grid = np.unique(np.r_[np.arange(0, float(frame.time_years.max())+.25, .25), knots])
            basis = np.column_stack([np.ones(len(grid)), grid, rcs_basis(grid, knots)])
            native = np.asarray(patsy.dmatrix("cr(t, knots=(k,), lower_bound=0, upper_bound=b) - 1",
                                             {"t": grid, "k": knots[1], "b": knots[2]}))
            q, _ = np.linalg.qr(basis)
            basis_error = float(np.max(np.abs(native-q @ (q.T @ native))))
            require(basis_error < 1e-10, "RCS与自然样条列空间不等价")
            reconstruction = reconstruct_primary(frame, x, beta, diag)
            vectors = contrast_vector_set(prediction_vectors(lm, x.design_info, "RS01", cohort))
            primary_contrasts.extend(evaluate_all(beta, covariance, vectors, cohort, "FROZEN_PRIMARY_REFERENCE"))
            cohorts.append({"cohort": cohort, "anchor": anchor, "designs": designs,
                            "knots": knots, "knots_hex": [float(v).hex() for v in knots],
                            "knot_source": {"panel_path": read_json(PANEL_FREEZE)["cohorts"][cohort]["parquet_path"],
                                            "panel_sha256": anchor["parquet_sha256"], "panel_freeze_sha256": sha(PANEL_FREEZE),
                                            "filter": "primary_complete_case_eligible=True; time_years>0",
                                            "quantiles": [.5, .9], "method": "numpy.quantile(method='linear')",
                                            "positive_time_rows": anchor["later"], "numpy": np.__version__},
                            "basis_space_max_abs_error": basis_error,
                            "primary_reconstruction": reconstruction, "time_support": time_support(frame)})
        write_csv(output / "frozen_primary_contrasts.csv", primary_contrasts)
        protection = check_protected(read_json(PROGRESS / "Stage0/implementation_before.json"))
        result.update({"status": "VALIDATED", "source_audit": source, "cohorts": cohorts,
                       "scale_equivalence": scale, "protected_assets": protection,
                       "next_stage": "SPECIFICATION_LOCK_ONLY; REAL_FIT_NOT_AUTHORIZED"})
    except Exception as exc:
        error = exc
        result["failure"] = type(exc).__name__ + ": " + str(exc)
        # 即使来源被阻断也记录历史例外，不把失败当成空白成功报告。
        if not (output / "historical_exceptions.json").exists():
            try:
                m = read_json(PRIMARY / "section_3_2_manifest.json")
                rows = classify_history({**m["outputs"], **m["protected_asset_hashes"]}, ROOT,
                                        read_json(ROOT / CLOSEOUT)["protected_hashes_after"])
                write_json(output / "historical_exceptions.json", rows)
            except Exception:
                pass
        (output / "failure_trace.txt").write_text(traceback.format_exc(), encoding="utf-8")
    write_json(output / "preflight_results.json", result)
    summary = "# Round 1B Stage0 Preflight\n\n"
    summary += f"状态：{result['status']}。真实模型拟合0，meta 0。\n\n"
    if result["status"] == "VALIDATED":
        source = result["source_audit"]
        summary += f"历史manifest核验{source['historical_items_checked']}项，{len(source['historical_exceptions'])}项非阻断历史例外；统计阻断0。\n\n"
        summary += "三队列保持冻结Primary行集、标准化和实际年数；RS01/NL01分别42/44列满秩。随机时间尺度合成核验及Primary残差重构通过。\n\n"
        summary += "| 队列 | 人数 | FI行 | 两次观察比例 | RCS结点 |\n|---|---:|---:|---:|---|\n"
        for c in result["cohorts"]:
            summary += f"| {c['cohort']} | {c['anchor']['participants']} | {c['anchor']['rows']} | {c['time_support']['exactly_two_observations_percent']:.2f}% | {c['knots']} |\n"
        summary += "\n标准化经验分布下的Primary对比仅由冻结参数推导，不是新的拟合结果。重构一致性不是原逐行残差的独立验证。\n\n"
        summary += "当前源码与历史运行源码仅meta函数不同；本模块不调用meta。Design整表历史SHA差异单列，必需locked决策逐条核验。\n\n"
        summary += "真实拟合授权仍未提供；Stage3完整QA、作者科学批准及Stage4/5尚未开始。\n"
    else:
        summary += result["failure"] + "\n"
    (output / "preflight_report.md").write_text(summary, encoding="utf-8")
    write_json(output / "manifest.json", {"schema": "round1b_preflight_v1", "status": result["status"],
               "code": code_hashes(), "inputs": result.get("source_audit", {}).get("inputs", {}),
               "outputs": {rel(p): sha(p) for p in sorted(output.iterdir()) if p.is_file()}, **provenance()})
    if error:
        raise error
    return {"status": result["status"], "path": rel(output), "model_fit_calls": 0}


def verify_preflight(folder):
    folder = Path(folder).resolve()
    require(folder.is_relative_to(PROGRESS / "Stage0"), "Preflight路径无效")
    manifest = read_json(folder / "manifest.json")
    require(manifest["status"] == "VALIDATED", "Preflight未通过")
    for p, h in {**manifest["inputs"], **manifest["outputs"], **manifest["code"]}.items():
        require(sha(repo_path(p)) == h, "Preflight来源或代码漂移：" + p)
    result = read_json(folder / "preflight_results.json")
    require(result["scientific_specification"] == scientific_spec(), "Preflight科学规格变化")
    require(result["model_fit_calls"] == 0 and result["status"] == "VALIDATED", "Preflight边界不符")
    return result, manifest


def lock_spec(folder):
    result, manifest = verify_preflight(folder)
    require(not CONTRACT_PATH.exists(), "拒绝覆盖既有Round1B合同")
    contract = {"schema": "model_robustness_round1b_v1", "status": "LOCKED",
                "specification_approval": "AUTHOR_APPROVED_FOR_CODE_AND_PREFLIGHT",
                "approval_basis": "作者明确要求PLEASE IMPLEMENT修订计划，同时保留本轮不拟合模型限制",
                "real_execution_authorization": "NOT_GRANTED_THIS_TURN",
                "scientific_specification": scientific_spec(), "inputs": manifest["inputs"], "code": manifest["code"],
                "preflight": {"path": rel(folder), "manifest_sha256": sha(folder / "manifest.json"),
                              "results_sha256": sha(folder / "preflight_results.json")},
                "cohort_preflight": result["cohorts"], "runtime": result["runtime"],
                "no_author_scientific_result_approval_yet": True, **provenance()}
    write_json(CONTRACT_PATH, contract)
    write_json(PROGRESS / "Stage1/specification_lock.json", {"status": "LOCKED_FOR_IMPLEMENTATION",
               "contract_path": rel(CONTRACT_PATH), "contract_sha256": sha(CONTRACT_PATH),
               "real_model_execution": "NOT_AUTHORIZED", "stages3_5": "NOT_STARTED"})
    return {"status": "LOCKED", "contract": rel(CONTRACT_PATH), "real_execution": "NOT_AUTHORIZED"}


def verify_execution_authorization(contract, path):
    require(path is not None, "本轮禁止真实拟合；需要作者另行明确授权记录")
    authorization = read_json(path)
    require(authorization.get("schema") == "round1b_real_execution_authorization_v1", "授权schema不符")
    require(authorization.get("authorization") == "GRANTED_FOR_SIX_ROUND1B_MODELS", "未获六模型真实执行授权")
    require(authorization.get("contract_sha256") == sha(CONTRACT_PATH), "授权未绑定当前锁定合同")
    require(authorization.get("preflight_manifest_sha256") == contract["preflight"]["manifest_sha256"], "授权未绑定Preflight")
    require(authorization.get("scope") == [f"{c}/{s}" for c in COHORTS for s in SPECS], "授权范围不是六个锁定单元")
    require(bool(authorization.get("human_instruction")) and bool(authorization.get("evidence_source")), "缺少真实作者指令及来源；不得由程序代作者批准")
    return authorization


def fit(run_id, authorization_path):
    contract = read_json(CONTRACT_PATH)
    require(contract["status"] == "LOCKED" and contract["scientific_specification"] == scientific_spec(), "执行合同或科学规格漂移")
    authorization = verify_execution_authorization(contract, authorization_path)
    require(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", run_id), "run_id必须为安全的单层名称")
    folder = repo_path(contract["preflight"]["path"])
    require(sha(folder / "manifest.json") == contract["preflight"]["manifest_sha256"], "合同Preflight哈希不符")
    verify_preflight(folder)
    require(provenance()["runtime"] == contract["runtime"], "锁定软件环境变化")
    source = verify_sources()
    source["inputs"][rel(PROGRESS / "Stage0/implementation_before.json")] = sha(PROGRESS / "Stage0/implementation_before.json")
    require(source["inputs"] == contract["inputs"], "统计输入或历史例外证据漂移")
    scale_equivalence_checks()
    # 独立统计来源已核验；保护基线取本次实际执行开始时，避免把作者先前稿件编辑误称本轮变更。
    template = read_json(PROGRESS / "Stage0/implementation_before.json")
    execution_before = {**template, **{"git_head": provenance()["git_commit"], "git_diff_sha256": provenance()["git_diff_sha256"]},
                        "protected_hashes": {name: sha(repo_path(name)) for name in template["protected_hashes"] if repo_path(name).is_file()}}
    output = RUNS / run_id
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "execution_authorization.json", authorization)
    write_json(output / "protected_assets_before.json", execution_before)
    write_json(output / "historical_exceptions_at_execution.json", source["historical_exceptions"])
    from fit_models import fit_unit
    from validate_results import validate_unit
    units = []
    comparison_rows = []
    for cohort in COHORTS:
        frame, lm, x, beta, covariance, diag, anchor = load_cohort(cohort)
        expected_anchor = next(r["anchor"] for r in contract["cohort_preflight"] if r["cohort"] == cohort)
        require(anchor == expected_anchor, "执行行集身份与Preflight不同")
        primary_vectors = contrast_vector_set(prediction_vectors(lm, x.design_info, "RS01", cohort))
        primary_average = next(r for r in evaluate_all(beta, covariance, primary_vectors, cohort, "Primary") if r["estimand_id"] == "A_0_8")
        comparison_rows.append({**primary_average, "source_kind": "FROZEN_PRIMARY_REFERENCE", "status": "FROZEN_REFERENCE",
                                "participants": anchor["participants"], "model_rows": anchor["rows"],
                                "se_ratio_to_primary": 1., "estimate_difference_from_primary": 0.})
        for spec in SPECS:
            model_x, design = model_design(frame, x, cohort, spec)
            vectors = contrast_vector_set(prediction_vectors(lm, x.design_info, spec, cohort))
            unit = output / ("Random Slope" if spec == "RS01" else "Nonlinear Time") / spec / cohort
            result = fit_unit(frame, model_x, design, anchor, cohort, spec, beta, diag, vectors, unit, authorization)
            average = next((r for r in result.get("contrasts", []) if r["estimand_id"] == "A_0_8"), None) if result["status"] != "HOLD_NOT_ESTIMABLE" else None
            comparison_rows.append({"cohort": cohort, "spec_id": spec, "estimand_id": "A_0_8", "units": "FI units/year",
                                    "source_kind": "NEW_LMM_FIT", "participants": anchor["participants"], "model_rows": anchor["rows"],
                                    **(average or {"estimate": None, "se": None, "ci_lower": None, "ci_upper": None}),
                                    "status": result["status"], "se_ratio_to_primary": average["se"]/primary_average["se"] if average else None,
                                    "estimate_difference_from_primary": average["estimate"]-primary_average["estimate"] if average else None})
            validation = validate_unit(unit, vectors)
            write_json(unit / "file_validation.json", validation)
            units.append({"cohort": cohort, "spec_id": spec, "status": result["status"], "path": rel(unit)})
            write_json(output / "run_manifest.json", {"stage": 2, "units": units, "status": "WORKING_UNFROZEN",
                       "contract_sha256": sha(CONTRACT_PATH), "inputs": contract["inputs"], "code": contract["code"],
                       "authorization_sha256": sha(authorization_path), "automatic_specification_change": False,
                       "meta_executed": False, "stage3_complete": False,
                       "outputs": {rel(p): sha(p) for p in sorted(output.rglob("*")) if p.is_file() and p.name != "run_manifest.json"},
                       **provenance()}, replace_owned=(output / "run_manifest.json").exists())
    comparison_dir = output / "Comparisons"
    comparison_dir.mkdir(exist_ok=False)
    write_csv(comparison_dir / "primary_comparison_long.csv", comparison_rows)
    write_json(output / "protected_assets_validation.json", check_protected(execution_before))
    manifest = read_json(output / "run_manifest.json")
    manifest.update({"status": "STAGE2_STOPPED_FOR_STAGE3_REVIEW",
                     "outputs": {rel(p): sha(p) for p in sorted(output.rglob("*")) if p.is_file() and p.name != "run_manifest.json"}})
    write_json(output / "run_manifest.json", manifest, replace_owned=True)
    return {"status": "STAGE2_STOPPED_FOR_STAGE3_REVIEW", "units": units}


def main():
    parser = argparse.ArgumentParser(description="Round1B Stage0–2分阶段执行；真实拟合需另行授权")
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("preflight")
    p.add_argument("--output", type=Path, default=PROGRESS / "Stage0/preflight_v2")
    p = commands.add_parser("lock-spec")
    p.add_argument("--preflight", type=Path, default=PROGRESS / "Stage0/preflight_v2")
    p = commands.add_parser("fit")
    p.add_argument("--run-id", required=True)
    p.add_argument("--authorization", type=Path)
    args = parser.parse_args()
    result = preflight(args.output) if args.command == "preflight" else (lock_spec(args.preflight) if args.command == "lock-spec" else fit(args.run_id, args.authorization))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
