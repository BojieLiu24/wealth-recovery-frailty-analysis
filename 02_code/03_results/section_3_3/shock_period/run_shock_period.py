"""Round 1E preflight、lock-spec与显式授权fit；没有全部阶段入口。"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import traceback

import pandas as pd

from sp_audit import no_models, clean, write_json, sha, rel, verify_records
from common import check_protected
from sp_contract import (ROOT, PROGRESS, PRIMARY, CONTRACT, COHORTS, UNITS, read_json,
    safe_id, source_records, prepare, scientific_spec, content_sha, code_hashes, junit_evidence,
    lock_spec, verify_contract, verify_authorization, checked, provenance, GateError)
from sp_fit import run_synthetic_validation, ExecutionPermit, fit_unit, core_estimate

RUNS = ROOT / "03_output/3.3 Robustness of the Main Findings/Shock Calendar Period/runs"


def preflight(preflight_id, test_path, baseline_path):
    safe_id(preflight_id)
    folder = PROGRESS / preflight_id
    checked(not folder.exists(), "拒绝覆盖准备验收")
    tests = junit_evidence(test_path)
    baseline = read_json(baseline_path)
    check_protected(baseline)
    # 纯数学优化先验证，不把真实数据交给任何优化器。
    synthetic = run_synthetic_validation()
    checked(synthetic["status"] == "PASS", "实际合成捕获验证未通过")
    counter = {"forbidden_calls": 0}
    with no_models(counter):
        sources, stage0_design = source_records()
        designs = {}
        for c in COHORTS:
            frame, x, design, *_ = prepare(c, stage0_design)
            designs[c] = design
        spec = scientific_spec(designs)
        spec_hash = content_sha(spec)
        verify_records(sources)
        check = check_protected(baseline)
        checked(counter["forbidden_calls"] == 0, "准备阶段有禁用调用")
        folder.mkdir(parents=True, exist_ok=False)
        write_json(folder / "synthetic_optimizer_validation.json", synthetic)
        write_json(folder / "design_readiness.json", {"cohorts": designs, "scientific_specification": spec,
            "scientific_specification_sha256": spec_hash, "real_model_fit_calls": 0})
        report = ["# Round 1E Stage 1准备验收", "", "状态：READY_FOR_SPECIFICATION_LOCK。真实模型拟合未授权。", "",
            f"必要测试{tests['tests']}项通过；本次独立合成优化调用{synthetic['synthetic_optimizer_calls']}次，真实LMM调用0。",
            "合成问题通过同一statsmodels底层分派、SciPy实际路径；捕获前后参数、目标值、梯度、状态、评估次数严格一致。",
            "正常结束、迭代限制、目标异常及异常后独立运行均通过；封装恢复已核对。",
            "合成问题不读取研究数据；研究数据仅用于同一行集与矩阵核验，没有真实优化。", "",
            "| 队列 | 人数 | FI行 | 列/秩 | 最晚波次恢复/参照人数 |",
            "|---|---:|---:|---:|---:|"]
        for c, d in designs.items():
            a = d["anchor"]
            groups = {g["group"]: g["participants"] for g in d["latest_wave_support"]}
            report.append(f"| {c} | {a['participants']} | {a['rows']} | {d['n_columns']}/{d['rank']} | {groups['substantial_recovery']}/{groups['persistent_or_limited_recovery']} |")
        report += ["", "三个最晚波次均只有landmark和一次后续FI；没有删人、合并波次或改原42列。",
            "Stage 3对三队列检查全部波次时间项精度、完整协方差/信息稳定性和核心估计影响，ELSA仍为重点。",
            "ELSA保留2021–2023年272行；本机S1一致不代表期刊上传身份已确认。",
            "SP01假定共同恢复年度速度差，不是时期异质性、危机效应或因果识别。",
            "HOLD结合独立原因码解释；优化未收敛不自动等于模型数学不可识别。",
            "科学批准绑定科学内容SHA及Preflight SHA；合同锁定后仍需新的三个单元真实拟合授权。",
            "Stage 0历史源码/审计、Primary、Round 1A–1D、稿件和已有冻结资产保持；不commit/push。", ""]
        with (folder / "readiness_report.md").open("x", encoding="utf-8", newline="\n") as f:
            f.write("\n".join(report))
        m = {"schema": "shock_period_preflight_v1", "preflight_id": preflight_id,
            "status": "READY_FOR_SPECIFICATION_LOCK", **provenance(), "sources": sources,
            "code": code_hashes(), "scientific_specification": spec,
            "scientific_specification_sha256": spec_hash, "tests": tests,
            "synthetic_optimizer_calls": synthetic["synthetic_optimizer_calls"], "real_model_fit_calls": 0,
            "forbidden_calls": counter["forbidden_calls"], "protected_check": check,
            "protected_hashes_before": baseline["protected_hashes"],
            "protected_hashes_after": baseline["protected_hashes"].copy(),
            "outputs": {rel(p): sha(p) for p in folder.iterdir() if p.is_file()},
            "real_execution_authorization": "NOT_GRANTED", "stage3_5_executed": False}
        check_protected(baseline)
        verify_records(sources)
        write_json(folder / "preflight_manifest.json", m)
    return {"status": m["status"], "preflight": rel(folder / "preflight_manifest.json"),
        "preflight_sha256": sha(folder / "preflight_manifest.json"), "scientific_specification_sha256": spec_hash,
        "synthetic_optimizer_calls": synthetic["synthetic_optimizer_calls"], "real_model_fit_calls": 0,
        "protected_files": check["files_checked"], "tests": tests["tests"]}


def fit(run_id, authorization_path):
    safe_id(run_id)
    # 无授权即拒绝；不能连建一个空结果目录都先于批准。
    checked(authorization_path is not None, "真实拟合未授权，拒绝创建运行目录")
    contract = verify_contract()
    auth = verify_authorization(contract, authorization_path)
    output = RUNS / run_id
    checked(not output.exists(), "拒绝覆盖已有真实运行")
    stage0_design = read_json(ROOT / "03_output/3.3 Robustness of the Main Findings/Shock Calendar Period/Audit/round1e_stage0_20261014_01/design_diagnostics.json")
    cached = {}
    counter = {"forbidden_calls": 0}
    with no_models(counter):
        for c in COHORTS:
            data = prepare(c, stage0_design)
            cached[c] = data
            expected = contract["scientific_specification"]["cohorts"][c]
            checked(data[2]["columns"] == expected["columns"] and data[2]["anchor"] == expected["anchor"], "执行设计或行集漂移")
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "execution_authorization.json", auth)
    permit = ExecutionPermit(contract, auth, sha(CONTRACT))
    results, rows = [], []
    def checkpoint(status):
        write_json(output / "execution_manifest.json", {"stage": 2, "status": status, "run_id": run_id,
            **provenance(), "contract_sha256": sha(CONTRACT), "authorization_sha256": sha(authorization_path),
            "preflight_manifest_sha256": contract["preflight"]["manifest_sha256"],
            "sources": contract["sources"], "code": contract["code"], "scope": UNITS,
            "units": [{"cohort": o["cohort"], "status": o["status"], "failure_reason_codes": o["failure_reason_codes"]} for o in results],
            "automatic_retry": False, "primary_refit": False, "stage3_5_executed": False,
            "outputs": {rel(p): sha(p) for p in output.rglob("*") if p.is_file() and p.name != "execution_manifest.json"}},
            replace_owned=(output / "execution_manifest.json").exists())
    checkpoint("RUNNING")
    for c in COHORTS:
        f, x, design, beta, cov, primary_diag, start = cached.pop(c)
        primary = core_estimate(beta, cov, design["primary_columns"])
        o = fit_unit(f, x, design, primary_diag, start, c, output / c / "SP01", permit)
        results.append(o)
        for name, estimate, state in (("Primary", primary, "FROZEN_REFERENCE"), ("SP01", o["core_estimate"], o["status"])):
            rows.append({"Cohort": c, "Specification": name, "Participants": design["anchor"]["participants"],
                "FI_observations": design["anchor"]["rows"], **(estimate or {}),
                "Basic_QA_status": state, "failure_reason": o["failure_reason"] if name == "SP01" else None,
                "Convergence": str(o["convergence"]) if name == "SP01" else "FROZEN_REFERENCE",
                "result_approved": False})
        checkpoint("RUNNING")
    verify_records(contract["sources"])
    with (output / "cohort_estimates.csv").open("x", encoding="utf-8", newline="") as f:
        pd.DataFrame(rows).to_csv(f, index=False)
    write_json(output / "model_diagnostics.json", {o["cohort"]: read_json(output / o["cohort"] / "SP01/diagnostics.json") for o in results})
    with (output / "stage2_execution_report.md").open("x", encoding="utf-8") as f:
        f.write("# Round 1E Stage 2执行报告\n\n基础数值状态不代表科学结果批准；停止于Stage 2。\n\n")
        for o in results:
            f.write(f"- {o['cohort']}/SP01：{o['status']}；原因码：{o['failure_reason_codes']}。\n")
        f.write("\n完整估计见cohort_estimates.csv；HOLD不作有效推断。波次调整不是时期异质性或因果识别。\n")
    checkpoint("COMPLETED_WITH_EXPLICIT_UNIT_STATES")
    return {"status": "COMPLETED_WITH_EXPLICIT_UNIT_STATES", "run": rel(output), "models": 3, "Stage3": "NOT_EXECUTED"}


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("preflight")
    p.add_argument("--preflight-id", required=True)
    p.add_argument("--test-evidence", type=Path, required=True)
    p.add_argument("--baseline", type=Path, required=True)
    p = commands.add_parser("lock-spec")
    p.add_argument("--preflight", type=Path, required=True)
    p.add_argument("--approval", type=Path, required=True)
    p = commands.add_parser("fit")
    p.add_argument("--run-id", required=True)
    p.add_argument("--authorization", type=Path)
    args = parser.parse_args()
    try:
        if args.command == "preflight":
            result = preflight(args.preflight_id, args.test_evidence, args.baseline)
        elif args.command == "lock-spec":
            result = lock_spec(args.preflight, args.approval)
        else:
            result = fit(args.run_id, args.authorization)
        import json
        print(json.dumps(clean(result), ensure_ascii=False, indent=2))
    except Exception as error:
        import json
        print(json.dumps({"status": "REFUSED_OR_FAILED", "failure_reason": getattr(error, "reason_code", "SOURCE_MISMATCH"),
            "message": str(error), "new_authorization_required": args.command == "fit"}, ensure_ascii=False))
        raise SystemExit(1)
