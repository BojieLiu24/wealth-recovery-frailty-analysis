"""作者批准的两单元数值精度补充；保留首次结果及全部科学规格。"""
from pathlib import Path
import sys
import json
import warnings
import time
from contextlib import contextmanager

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))
import numpy as np
import statsmodels.base.optimizer as backend_optimizer
import fit_models
from common import (ROOT, RUNS, PROGRESS, CONTRACT_PATH, read_json, write_json, sha, rel,
                    require, clean, provenance, check_protected)
from run_model_robustness import verify_preflight
from prepare_inputs import verify_sources, load_cohort, model_design
from contrasts import prediction_vectors, contrast_vector_set
from validate_results import validate_unit

UNITS = [("HRS", "RS01"), ("KLoSA", "NL01")]
SETTINGS = {"bfgs": {"gtol": 1e-8}, "lbfgs": {"pgtol": 1e-10, "factr": 100.}, "maxiter": 2000}


@contextmanager
def strict_lbfgs_adapter(evidence):
    """MixedLM过滤pgtol/factr；在当前进程适配原低层函数，不修改库文件。"""
    original = backend_optimizer._fit_lbfgs

    def adapted(f, score, start_params, fargs, kwargs, *args, **kw):
        effective = {**kwargs, **SETTINGS["lbfgs"]}
        evidence.append({"effective_kwargs": clean(effective), "adapter": "original_statsmodels_low_level_lbfgs"})
        return original(f, score, start_params, fargs, effective, *args, **kw)

    backend_optimizer._fit_lbfgs = adapted
    try:
        yield
    finally:
        backend_optimizer._fit_lbfgs = original


def optimize_precision(model, method, start):
    began = time.perf_counter()
    result, error, effective = None, None, []
    with warnings.catch_warnings(record=True) as records:
        warnings.simplefilter("always")
        try:
            if method == "bfgs":
                result = model.fit(start_params=start, reml=False, method=method, maxiter=2000,
                                   full_output=True, disp=False, gtol=1e-8)
            else:
                with strict_lbfgs_adapter(effective):
                    result = model.fit(start_params=start, reml=False, method=method,
                                       maxiter=2000, full_output=True, disp=False)
                require(len(effective) == 1, "更严格L-BFGS设置未实际传递")
        except Exception as exc:
            error = type(exc).__name__ + ": " + str(exc)
    log = {"optimizer": method, "elapsed_seconds": time.perf_counter()-began,
           "error": error, "requested_settings": SETTINGS[method], "maxiter": 2000,
           "effective_low_level_settings": effective,
           "warnings": [{"category": w.category.__name__, "message": str(w.message)} for w in records]}
    if result is not None:
        log.update({"converged": bool(result.converged), "log_likelihood": float(result.llf),
                    "parameters": np.asarray(result.params).tolist(), "parameter_names": list(result.params.index),
                    "optimization_history": getattr(result, "hist", None)})
    return result, clean(log)


def main():
    old = RUNS / "round1b_stage2_20261014_01"
    output = RUNS / "round1b_stage2_20261014_precision01"
    progress = PROGRESS / "Stage2/round1b_stage2_20261014_precision01"
    contract = read_json(CONTRACT_PATH)
    verify_preflight(ROOT / contract["preflight"]["path"])
    sources = verify_sources()
    sources["inputs"][rel(PROGRESS / "Stage0/implementation_before.json")] = sha(PROGRESS / "Stage0/implementation_before.json")
    require(sources["inputs"] == contract["inputs"], "数值补充前统计来源变化")
    require(provenance()["runtime"] == contract["runtime"], "环境变化")
    require(all(read_json(old/("Random Slope" if s == "RS01" else "Nonlinear Time")/s/c/"result_object.json")["status"] == "HOLD_NOT_ESTIMABLE" for c, s in UNITS), "补充范围不符合原HOLD单元")
    output.mkdir(parents=True, exist_ok=False)
    progress.mkdir(parents=True, exist_ok=False)
    authorization = {"schema": "round1b_numerical_precision_addendum_v1",
                     "authorization": "GRANTED_FOR_SIX_ROUND1B_MODELS",
                     "actual_scope": [f"{c}/{s}" for c, s in UNITS],
                     "human_instruction": "批准两单元数值精度补充",
                     "evidence_source": "当前聊天对两单元更严格BFGS/L-BFGS设置的作者明确回复",
                     "contract_sha256": sha(CONTRACT_PATH), "settings": SETTINGS,
                     "scientific_specification_changed": False, "old_results_preserved": True,
                     "qa_thresholds_changed": False, "initial_run": rel(old), **provenance()}
    write_json(progress/"authorization.json", authorization)
    write_json(output/"execution_authorization.json", authorization)
    template = read_json(old/"protected_assets_before.json")
    protection = {**template, "protected_hashes": {p: sha(ROOT/p) for p in template["protected_hashes"]}}
    old_hashes = {rel(p): sha(p) for p in old.rglob("*") if p.is_file()}
    write_json(output/"protected_assets_before.json", protection)
    records = []
    original_optimize = fit_models.optimize
    fit_models.optimize = optimize_precision
    try:
        for cohort, spec in UNITS:
            print(f"START {cohort}/{spec}", flush=True)
            frame, lm, x, beta, covariance, diag, anchor = load_cohort(cohort)
            expected = next(r["anchor"] for r in contract["cohort_preflight"] if r["cohort"] == cohort)
            require(anchor == expected, "补充样本行集变化")
            design_x, design = model_design(frame, x, cohort, spec)
            vectors = contrast_vector_set(prediction_vectors(lm, x.design_info, spec, cohort))
            folder = output/("Random Slope" if spec == "RS01" else "Nonlinear Time")/spec/cohort
            obj = fit_models.fit_unit(frame, design_x, design, anchor, cohort, spec, beta, diag, vectors, folder, authorization)
            validation = validate_unit(folder, vectors)
            write_json(folder/"file_validation.json", validation)
            records.append({"cohort": cohort, "spec_id": spec, "path": rel(folder), "status": obj["status"],
                            "supersedes_for_active_reading": rel(old/("Random Slope" if spec == "RS01" else "Nonlinear Time")/spec/cohort)})
            print(f"FINISH {cohort}/{spec}: {obj['status']}", flush=True)
    finally:
        fit_models.optimize = original_optimize
    for p, h in old_hashes.items():
        require(sha(ROOT/p) == h, "首次运行资产发生变化")
    write_json(output/"protected_assets_validation.json", check_protected(protection))
    write_json(output/"run_manifest.json", {"stage": 2, "status": "NUMERICAL_ADDENDUM_COMPLETE_UNFROZEN",
               "units": records, "settings": SETTINGS, "source_code": {rel(Path(__file__)): sha(__file__)},
               "locked_model_code": contract["code"], "inputs": contract["inputs"],
               "initial_run_hashes": old_hashes, "scientific_models_added": 0, "refitted_existing_units": 2,
               "author_scientific_result_approval": "PENDING", "meta_executed": False,
               "outputs": {rel(p): sha(p) for p in output.rglob("*") if p.is_file()}, **provenance()})
    print(json.dumps(records, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
