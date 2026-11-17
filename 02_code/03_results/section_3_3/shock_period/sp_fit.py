"""SP01原生优化证据捕获、基础数值分类及获授权单元执行。"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import inspect
import time
import traceback
import warnings

import numpy as np
import scipy.optimize._optimize as bfgs_native
import scipy.optimize._lbfgsb_py as lbfgs_native
import statsmodels.api as sm
import statsmodels.base.optimizer as backend
from statsmodels.regression.mixed_linear_model import MixedLMParams

from sp_audit import clean, write_json, sha, require, INTERPRETATION
from basic_qa import extract_numerical

POLICY = {"maxiter": 2000, "bfgs_gtol": 1e-8, "lbfgs_pgtol": 1e-10, "lbfgs_factr": 100.,
    "loglik_atol": 1e-4, "beta_atol": 1e-6, "se_rtol": 1e-3,
    "gradient_max": 1e-3, "primary_reconstruction_atol": 1e-10,
    "boundary_variance_ratio": 1e-6}
REASONS = ("SOURCE_MISMATCH", "DESIGN_RANK_DEFICIENT", "INFERENCE_UNAVAILABLE",
    "OPTIMIZER_NOT_CONVERGED", "OPTIMIZER_DISAGREEMENT", "NUMERICAL_QA_FAILED",
    "TERMINATION_LOG_INCOMPLETE", "BOUNDARY_REQUIRES_REVIEW")
TERM = "recovery:time_years"
Z95 = 1.959963984540054


def safe_values(value):
    """只保存参数与函数身份，捕获过程中不额外求值目标函数。"""
    if callable(value):
        return {"callable": getattr(value, "__module__", "") + "." + getattr(value, "__qualname__", type(value).__name__)}
    if isinstance(value, dict):
        return {str(k): safe_values(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe_values(v) for v in value]
    if isinstance(value, np.ndarray):
        return safe_values(value.tolist())
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return "NaN" if np.isnan(value) else "Infinity" if value > 0 else "-Infinity"
    return clean(value)


def bound_values(function, args, kwargs):
    bound = inspect.signature(function).bind(*args, **kwargs)
    bound.apply_defaults()
    return safe_values(bound.arguments)


@contextmanager
def capture_optimizers(events, *, research_policy=True):
    """局部捕获同一原生路径；两个层次的返回值均原样传回。"""
    originals = {"bfgs": backend._fit_bfgs, "lbfgs": backend._fit_lbfgs,
        "bfgs_native": bfgs_native._minimize_bfgs,
        "lbfgs_native": lbfgs_native._minimize_lbfgsb,
        "lbfgs_public": backend.optimize.fmin_l_bfgs_b}

    def internal(method, original):
        def wrapped(*args, **kwargs):
            event = {"kind": "scipy_internal", "optimizer": method,
                     "effective_arguments": bound_values(original, args, kwargs), "completed": False}
            events.append(event)
            try:
                result = original(*args, **kwargs)
                event.update(completed=True, raw_result=safe_values(dict(result)),
                    status=result.get("status"), message=result.get("message"),
                    task=None, task_availability="NOT_PROVIDED_BY_THIS_INTERFACE")
                return result
            except Exception as error:
                event["exception"] = {"type": type(error).__name__, "message": str(error)}
                raise
        return wrapped

    def public_lbfgs(*args, **kwargs):
        original = originals["lbfgs_public"]
        event = {"kind": "scipy_public", "optimizer": "lbfgs",
                 "effective_arguments": bound_values(original, args, kwargs), "completed": False}
        events.append(event)
        try:
            returned = original(*args, **kwargs)
            event.update(completed=True, raw_x=safe_values(returned[0]), raw_fun=safe_values(returned[1]),
                         raw_information=safe_values(returned[2]), task=returned[2].get("task"))
            return returned
        except Exception as error:
            event["exception"] = {"type": type(error).__name__, "message": str(error)}
            raise

    def dispatch(method):
        original = originals[method]
        def wrapped(f, score, start_params, fargs, kwargs, *args, **kw):
            effective = dict(kwargs)
            if method == "bfgs":
                require(effective.get("gtol") == POLICY["bfgs_gtol"], "BFGS有效精度不匹配")
            else:
                effective.update(pgtol=POLICY["lbfgs_pgtol"], factr=POLICY["lbfgs_factr"])
            bound = inspect.signature(original).bind(f, score, start_params, fargs, effective, *args, **kw)
            bound.apply_defaults()
            if research_policy:
                require(bound.arguments["maxiter"] == POLICY["maxiter"], "真实拟合maxiter不匹配")
            events.append({"kind": "statsmodels_dispatch", "optimizer": method,
                           "effective_arguments": safe_values(bound.arguments)})
            return original(f, score, start_params, fargs, effective, *args, **kw)
        return wrapped

    try:
        bfgs_native._minimize_bfgs = internal("bfgs", originals["bfgs_native"])
        lbfgs_native._minimize_lbfgsb = internal("lbfgs", originals["lbfgs_native"])
        backend.optimize.fmin_l_bfgs_b = public_lbfgs
        backend._fit_bfgs = dispatch("bfgs")
        backend._fit_lbfgs = dispatch("lbfgs")
        yield
    finally:
        backend._fit_bfgs = originals["bfgs"]
        backend._fit_lbfgs = originals["lbfgs"]
        bfgs_native._minimize_bfgs = originals["bfgs_native"]
        lbfgs_native._minimize_lbfgsb = originals["lbfgs_native"]
        backend.optimize.fmin_l_bfgs_b = originals["lbfgs_public"]


def terminal_evidence(events, method):
    inner = [e for e in events if e["kind"] == "scipy_internal" and e["optimizer"] == method]
    dispatch = [e for e in events if e["kind"] == "statsmodels_dispatch" and e["optimizer"] == method]
    complete = len(inner) == len(dispatch) == 1 and inner[0].get("completed", False)
    status, message, task = None, None, None
    if len(inner) == 1:
        status, message = inner[0].get("status"), inner[0].get("message")
        complete = complete and status is not None and isinstance(message, str) and bool(message)
    if method == "lbfgs":
        public = [e for e in events if e["kind"] == "scipy_public" and e["optimizer"] == method]
        complete = complete and len(public) == 1 and public[0].get("completed", False)
        if len(public) == 1:
            task = public[0].get("task")
        complete = complete and task is not None
    native_exception = inner[0].get("exception") if len(inner) == 1 else None
    exception_logged = len(dispatch) == len(inner) == 1 and native_exception is not None
    return {"complete": bool(complete or exception_logged), "status": status, "message": message, "task": task,
            "native_result_returned": bool(inner and inner[0].get("completed")), "native_exception": native_exception,
            "task_availability": "NOT_RETURNED_DUE_EXCEPTION" if method == "lbfgs" and exception_logged else
                "PROVIDED" if method == "lbfgs" else "NOT_PROVIDED_BY_BFGS_INTERFACE"}


def model_result_failure(log):
    """原生优化已成功而结果后处理失败时，不能称优化器未收敛。"""
    terminal = log["terminal"]
    codes = ["INFERENCE_UNAVAILABLE" if terminal.get("status") == 0 and terminal.get("native_result_returned")
             else "OPTIMIZER_NOT_CONVERGED"]
    if not terminal.get("complete"):
        codes.append("TERMINATION_LOG_INCOMPLETE")
    return reason_state(codes, {"native_terminal": terminal, "result_postprocessing_error": log.get("error")})


def run_synthetic_validation():
    """独立纯数学问题真实运行两个底层优化器，不读取数据或构造LMM。"""
    matrix = np.array([[4., .7], [.7, 2.]])
    center, start = np.array([.35, -.65]), np.array([3.2, 2.4])
    records, calls = [], 0
    def fun(x):
        d = x - center
        return float(.5 * d @ matrix @ d)
    def score(x):
        return matrix @ (x - center)
    def fail_fun(x):
        raise RuntimeError("SYNTHETIC_OBJECTIVE_EXCEPTION")
    baseline_functions = (backend._fit_bfgs, backend._fit_lbfgs, bfgs_native._minimize_bfgs,
                          lbfgs_native._minimize_lbfgsb, backend.optimize.fmin_l_bfgs_b)
    def current_functions():
        return (backend._fit_bfgs, backend._fit_lbfgs, bfgs_native._minimize_bfgs,
                lbfgs_native._minimize_lbfgsb, backend.optimize.fmin_l_bfgs_b)
    def invoke(method, iterations, f=fun):
        nonlocal calls
        calls += 1
        kwargs = {"gtol": POLICY["bfgs_gtol"]} if method == "bfgs" else {
            "pgtol": POLICY["lbfgs_pgtol"], "factr": POLICY["lbfgs_factr"]}
        return getattr(backend, "_fit_" + method)(f, score, start.copy(), (), kwargs,
            disp=False, maxiter=iterations, full_output=True)
    for method in ("bfgs", "lbfgs"):
        for iterations, scenario in ((2000, "normal"), (1, "iteration_limit")):
            baseline = invoke(method, iterations)
            events = []
            with capture_optimizers(events, research_policy=False):
                captured = invoke(method, iterations)
            require(current_functions() == baseline_functions, "合成验证后封装未恢复")
            np.testing.assert_array_equal(baseline[0], captured[0])
            require(baseline[1].keys() == captured[1].keys(), "捕获改变返回字段")
            for key in baseline[1]:
                np.testing.assert_array_equal(baseline[1][key], captured[1][key])
            t = terminal_evidence(events, method)
            require(t["complete"], "合成问题准确终止日志不完整")
            if scenario == "normal":
                require(t["status"] == 0, "合成正常优化未成功")
            else:
                require(t["status"] != 0, "未触发预定非正常完成")
            native = next(e for e in events if e["kind"] == "scipy_internal")
            if method == "lbfgs":
                require(native["effective_arguments"]["gtol"] == POLICY["lbfgs_pgtol"], "pgtol未传到底层")
                require(native["effective_arguments"]["ftol"] == POLICY["lbfgs_factr"] * np.finfo(float).eps, "factr/ftol不匹配")
            else:
                require(native["effective_arguments"]["gtol"] == POLICY["bfgs_gtol"], "gtol未传到底层")
            records.append({"optimizer": method, "scenario": scenario, "result_unchanged": True,
                "baseline_parameters": baseline[0].tolist(), "baseline_retvals": clean(baseline[1]),
                "terminal": t, "events": events, "interfaces_restored": True})
        for captured in (False, True):
            events = []
            try:
                if captured:
                    with capture_optimizers(events, research_policy=False):
                        invoke(method, 2000, fail_fun)
                else:
                    invoke(method, 2000, fail_fun)
            except RuntimeError as error:
                require(str(error) == "SYNTHETIC_OBJECTIVE_EXCEPTION", "异常内容被封装改变")
            else:
                raise ValueError("合成异常没有抛出")
            require(current_functions() == baseline_functions, "异常后封装未恢复")
            records.append({"optimizer": method, "scenario": "objective_exception", "captured": captured,
                "error": "SYNTHETIC_OBJECTIVE_EXCEPTION", "events": events, "interfaces_restored": True})
        again = invoke(method, 2000)
        reference = next(r for r in records if r["optimizer"] == method and r["scenario"] == "normal")
        np.testing.assert_array_equal(again[0], reference["baseline_parameters"])
        for key in again[1]:
            np.testing.assert_array_equal(again[1][key], reference["baseline_retvals"][key])
        records.append({"optimizer": method, "scenario": "post_exception_independent_run", "result_unchanged": True})
    return clean({"status": "PASS", "problem": "deterministic_positive_definite_quadratic_2D",
        "matrix": matrix, "center": center, "start": start, "strict_array_equality": True,
        "research_data_read": False, "real_model_fit_calls": 0, "synthetic_optimizer_calls": calls,
        "baseline_and_captured_calls_included": True, "records": records})


def core_estimate(beta, covariance, names):
    i = names.index(TERM)
    variance = float(covariance[i, i])
    require(np.isfinite(beta[i]) and np.isfinite(variance) and variance > 0, "核心推断不可用")
    se = float(np.sqrt(variance))
    from scipy.stats import norm
    return {"term": TERM, "estimate": float(beta[i]), "se": se,
        "ci_lower": float(beta[i] - Z95 * se), "ci_upper": float(beta[i] + Z95 * se),
        "p_value": float(2 * norm.sf(abs(beta[i]) / se)), "units": "FI units/year",
        "direction": "substantial_recovery_minus_persistent_or_limited_recovery"}


def reason_state(codes, details=None):
    ordered = [r for r in REASONS if r in codes]
    require(set(codes).issubset(REASONS), "未知失败原因码")
    status = "HOLD_NOT_ESTIMABLE" if any(r != "BOUNDARY_REQUIRES_REVIEW" for r in ordered) else (
        "BOUNDARY_REVIEW" if ordered else "PASS")
    return {"status": status, "failure_reason": ordered[0] if ordered else None,
        "failure_reason_codes": ordered, "reason_details": details or {},
        "stage3_complete": False, "release_eligible": False, "author_scientific_review": "PENDING",
        "HOLD_is_not_automatic_mathematical_nonidentifiability_claim": True}


def classify(diagnostics, checks, terminal_complete):
    codes, details = [], {}
    if not diagnostics.get("converged") or not checks.get("bfgs_converged") or not checks.get("lbfgs_converged"):
        codes.append("OPTIMIZER_NOT_CONVERGED")
    fixed = diagnostics.get("fixed_covariance_spectrum", {})
    if not fixed.get("symmetric") or fixed.get("eigenvalues") is None or min(fixed["eigenvalues"]) <= 0:
        codes.append("INFERENCE_UNAVAILABLE")
    for key, threshold, exclusive in (("loglik_absolute_difference", 1e-4, True),
            ("core_beta_absolute_difference", 1e-6, True), ("core_se_relative_difference", 1e-3, False)):
        value = checks.get(key, np.nan)
        if not np.isfinite(value) or (value >= threshold if exclusive else value > threshold):
            codes.append("OPTIMIZER_DISAGREEMENT")
        details[key] = {"value": clean(value), "threshold": threshold, "strict": exclusive}
    sigma = diagnostics.get("residual_variance", np.nan)
    g = diagnostics.get("random_spectrum", {})
    if not np.isfinite(sigma) or sigma <= 0 or not np.isfinite(diagnostics.get("log_likelihood", np.nan)) or not diagnostics.get("score_finite"):
        codes.append("NUMERICAL_QA_FAILED")
    if not g.get("symmetric") or g.get("eigenvalues") is None or min(g["eigenvalues"]) < 0:
        codes.append("NUMERICAL_QA_FAILED")
    if diagnostics.get("boundary_candidate"):
        codes.append("BOUNDARY_REQUIRES_REVIEW")
        details["boundary"] = diagnostics.get("boundary_kkt")
    elif not diagnostics.get("balanced_information_cholesky") or diagnostics.get("scaled_gradient_max") is None or not np.isfinite(diagnostics["scaled_gradient_max"]) or diagnostics["scaled_gradient_max"] > POLICY["gradient_max"]:
        codes.append("NUMERICAL_QA_FAILED")
    if not terminal_complete:
        codes.append("TERMINATION_LOG_INCOMPLETE")
    return reason_state(codes, details)


@dataclass(frozen=True)
class ExecutionPermit:
    contract: dict
    authorization: dict
    contract_sha256: str


def optimize_model(model, method, start):
    events, result, error = [], None, None
    began = time.perf_counter()
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        try:
            with capture_optimizers(events):
                result = model.fit(start_params=start, reml=False, method=method, full_output=True,
                    maxiter=POLICY["maxiter"], disp=False,
                    **({"gtol": POLICY["bfgs_gtol"]} if method == "bfgs" else {}))
        except Exception as exc:
            error = {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
    log = {"optimizer": method, "events": events, "terminal": terminal_evidence(events, method),
        "error": error, "elapsed_seconds": time.perf_counter() - began,
        "warnings": [{"category": w.category.__name__, "message": str(w.message)} for w in captured]}
    if result is not None:
        log.update(converged=bool(result.converged), log_likelihood=float(result.llf),
                   optimization_history=clean(getattr(result, "hist", None)))
    return result, clean(log)


def save_candidate(folder, method, result):
    """即刻保留候选，不因后一个优化器失败丢弃前一个结果。"""
    if result is None:
        return
    path = folder / (method + "_candidate.npz")
    with path.open("xb") as f:
        np.savez_compressed(f, fixed_effect_names=np.asarray(result.fe_params.index),
            fixed_effects=np.asarray(result.fe_params), parameters=np.asarray(result.params),
            parameter_names=np.asarray(result.params.index), random_covariance=np.asarray(result.cov_re),
            random_covariance_unscaled=np.asarray(result.cov_re_unscaled), residual_variance=np.asarray(result.scale))


def fit_unit(frame, x, design, primary_diag, start_beta, cohort, folder, permit):
    from sp_contract import CONTRACT, UNITS
    require(isinstance(permit, ExecutionPermit) and permit.contract_sha256 == sha(CONTRACT), "执行许可不匹配")
    require(permit.authorization.get("scope") == UNITS and f"{cohort}/SP01" in UNITS, "超出三个SP01范围")
    folder.mkdir(parents=True, exist_ok=False)
    write_json(folder / "sample_design.json", design)
    logs, diag, state = [], {}, reason_state(["INFERENCE_UNAVAILABLE"])
    result = {"cohort": cohort, "spec_id": "SP01", "interpretation": INTERPRETATION,
        "sample": design["anchor"], "core_estimate": None, "participant_records_saved": False}
    try:
        require(list(x) == permit.contract["scientific_specification"]["cohorts"][cohort]["columns"], "执行列漂移")
        model = sm.MixedLM(frame.fi_primary.astype(float), x.astype(float), frame.person_id.to_numpy(),
                           exog_re=np.ones((len(frame), 1)), use_sqrt=True)
        q = primary_diag["random_intercept_variance"] / primary_diag["residual_variance"]
        start = MixedLMParams.from_components(fe_params=start_beta, cov_re=np.array([[q]]))
        write_json(folder / "initial_parameters.json", {"fixed_effects": start_beta,
            "random_covariance_unscaled": [[q]], "fixed_effects_profiled": True,
            "optimizer_start_packed_cholesky_no_fixed_effects": start.get_packed(use_sqrt=True, has_fe=False)})
        first, log = optimize_model(model, "bfgs", start)
        logs.append(log); save_candidate(folder, "bfgs", first)
        write_json(folder / "bfgs_optimization.json", log)
        if first is None:
            state = model_result_failure(log)
        else:
            final, log = optimize_model(model, "lbfgs", first.params_object.copy())
            logs.append(log); save_candidate(folder, "lbfgs", final)
            write_json(folder / "lbfgs_optimization.json", log)
            if final is None:
                state = model_result_failure(log)
            else:
                beta, covariance, diag = extract_numerical(model, final)
                with (folder / "model_parameters.npz").open("xb") as f:
                    np.savez_compressed(f, fixed_effect_names=np.asarray(list(x)), fixed_effects=beta,
                        fixed_effect_covariance=covariance, hessian=np.asarray(diag["hessian"]),
                        score=np.asarray(diag["score_direct_unscaled_covariance"]))
                first_beta, first_cov, first_diag = extract_numerical(model, first)
                diag["bfgs_full_diagnostics"] = first_diag
                with (folder / "bfgs_numerics.npz").open("xb") as f:
                    np.savez_compressed(f, fixed_effects=first_beta, fixed_effect_covariance=first_cov,
                        hessian=np.asarray(first_diag["hessian"]), score=np.asarray(first_diag["score_direct_unscaled_covariance"]))
                a = core_estimate(first_beta, first_cov, list(x))
                b = core_estimate(beta, covariance, list(x))
                checks = {"bfgs_converged": bool(first.converged), "lbfgs_converged": bool(final.converged),
                    "loglik_absolute_difference": abs(float(first.llf-final.llf)),
                    "core_beta_absolute_difference": abs(a["estimate"]-b["estimate"]),
                    "core_se_relative_difference": abs(a["se"]-b["se"])/a["se"]}
                state = classify(diag, checks, all(l["terminal"]["complete"] for l in logs))
                diag.update(optimizer_checks=checks, bfgs_full_diagnostics=first_diag)
                result.update(fixed_effect_names=list(x), fixed_effects=beta.tolist(),
                    fixed_effect_covariance=covariance.tolist(), random_intercept_variance=float(final.cov_re.iloc[0, 0]),
                    residual_variance=float(final.scale))
                if state["status"] != "HOLD_NOT_ESTIMABLE":
                    result["core_estimate"] = b
                fixed_rows = []
                for i, name in enumerate(x):
                    se = float(np.sqrt(covariance[i, i])) if covariance[i, i] > 0 else None
                    fixed_rows.append({"term": name, "estimate": float(beta[i]), "se": se,
                        "ci_lower": float(beta[i]-Z95*se) if se is not None else None,
                        "ci_upper": float(beta[i]+Z95*se) if se is not None else None,
                        "role": "core_common_recovery_slope" if name == TERM else "adjustment_parameter",
                        "stage2_status": state["status"], "publication_eligible": False})
                import pandas as pd
                with (folder / "fixed_effects.csv").open("x", encoding="utf-8", newline="") as f:
                    pd.DataFrame(fixed_rows).to_csv(f, index=False)
    except Exception as error:
        codes = [getattr(error, "reason_code", "INFERENCE_UNAVAILABLE")]
        if any(l.get("converged") is False for l in logs):
            codes.append("OPTIMIZER_NOT_CONVERGED")
        if logs and not all(l["terminal"]["complete"] for l in logs):
            codes.append("TERMINATION_LOG_INCOMPLETE")
        state = reason_state(codes, {"exception": str(error)})
        with (folder / "failure_trace.txt").open("x", encoding="utf-8") as f:
            f.write(traceback.format_exc())
    result.update(state)
    result["convergence"] = {l["optimizer"]: l.get("converged") for l in logs}
    write_json(folder / "optimization_log.json", logs)
    write_json(folder / "diagnostics.json", {**diag, **state})
    write_json(folder / "result_object.json", result)
    return result
