"""授权后的六个拟合单元；本次实现不运行该入口。"""
from __future__ import annotations

import time
import traceback
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import norm
from statsmodels.regression.mixed_linear_model import MixedLMParams

from common import write_json, write_csv, clean, require, TERM, Z95
from specification import POLICY
from time_basis import random_design, covariance_to_years
from contrasts import evaluate_all, optimizer_contrast_checks, nonlinear_tests
from basic_qa import extract_numerical, classify_numerical


def optimize(model, method, start):
    """仅在授权入口内调用；逐优化器独立保存warning和历史。"""
    began = time.perf_counter()
    result, error = None, None
    with warnings.catch_warnings(record=True) as records:
        warnings.simplefilter("always")
        try:
            result = model.fit(start_params=start, reml=False, method=method,
                               maxiter=POLICY["maxiter"], full_output=True, disp=False)
        except Exception as exc:
            error = type(exc).__name__ + ": " + str(exc)
    log = {"optimizer": method, "elapsed_seconds": time.perf_counter()-began, "error": error,
           "warnings": [{"category": w.category.__name__, "message": str(w.message),
                         "source_name": str(w.filename).replace("\\", "/").split("/")[-1], "line": w.lineno} for w in records]}
    if result is not None:
        log.update({"converged": bool(result.converged), "log_likelihood": float(result.llf),
                    "parameter_names": list(result.params.index), "parameters": np.asarray(result.params).tolist(),
                    "optimization_history": getattr(result, "hist", None)})
    return result, clean(log)


def fit_unit(frame, x, design, anchor, cohort, spec, primary_beta, primary_diag, vectors, output, authorization):
    require(authorization.get("authorization") == "GRANTED_FOR_SIX_ROUND1B_MODELS", "缺少六模型真实执行授权")
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "design_matrix.json", design)
    write_json(output / "sample_anchor.json", anchor)
    logs, diagnostics = [], {}
    result_object = {"cohort": cohort, "spec_id": spec, "source_kind": "NEW_LMM_FIT",
                     "sample": anchor, "status": "HOLD_NOT_ESTIMABLE", "author_scientific_review": "PENDING",
                     "participant_records_saved": False, "stage3_complete": False, "release_eligible": False}
    try:
        z = random_design(frame.time_years.to_numpy(float), spec)
        model = sm.MixedLM(frame.fi_primary.astype(float), x, frame.person_id.to_numpy(), exog_re=z, use_sqrt=True)
        q = np.array([[primary_diag["random_intercept_variance"]/primary_diag["residual_variance"]]])
        beta_start = primary_beta.copy()
        if spec == "RS01":
            q = np.diag([q[0, 0], POLICY["rs_start_slope_unscaled_variance"]])
        else:
            beta_start = np.r_[beta_start, 0., 0.]
        start = MixedLMParams.from_components(fe_params=beta_start, cov_re=q)
        bfgs, log = optimize(model, "bfgs", start)
        logs.append(log)
        write_json(output / "optimization_log.json", logs)
        require(bfgs is not None, "BFGS未返回结果；不自动更换优化器或规格")
        # 参数对象明确保存非平方根协方差，防止数组被重新解释为Cholesky起点。
        lbfgs, log = optimize(model, "lbfgs", bfgs.params_object.copy())
        logs.append(log)
        write_json(output / "optimization_log.json", logs, replace_owned=True)
        require(lbfgs is not None, "L-BFGS未返回结果；不自动更换规格")
        names = list(x.columns)
        beta1 = np.asarray(bfgs.fe_params, dtype=float)
        cov1 = np.asarray(bfgs.cov_params().loc[names, names], dtype=float)
        beta, covariance, diagnostics = extract_numerical(model, lbfgs)
        require(np.isfinite(beta).all(), "固定效应非有限")
        checks = optimizer_contrast_checks((beta1, cov1), (beta, covariance), vectors, spec)
        checks.update({"loglik_absolute_difference": abs(float(bfgs.llf-lbfgs.llf)), "bfgs_converged": bool(bfgs.converged)})
        classification = classify_numerical(diagnostics, checks, primary_diag["log_likelihood"])
        diagnostics.update({"optimizer_checks": checks, **classification})
        # 参数文件不含模型数据；失败和边界参数也保留，明确禁止投稿读取。
        np.savez_compressed(output / "model_parameters.npz", fixed_effect_names=np.array(names),
                            fixed_effects=beta, fixed_effect_covariance=covariance,
                            parameter_names=np.array(lbfgs.params.index), parameters=np.asarray(lbfgs.params),
                            random_covariance_scaled=np.asarray(lbfgs.cov_re),
                            random_covariance_unscaled=np.asarray(lbfgs.cov_re_unscaled),
                            residual_variance=np.asarray(lbfgs.scale), hessian=np.asarray(diagnostics["hessian"]))
        fixed_rows = []
        for i, name in enumerate(names):
            variance = float(covariance[i, i])
            se = float(np.sqrt(variance)) if np.isfinite(variance) and variance > 0 else None
            estimate = float(beta[i])
            fixed_rows.append({"cohort": cohort, "spec_id": spec, "term": name, "estimate": estimate,
                               "se": se, "ci_lower": estimate-Z95*se if se is not None else None,
                               "ci_upper": estimate+Z95*se if se is not None else None,
                               "p_value": float(2*norm.sf(abs(estimate/se))) if se is not None else None,
                               "status": classification["status"]})
        write_csv(output / "fixed_effects.csv", fixed_rows)
        write_csv(output / "fixed_effect_covariance.csv", [
            {"term_i": a, "term_j": b, "covariance": float(covariance[i, j])}
            for i, a in enumerate(names) for j, b in enumerate(names)])
        annual = covariance_to_years(lbfgs.cov_re) if spec == "RS01" else np.asarray(lbfgs.cov_re)
        write_json(output / "random_effects.json", {"random_design": "[1,T/8]" if spec == "RS01" else "[1]",
                   "covariance_fit_scale": np.asarray(lbfgs.cov_re), "covariance_annual_scale": annual,
                   "slope_variance_divisor": 64 if spec == "RS01" else None,
                   "intercept_slope_covariance_divisor": 8 if spec == "RS01" else None,
                   "residual_variance": float(lbfgs.scale)})
        result_object.update({**classification, "fixed_effect_names": names, "fixed_effects": fixed_rows,
                              "fixed_effect_covariance": covariance.tolist(), "random_covariance_annual": annual.tolist(),
                              "log_likelihood": float(lbfgs.llf), "aic": float(lbfgs.aic), "bic": float(lbfgs.bic),
                              "bic_n_definition": "FI model observation rows", "normal_Wald": True,
                              "primary_estimand_beta": float(beta[names.index(TERM)]) if spec == "RS01" else None})
        if classification["status"] != "HOLD_NOT_ESTIMABLE":
            contrasts = evaluate_all(beta, covariance, vectors, cohort, spec)
            for row in contrasts:
                row.update({"status": classification["status"], "author_scientific_review": "PENDING"})
            write_csv(output / "contrasts.csv", contrasts)
            result_object["contrasts"] = contrasts
            if spec == "NL01":
                write_json(output / "nonlinear_tests.json", nonlinear_tests(beta, covariance, names))
    except Exception as exc:
        reason = type(exc).__name__ + ": " + str(exc)
        result_object.update({"status": "HOLD_NOT_ESTIMABLE", "failure_reasons": [reason], "release_eligible": False})
        diagnostics.update({"status": "HOLD_NOT_ESTIMABLE", "failure_reasons": [reason]})
        # 不保存locals或模型对象，堆栈只含代码定位。
        (output / "failure_trace.txt").write_text(traceback.format_exc(), encoding="utf-8")
    write_json(output / "optimization_log.json", logs, replace_owned=(output / "optimization_log.json").exists())
    write_json(output / "diagnostics.json", diagnostics)
    write_json(output / "result_object.json", result_object)
    return result_object
