"""Stage2基础数值QA；边界审阅与作者科学批准严格分开。"""
from __future__ import annotations

import warnings
import numpy as np

from common import require
from specification import POLICY


def matrix_spectrum(matrix):
    a = np.asarray(matrix, dtype=float)
    finite = bool(np.isfinite(a).all())
    symmetric = finite and bool(np.allclose(a, a.T, rtol=1e-10, atol=1e-12))
    if not symmetric:
        return {"finite": finite, "symmetric": symmetric, "eigenvalues": None, "condition_number": None}
    return {"finite": True, "symmetric": True, "eigenvalues": np.linalg.eigvalsh(a).tolist(),
            "condition_number": float(np.linalg.cond(a))}


def boundary_kkt(covariance_unscaled, packed_score, information):
    """PSD锥最大化的一阶条件：正方差方向驻点，零空间方向不能提高似然。"""
    q = np.asarray(covariance_unscaled, dtype=float)
    score = np.asarray(packed_score, dtype=float)
    info = np.asarray(information, dtype=float)
    size = q.shape[0]
    require(size in (1, 2) and len(score) == size*(size+1)//2, "KKT参数维度错误")
    eig, u = np.linalg.eigh(q)
    positive = eig > max(float(np.max(np.abs(eig)))*1e-6, 1e-12)
    s = np.zeros_like(q)
    indices = np.tril_indices(size)
    for value, i, j in zip(score, *indices):
        # 打包的非对角坐标同时改变两个对称元素，其矩阵梯度须除以2。
        s[i, j] = value if i == j else value/2
        s[j, i] = s[i, j]
    transformed = u.T @ s @ u
    stationary_score = s - (u[:, ~positive] @ transformed[np.ix_(~positive, ~positive)] @ u[:, ~positive].T)
    stationary = np.array([stationary_score[i, j] if i == j else 2*stationary_score[i, j] for i, j in zip(*indices)])
    diagonals = np.diag(info)
    if not np.isfinite(diagonals).all() or (diagonals <= 0).any():
        return {"status": "UNRESOLVED", "reason": "方差坐标信息对角不足以标准化", "score_matrix": s.tolist()}
    stationary_norm = float(np.max(np.abs(stationary)/np.sqrt(diagonals)))
    null_eigen = np.linalg.eigvalsh(transformed[np.ix_(~positive, ~positive)]) if (~positive).any() else np.array([])
    # 此标量只用于一阶数值筛查；Stage3仍须审阅边界弱识别和推断支持。
    forbidden = max(float(null_eigen.max()), 0.) if len(null_eigen) else 0.
    null_violation = forbidden / float(np.sqrt(diagonals.min()))
    return {"status": "PASS_SCREEN" if max(stationary_norm, null_violation) <= 1e-3 else "FAIL_SCREEN",
            "stationary_scaled_score": stationary_norm, "nullspace_positive_score_scaled": null_violation,
            "score_matrix": s.tolist(), "covariance_eigenvalues": eig.tolist(),
            "parameterization": "direct_unscaled_covariance; offdiagonal_half_score",
            "evidence": "first_order_PSD_screen_not_author_approval"}


def extract_numerical(model, result):
    names = list(result.fe_params.index)
    beta = np.asarray(result.fe_params, dtype=float)
    cov = np.asarray(result.cov_params().loc[names, names], dtype=float)
    g = np.asarray(result.cov_re, dtype=float)
    with warnings.catch_warnings(record=True) as records:
        warnings.simplefilter("always")
        hessian, singular = model.hessian(result.params_object)
        parts = model.score_full(result.params_object, calc_fe=True)
    score = np.concatenate(parts)
    information = -np.asarray(hessian, dtype=float)
    spec_g, spec_cov, spec_info = map(matrix_spectrum, (g, cov, information))
    eigen = np.asarray(spec_g["eigenvalues"] if spec_g["eigenvalues"] is not None else [np.nan])
    sigma = float(result.scale)
    relative = float(eigen.min()/eigen.max()) if np.isfinite(eigen).all() and eigen.max() > 0 else 0.
    rho = float(g[0, 1]/np.sqrt(g[0, 0]*g[1, 1])) if g.shape == (2, 2) and g[0, 0] > 0 and g[1, 1] > 0 else None
    boundary = relative < POLICY["near_singular_eigenvalue_ratio"] or (rho is not None and abs(rho) >= .99)
    if g.shape == (1, 1):
        boundary = float(g[0, 0]) / sigma < 1e-6 if sigma > 0 else True
    balanced_cholesky = False
    scaled_gradient = None
    if spec_info["symmetric"] and (np.diag(information) > 0).all():
        d = np.sqrt(np.diag(information))
        balanced = information/d[:, None]/d[None, :]
        try:
            np.linalg.cholesky(balanced)
            balanced_cholesky = True
        except np.linalg.LinAlgError:
            pass
        scaled_gradient = float(np.max(np.abs(score)/d))
    kkt = boundary_kkt(np.asarray(result.cov_re_unscaled), score[len(names):], information[len(names):, len(names):]) if boundary else None
    return beta, cov, {
        "converged": bool(result.converged), "log_likelihood": float(result.llf),
        "residual_variance": sigma, "random_covariance_scaled": g.tolist(),
        "random_covariance_unscaled": np.asarray(result.cov_re_unscaled).tolist(),
        "random_spectrum": spec_g, "random_eigenvalue_ratio": relative, "random_correlation": rho,
        "boundary_candidate": boundary, "fixed_covariance_spectrum": spec_cov,
        "information_spectrum": spec_info, "balanced_information_cholesky": balanced_cholesky,
        "hessian_covariance_pseudoinverse_flag": bool(singular),
        "score_direct_unscaled_covariance": score.tolist(), "scaled_gradient_max": scaled_gradient,
        "score_finite": bool(np.isfinite(score).all()),
        "boundary_kkt": kkt, "hessian": np.asarray(hessian).tolist(),
        "parameter_names": list(result.params.index),
        "diagnostic_warnings": [{"category": w.category.__name__, "message": str(w.message)} for w in records],
    }


def classify_numerical(diagnostics, optimizer_checks, primary_loglik):
    reasons, caveats = [], []
    if not diagnostics.get("score_finite", True):
        reasons.append("梯度非有限")
    for key in ("loglik_absolute_difference", "max_contrast_absolute_difference", "max_core_se_relative_difference"):
        if not np.isfinite(optimizer_checks[key]):
            reasons.append("优化器对比非有限：" + key)
    if not diagnostics["converged"]:
        reasons.append("优化器未收敛")
    if not np.isfinite(diagnostics["log_likelihood"]) or diagnostics["log_likelihood"] < primary_loglik-1e-4:
        reasons.append("似然非有限或低于嵌套Primary")
    sigma = diagnostics["residual_variance"]
    if not np.isfinite(sigma) or sigma <= 0:
        reasons.append("残差方差非法")
    spectrum = diagnostics["random_spectrum"]
    if not spectrum["symmetric"]:
        reasons.append("随机协方差非有限或不对称")
    elif min(spectrum["eigenvalues"]) < -1e-12*max(max(map(abs, spectrum["eigenvalues"])), 1e-12):
        reasons.append("随机协方差不合法半正定")
    fixed = diagnostics["fixed_covariance_spectrum"]
    if not fixed["symmetric"] or min(fixed["eigenvalues"]) <= 0:
        reasons.append("固定效应协方差不可用于Wald推断")
    if optimizer_checks["loglik_absolute_difference"] >= 1e-4:
        reasons.append("双优化器loglik差异超门槛")
    if optimizer_checks["max_contrast_absolute_difference"] >= 1e-6:
        reasons.append("双优化器核心估计差异超门槛")
    if optimizer_checks["max_core_se_relative_difference"] > 1e-3:
        reasons.append("双优化器核心SE差异超门槛")
    if not optimizer_checks["bfgs_converged"]:
        reasons.append("BFGS未收敛")
    boundary = diagnostics["boundary_candidate"]
    if boundary:
        caveats.append("边界或近奇异候选；需Stage3审阅，不自动简化或升级PASS_WITH_CAVEAT")
        if diagnostics["boundary_kkt"]["status"] != "PASS_SCREEN":
            caveats.append("边界KKT筛查未通过或未能标准化")
        if not diagnostics["balanced_information_cholesky"]:
            caveats.append("完整观察信息不支持内点Cholesky；Stage3须区分方差弱识别及核心推断失效")
    elif (not diagnostics["balanced_information_cholesky"] or diagnostics["scaled_gradient_max"] is None
          or not np.isfinite(diagnostics["scaled_gradient_max"]) or diagnostics["scaled_gradient_max"] > 1e-3):
        reasons.append("内点信息矩阵或标准化梯度未通过")
    return {"status": "HOLD_NOT_ESTIMABLE" if reasons else ("BOUNDARY_REVIEW" if boundary else "PASS"),
            "failure_reasons": reasons, "caveats": caveats,
            "author_scientific_review": "PENDING", "stage3_complete": False,
            "release_eligible": False, "automatic_specification_change": False}
