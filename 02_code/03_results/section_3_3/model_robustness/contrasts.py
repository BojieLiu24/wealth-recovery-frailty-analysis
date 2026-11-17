"""参数层面的标准化均值及对比；不拟合、不制表、不输出个人记录。"""
from __future__ import annotations

import numpy as np
import patsy
from scipy.stats import chi2, norm

from common import TERM, Z95, require
from specification import KNOTS
from time_basis import rcs_basis


def prediction_vectors(landmark, primary_design_info, spec, cohort):
    """原固定效应仅线性time，先构造均值截距及导数，避免重复展开个人网格。"""
    result = {}
    for recovery in (0, 1):
        means = []
        for t in (0., 1.):
            temp = landmark.assign(recovery=recovery, time_years=t)
            matrix = patsy.build_design_matrices([primary_design_info], temp)[0]
            means.append(np.asarray(matrix, dtype=float).mean(axis=0))
        result[recovery] = (means[0], means[1]-means[0])

    def vector(recovery, time, derivative=False):
        base, slope = result[recovery]
        value = slope.copy() if derivative else base+time*slope
        if spec == "NL01":
            h = float(rcs_basis(time, KNOTS[cohort], derivative=int(derivative)))
            value = np.r_[value, h, recovery*h]
        return value

    return vector


def contrast_vector_set(vector):
    rows = []
    initial_difference = vector(1, 0)-vector(0, 0)
    for time in np.arange(0., 8.125, .25):
        mu0, mu1 = vector(0, time), vector(1, time)
        difference = mu1-mu0
        for estimand, v, units in [
            ("mu_0", mu0, "FI units"), ("mu_1", mu1, "FI units"),
            ("D", difference, "FI units"), ("C", difference-initial_difference, "FI units"),
            ("S", vector(1, time, True)-vector(0, time, True), "FI units/year")]:
            rows.append((estimand, float(time), v, units))
    for recovery in (0, 1):
        rows.append((f"group_{recovery}_average_slope_0_8", None,
                     (vector(recovery, 8)-vector(recovery, 0))/8, "FI units/year"))
    rows.append(("A_0_8", None, (vector(1, 8)-vector(0, 8)-initial_difference)/8, "FI units/year"))
    return rows


def evaluate_contrast(beta, covariance, vector):
    beta, covariance, vector = map(lambda x: np.asarray(x, dtype=float), (beta, covariance, vector))
    estimate = float(vector @ beta)
    variance = float(vector @ covariance @ vector)
    require(np.isfinite([estimate, variance]).all(), "对比或方差非有限")
    deterministic = np.count_nonzero(vector) == 0
    require(variance == 0 if deterministic else variance > 0, "对比方差不可用；禁止裁剪或伪逆")
    se = float(np.sqrt(variance))
    return {"estimate": estimate, "se": se, "ci_lower": estimate-Z95*se,
            "ci_upper": estimate+Z95*se, "p_value": None if deterministic else float(2*norm.sf(abs(estimate/se))),
            "deterministic_identity": deterministic, "ci_type": "pointwise_normal_Wald_95"}


def evaluate_all(beta, covariance, vectors, cohort, spec):
    return [{"cohort": cohort, "spec_id": spec, "estimand_id": name, "time_years": time,
             "units": units, **evaluate_contrast(beta, covariance, v)} for name, time, v, units in vectors]


def optimizer_contrast_checks(first, second, vectors, spec):
    selected = [row for row in vectors if row[0] in (("A_0_8",) if spec == "RS01" else ("D", "C", "S", "A_0_8"))]
    max_estimate, max_se = 0., 0.
    for _, _, v, _ in selected:
        a = evaluate_contrast(first[0], first[1], v)
        b = evaluate_contrast(second[0], second[1], v)
        max_estimate = max(max_estimate, abs(a["estimate"]-b["estimate"]))
        if a["se"] > 0:
            max_se = max(max_se, abs(a["se"]-b["se"])/a["se"])
    return {"max_contrast_absolute_difference": max_estimate, "max_core_se_relative_difference": max_se}


def nonlinear_tests(beta, covariance, names):
    rows = []
    for test_id, terms in [("all_nonlinear_terms", ["rcs_time", "recovery:rcs_time"]),
                           ("nonconstant_group_slope_difference", ["recovery:rcs_time"]),
                           ("group_change_difference", [TERM, "recovery:rcs_time"])]:
        indices = [names.index(term) for term in terms]
        b = np.asarray(beta)[indices]
        cov = np.asarray(covariance)[np.ix_(indices, indices)]
        np.linalg.cholesky(cov)
        statistic = float(b @ np.linalg.solve(cov, b))
        rows.append({"test_id": test_id, "terms": terms, "df": len(terms), "statistic": statistic,
                     "p_value": float(chi2.sf(statistic, len(terms))), "method": "model_based_Wald",
                     "model_selection_rule": False})
    return rows
