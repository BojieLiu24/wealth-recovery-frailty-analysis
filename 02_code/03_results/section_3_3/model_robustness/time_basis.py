"""RCS、解析导数与随机时间单位转换；不包含拟合。"""
from __future__ import annotations

import numpy as np
from common import require


def rcs_basis(time, knots, derivative=0):
    k1, k2, k3 = np.asarray(knots, dtype=float)
    require(np.isfinite([k1, k2, k3]).all() and k1 == 0 and k1 < k2 < k3, "RCS结点无效")
    require(derivative in (0, 1, 2), "RCS仅支持0、1、2阶导数")
    time = np.asarray(time, dtype=float)
    require(np.isfinite(time).all(), "RCS时间非有限")
    power = 3 - derivative
    factor = (1., 3., 6.)[derivative]
    value = factor * (np.maximum(time-k1, 0)**power
                     - (k3-k1)/(k3-k2)*np.maximum(time-k2, 0)**power
                     + (k2-k1)/(k3-k2)*np.maximum(time-k3, 0)**power) / (k3-k1)**2
    return value


def random_design(time, spec):
    time = np.asarray(time, dtype=float)
    require(spec in ("RS01", "NL01"), "未知模型规格")
    return np.column_stack([np.ones(len(time)), time / 8]) if spec == "RS01" else np.ones((len(time), 1))


def covariance_to_years(covariance):
    covariance = np.asarray(covariance, dtype=float)
    require(covariance.shape == (2, 2), "随机截距斜率协方差必须为2x2")
    d = np.diag([1., 1/8])
    return d @ covariance @ d


def covariance_to_scaled(covariance):
    d = np.diag([1., 8.])
    return d @ np.asarray(covariance, dtype=float) @ d


def conditional_prediction(y, x, beta, z, covariance, sigma2):
    """允许合法半正定G；通过V求解，不对G取逆。"""
    v = z @ covariance @ z.T + sigma2 * np.eye(len(y))
    random_coefficients = covariance @ z.T @ np.linalg.solve(v, y-x @ beta)
    return x @ beta + z @ random_coefficients


def gaussian_loglik(y, mean, covariance):
    factor = np.linalg.cholesky(covariance)
    residual = np.linalg.solve(factor, y-mean)
    return float(-.5 * (len(y)*np.log(2*np.pi) + 2*np.log(np.diag(factor)).sum() + residual @ residual))


def scale_equivalence_checks():
    """确定种子、正负协方差和不规则访视下固定参数等价性；不调用优化器。"""
    rng = np.random.default_rng(20261014)
    cases = []
    for correlation in (-.55, .45):
        g = np.array([[.008, correlation*np.sqrt(.008*.003)],
                      [correlation*np.sqrt(.008*.003), .003]])
        annual = covariance_to_years(g)
        np.testing.assert_allclose(covariance_to_scaled(annual), g, rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(annual[1, 1], g[1, 1]/64, rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(annual[0, 1], g[0, 1]/8, rtol=1e-10, atol=1e-12)
        for t in ([0., 2.17], [0., .83, 5.92], [0., 1.75, 4.08, 8.33, 15.5]):
            t = np.asarray(t)
            zu, zt = random_design(t, "RS01"), np.column_stack([np.ones(len(t)), t])
            x = np.column_stack([np.ones(len(t)), t])
            beta = np.array([.2, .007])
            y = x @ beta + rng.normal(0, .04, len(t))
            a = rng.multivariate_normal(np.zeros(2), g)
            b = a / np.array([1., 8.])
            vu = zu @ g @ zu.T + .005*np.eye(len(t))
            vt = zt @ annual @ zt.T + .005*np.eye(len(t))
            pu = conditional_prediction(y, x, beta, zu, g, .005)
            pt = conditional_prediction(y, x, beta, zt, annual, .005)
            for first, second in [(vu, vt), (zu @ a, zt @ b), (pu, pt), (x @ beta, x @ beta)]:
                np.testing.assert_allclose(first, second, rtol=1e-10, atol=1e-12)
            ll_error = abs(gaussian_loglik(y, x @ beta, vu)-gaussian_loglik(y, x @ beta, vt))
            require(ll_error <= 1e-8, "随机时间尺度似然不等价")
            cases.append({"correlation": correlation, "observations": len(t),
                          "covariance_max_abs_error": float(np.max(np.abs(vu-vt))),
                          "conditional_prediction_max_abs_error": float(np.max(np.abs(pu-pt))),
                          "loglik_absolute_error": ll_error})
    return {"status": "PASS", "seed": 20261014, "fit_calls": 0, "cases": cases,
            "variance_divisor": 64, "covariance_divisor": 8,
            "rtol": 1e-10, "atol": 1e-12, "loglik_atol": 1e-8}
