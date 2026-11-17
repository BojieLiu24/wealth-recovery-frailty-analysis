from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import shutil
import subprocess
import sys
import warnings
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import patsy
import scipy
from PIL import Image
from scipy.stats import norm, t as student_t
import statsmodels
import statsmodels.api as sm


ROOT = Path(__file__).resolve().parents[3]
FINAL_OUTPUT_DIR = ROOT / "03_output/3.2 Wealth Recovery and Subsequent Frailty Progression"
METAFOR_FIXTURE_PATH = ROOT / "00_admin/model_manifests/primary_lmm/metafor_reml_fixture_v1.json"
PANEL_CODE_DIR = ROOT / "02_code" / "01_panel_build"
if str(PANEL_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(PANEL_CODE_DIR))
import analysis_panel


COHORTS = ("HRS", "ELSA", "KLoSA")
GROUPS = ("substantial_recovery", "persistent_or_limited_recovery")
EXPECTED = {
    "HRS": {"participants": 5114, "later": 18297, "rows": 23411, "groups": {"substantial_recovery": 1703, "persistent_or_limited_recovery": 3411}},
    "ELSA": {"participants": 699, "later": 1885, "rows": 2584, "groups": {"substantial_recovery": 271, "persistent_or_limited_recovery": 428}},
    "KLoSA": {"participants": 2502, "later": 8260, "rows": 10762, "groups": {"substantial_recovery": 943, "persistent_or_limited_recovery": 1559}},
}
CONTINUOUS = ("age", "household_size", "household_income_rank", "wealth_loss_fraction", "fi_t")
CATEGORICAL = {
    "sex": ("Male", "Female"),
    "education": ("High", "Medium", "Low"),
    "marital_status": ("Married/partnered", "Separated/divorced", "Widowed", "Never married"),
    "labour_force_status": ("Working", "Unemployed", "Retired or partly retired", "Other economically inactive"),
    "current_smoking": ("False", "True"),
    "wealth_quintile": ("5", "1", "2", "3", "4"),
}
REFERENCES = {"sex": "Male", "education": "High", "marital_status": "Married/partnered", "labour_force_status": "Working", "current_smoking": "False", "wealth_quintile": "5"}
FORMULA = (
    "fi_primary ~ recovery * time_years + age_z + household_size_z + household_income_rank_z + wealth_loss_fraction_z + fi_t_z "
    "+ C(sex, Treatment(reference='Male')) + C(education, Treatment(reference='High')) "
    "+ C(marital_status, Treatment(reference='Married/partnered')) "
    "+ C(labour_force_status, Treatment(reference='Working')) "
    "+ C(current_smoking, Treatment(reference='False')) "
    "+ C(wealth_quintile, Treatment(reference='5')) "
    "+ age_z:time_years + household_size_z:time_years + household_income_rank_z:time_years + wealth_loss_fraction_z:time_years + fi_t_z:time_years "
    "+ C(sex, Treatment(reference='Male')):time_years "
    "+ C(education, Treatment(reference='High')):time_years "
    "+ C(marital_status, Treatment(reference='Married/partnered')):time_years "
    "+ C(labour_force_status, Treatment(reference='Working')):time_years "
    "+ C(current_smoking, Treatment(reference='False')):time_years "
    "+ C(wealth_quintile, Treatment(reference='5')):time_years"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def now_utc() -> str:
    return datetime.now(UTC).isoformat()


def relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def output_manifest_path(path: Path, output_dir: Path) -> str:
    """将staging中的产物映射为最终发布目录的相对路径。"""
    return relative(FINAL_OUTPUT_DIR / path.relative_to(output_dir))


def git_value(args: list[str]) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True, encoding="utf-8").strip()
    except Exception:
        return "unavailable"


def normalize_rgb_raster(path: Path) -> None:
    """将Matplotlib栅格输出固定为RGB并保留600 dpi。"""
    with Image.open(path) as image:
        dpi = image.info.get("dpi", (600, 600))
        rgb = image.convert("RGB")
        if path.suffix.lower() in {".tif", ".tiff"}:
            rgb.save(path, dpi=dpi, compression="tiff_lzw")
        else:
            rgb.save(path, dpi=dpi)


def load_contract() -> dict[str, Any]:
    path = ROOT / "00_admin" / "model_manifests" / "primary_lmm" / "primary_lmm_v1_contract.json"
    return json.loads(path.read_text(encoding="utf-8"))


def verify_inputs() -> dict[str, Any]:
    freeze_path = ROOT / "00_admin" / "panel_manifests" / "analysis_panel" / "analysis_panel_v1_freeze.json"
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    verified = analysis_panel.verify_analysis_panel_freeze()
    if verified.get("freeze_status") != "FROZEN":
        raise RuntimeError("Analysis Panel v1未处于FROZEN状态。")
    if freeze.get("freeze_status") != "FROZEN":
        raise RuntimeError("Analysis Panel freeze JSON 未声明 FROZEN。")
    if not METAFOR_FIXTURE_PATH.is_file():
        raise RuntimeError(f"缺少metafor reference fixture：{METAFOR_FIXTURE_PATH}")
    freeze_assets = {
        "table_1": ROOT / "03_output/3.1 Study Population, Wealth Recovery, and Follow-up/Table 1/table_1_freeze.json",
        "table_s5": ROOT / "03_output/3.1 Study Population, Wealth Recovery, and Follow-up/Table S5/table_s5_freeze.json",
    }
    for label, path in freeze_assets.items():
        if not path.is_file():
            raise RuntimeError(f"缺少冻结表资产：{label} ({path})")
        asset_freeze = json.loads(path.read_text(encoding="utf-8"))
        if asset_freeze.get("freeze_status") != "FROZEN":
            raise RuntimeError(f"{label} freeze JSON 未声明 FROZEN。")
    return {
        "analysis_panel_freeze": {"path": relative(freeze_path), "sha256": sha256_file(freeze_path)},
        "verified_status": verified.get("analysis_panel_status"),
        "table_1_freeze": {"path": relative(freeze_assets["table_1"]), "sha256": sha256_file(freeze_assets["table_1"])},
        "table_s5_freeze": {"path": relative(freeze_assets["table_s5"]), "sha256": sha256_file(freeze_assets["table_s5"])},
        "metafor_fixture": {"path": relative(METAFOR_FIXTURE_PATH), "sha256": sha256_file(METAFOR_FIXTURE_PATH)},
    }


def verify_table_anchors(model_anchors: list[dict[str, Any]]) -> dict[str, Any]:
    """核对Table 1、Table S5与本阶段模型样本的重叠锚点。"""
    base = ROOT / "03_output/3.1 Study Population, Wealth Recovery, and Follow-up"
    table1 = pd.read_csv(base / "Table 1/table_1_summary_long.csv", encoding="utf-8-sig")
    table_s5 = pd.read_csv(base / "Table S5/table_s5_summary_long.csv", encoding="utf-8-sig")
    for label, table in (("Table 1", table1), ("Table S5", table_s5)):
        required = {"row_id", "cohort", "recovery_status", "value", "median", "q1", "q3"}
        if not required.issubset(table.columns):
            raise RuntimeError(f"{label} summary缺少锚点字段。")
    checks: list[dict[str, Any]] = []
    for cohort in COHORTS:
        for group in GROUPS:
            key = {"cohort": cohort, "recovery_status": group}
            t1 = table1[(table1["cohort"] == cohort) & (table1["recovery_status"] == group)]
            s5 = table_s5[(table_s5["cohort"] == cohort) & (table_s5["recovery_status"] == group)]
            if len(t1) == 0 or len(s5) == 0:
                raise RuntimeError(f"{cohort}/{group}: Table 1或Table S5锚点缺失。")
            row_values: dict[str, Any] = {}
            for row_id in ("participants", "total_post_landmark_fi_observations", "follow_up_duration"):
                a = t1.loc[t1["row_id"] == row_id].iloc[0]
                b = s5.loc[s5["row_id"] == row_id].iloc[0]
                fields = ("median", "q1", "q3") if row_id == "follow_up_duration" else ("value",)
                for field in fields:
                    source_field = "numerator" if row_id == "participants" and field == "value" else field
                    av = float(a[source_field]); bv = float(b[field])
                    if not np.isfinite(av) or not np.isfinite(bv) or not np.isclose(av, bv, rtol=0.0, atol=1e-12):
                        raise RuntimeError(f"{cohort}/{group}/{row_id}/{field}: Table 1与S5锚点漂移。")
                    row_values[f"{row_id}_{field}"] = av
            checks.append({**key, **row_values})
    return {"status": "PASS", "anchors": checks}


def prepare_frame(frame: pd.DataFrame, cohort: str) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    required = ["cohort", "person_id", "wave", "fi_time_role", "time_years", "fi_primary", "recovery_status", *CONTINUOUS, *CATEGORICAL]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise RuntimeError(f"{cohort}:缺少模型列 {missing}")
    frame = frame.loc[frame["primary_complete_case_eligible"].astype(bool)].copy()
    frame["person_id"] = frame["person_id"].astype(str)
    if frame[["person_id", "wave"]].duplicated().any():
        raise RuntimeError(f"{cohort}:存在重复person-wave。")
    landmark = frame.loc[frame["fi_time_role"].eq("landmark")].copy()
    later = frame.loc[frame["fi_time_role"].eq("later")].copy()
    if landmark.groupby("person_id").size().ne(1).any() or later.groupby("person_id").size().lt(1).any():
        raise RuntimeError(f"{cohort}:landmark/later FI支持不满足每人一条landmark且至少一条later。")
    if not np.isfinite(frame["time_years"].astype(float)).all() or not (later["time_years"] > 0).all():
        raise RuntimeError(f"{cohort}:time_years非有限或later时间不为正。")
    for column in ["fi_primary", *CONTINUOUS, *CATEGORICAL, "recovery_status"]:
        if frame[column].isna().any():
            raise RuntimeError(f"{cohort}:{column}存在primary模型缺失。")
    if set(landmark["recovery_status"]) != set(GROUPS):
        raise RuntimeError(f"{cohort}:recovery status缺少组别。")
    if len(landmark) != EXPECTED[cohort]["participants"] or len(later) != EXPECTED[cohort]["later"] or len(frame) != EXPECTED[cohort]["rows"]:
        raise RuntimeError(f"{cohort}:primary模型人数/行数anchor不匹配。")
    if landmark["recovery_status"].value_counts().to_dict() != EXPECTED[cohort]["groups"]:
        raise RuntimeError(f"{cohort}:recovery组人数anchor不匹配。")
    for column, levels in CATEGORICAL.items():
        frame[column] = frame[column].astype(str)
        landmark[column] = landmark[column].astype(str)
        if not set(landmark[column]).issubset(set(levels)):
            raise RuntimeError(f"{cohort}:{column}出现未批准水平。")
    constants: dict[str, Any] = {}
    for column in CONTINUOUS:
        values = pd.to_numeric(landmark[column], errors="raise").astype(float)
        mean = float(values.mean())
        sd = float(values.std(ddof=1))
        if not np.isfinite(mean) or not np.isfinite(sd) or sd <= 0:
            raise RuntimeError(f"{cohort}:{column}标准化常数无效。")
        constants[column] = {"mean": mean, "sample_sd": sd, "n_unique_participants": int(values.size)}
        participant_values = landmark.set_index("person_id")[column].astype(float)
        if frame[column].astype(float).groupby(frame["person_id"]).nunique().gt(1).any():
            raise RuntimeError(f"{cohort}:{column}未在participant内保持固定。")
        frame[f"{column}_z"] = (frame[column].astype(float) - mean) / sd
    frame["recovery"] = frame["recovery_status"].eq("substantial_recovery").astype(int)
    frame["wealth_quintile"] = frame["wealth_quintile"].astype(str)
    frame["current_smoking"] = frame["current_smoking"].astype(str)
    return frame, landmark, constants


def fit_model(frame: pd.DataFrame, cohort: str) -> tuple[Any, Any, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    y, x = patsy.dmatrices(FORMULA, frame, return_type="dataframe", NA_action="raise")
    if x.shape[1] != 42 or np.linalg.matrix_rank(x.to_numpy()) != x.shape[1]:
        raise RuntimeError(f"{cohort}:设计矩阵不是42/42满秩。")
    model = sm.MixedLM(endog=y.iloc[:, 0], exog=x, groups=frame["person_id"].to_numpy(), exog_re=np.ones((len(frame), 1)))
    with warnings.catch_warnings(record=True) as fit_warnings:
        warnings.simplefilter("always")
        # 先用BFGS取得稳定的可行起点，再由正式L-BFGS完成最终优化；两个结果仍须满足一致性门槛。
        bfgs = model.fit(reml=False, method="bfgs", maxiter=2000, disp=False)
        lbfgs = model.fit(start_params=bfgs.params, reml=False, method="lbfgs", maxiter=2000, disp=False)
    for label, result in (("lbfgs", lbfgs), ("bfgs", bfgs)):
        if not bool(result.converged):
            raise RuntimeError(f"{cohort}:{label}未收敛。")
        if not np.isfinite(np.asarray(result.fe_params, dtype=float)).all() or not np.isfinite(np.asarray(result.cov_params(), dtype=float)).all():
            raise RuntimeError(f"{cohort}:{label}存在非有限估计或协方差。")
    term = "recovery:time_years"
    if term not in lbfgs.fe_params.index:
        raise RuntimeError(f"{cohort}:找不到primary recovery×time term。")
    beta_diff = abs(float(lbfgs.fe_params[term] - bfgs.fe_params[term]))
    ll_diff = abs(float(lbfgs.llf - bfgs.llf))
    if beta_diff >= 1e-6 or ll_diff >= 1e-4:
        raise RuntimeError(f"{cohort}:L-BFGS/BFGS primary term或log-likelihood不一致。")
    hessian, hessian_singular = model.hessian(lbfgs.params_object)
    if hessian_singular or not np.isfinite(np.asarray(hessian, dtype=float)).all():
        raise RuntimeError(f"{cohort}:最终L-BFGS Hessian不可用。")
    names = list(lbfgs.fe_params.index)
    cov = lbfgs.cov_params().loc[names, names]
    estimates = []
    for name in names:
        estimate = float(lbfgs.fe_params[name])
        se = float(math.sqrt(max(float(cov.loc[name, name]), 0.0)))
        z_value = estimate / se if se > 0 else np.nan
        estimates.append({"cohort": cohort, "term": name, "estimate": estimate, "se": se, "ci_lower": estimate - 1.959963984540054 * se, "ci_upper": estimate + 1.959963984540054 * se, "p_value": 2 * norm.sf(abs(z_value)) if np.isfinite(z_value) else np.nan, "is_primary_estimand": name == term})
    estimates_df = pd.DataFrame(estimates)
    cov_rows = [{"cohort": cohort, "term_i": i, "term_j": j, "covariance": float(cov.loc[i, j])} for i in names for j in names]
    covariance_df = pd.DataFrame(cov_rows)
    residual = np.asarray(lbfgs.resid, dtype=float)
    fitted = np.asarray(lbfgs.fittedvalues, dtype=float)
    diagnostics = {
        "cohort": cohort,
        "n_rows": int(len(frame)),
        "n_participants": int(frame["person_id"].nunique()),
        "design_matrix_columns": int(x.shape[1]),
        "design_matrix_rank": int(np.linalg.matrix_rank(x.to_numpy())),
        "design_matrix_condition_number": float(np.linalg.cond(x.to_numpy())),
        "lbfgs_converged": bool(lbfgs.converged),
        "bfgs_converged": bool(bfgs.converged),
        "beta3_optimizer_absolute_difference": beta_diff,
        "loglik_optimizer_absolute_difference": ll_diff,
        "log_likelihood": float(lbfgs.llf),
        "random_intercept_variance": float(np.asarray(lbfgs.cov_re)[0, 0]),
        "residual_variance": float(lbfgs.scale),
        "random_intercept_to_residual_variance_ratio": float(np.asarray(lbfgs.cov_re)[0, 0] / lbfgs.scale),
        "hessian_singular": bool(hessian_singular),
        "hessian_min_eigenvalue": float(np.min(np.linalg.eigvalsh(np.asarray(hessian, dtype=float)))),
        "hessian_condition_number": float(np.linalg.cond(np.asarray(hessian, dtype=float))),
        "wealth_loss_fraction_min": float(frame["wealth_loss_fraction"].astype(float).min()),
        "wealth_loss_fraction_max": float(frame["wealth_loss_fraction"].astype(float).max()),
        "residual_mean": float(np.mean(residual)),
        "residual_sd": float(np.std(residual, ddof=1)),
        "residual_rmse": float(np.sqrt(np.mean(residual**2))),
        "residual_time_correlation": float(np.corrcoef(residual, frame["time_years"].to_numpy())[0, 1]),
        "residual_fitted_correlation": float(np.corrcoef(residual, fitted)[0, 1]),
        "fit_warning_count": len(fit_warnings),
        "fit_warning_messages": [str(item.message) for item in fit_warnings],
    }
    return lbfgs, bfgs, estimates_df, covariance_df, diagnostics


def predictions(result: Any, frame: pd.DataFrame, landmark: pd.DataFrame, cohort: str, start: float = 0.0, stop: float = 8.0, step: float = 0.25) -> pd.DataFrame:
    design_info = patsy.dmatrix(FORMULA.split("~", 1)[1], frame, return_type="dataframe").design_info
    participant = landmark.copy()
    for column in CONTINUOUS:
        mean = float(pd.to_numeric(landmark[column]).mean())
        sd = float(pd.to_numeric(landmark[column]).std(ddof=1))
        participant[f"{column}_z"] = (pd.to_numeric(participant[column]).astype(float) - mean) / sd
    participant["wealth_quintile"] = participant["wealth_quintile"].astype(str)
    participant["current_smoking"] = participant["current_smoking"].astype(str)
    rows: list[dict[str, Any]] = []
    beta = np.asarray(result.fe_params, dtype=float)
    cov = np.asarray(result.cov_params().loc[result.fe_params.index, result.fe_params.index], dtype=float)
    times = np.round(np.arange(start, stop + step / 2, step), 10)
    for group_value, group_label in ((1, "Substantial wealth recovery"), (0, "Persistent loss or limited recovery")):
        for time_value in times:
            pred = participant.copy()
            pred["recovery"] = group_value
            pred["time_years"] = time_value
            x = patsy.build_design_matrices([design_info], pred, return_type="dataframe")[0]
            mean_x = np.asarray(x, dtype=float).mean(axis=0)
            estimate = float(mean_x @ beta)
            variance = float(mean_x @ cov @ mean_x)
            se = math.sqrt(max(variance, 0.0))
            rows.append({"cohort": cohort, "recovery_status": group_label, "time_years": float(time_value), "estimate": estimate, "se": se, "ci_lower": estimate - 1.959963984540054 * se, "ci_upper": estimate + 1.959963984540054 * se, "participants": int(len(participant))})
    return pd.DataFrame(rows)


def meta_analysis(primary: pd.DataFrame) -> dict[str, Any]:
    """复现metafor 5.0-1 rma.uni(REML, test='adhoc')的三研究截距模型。"""
    rows = primary.loc[primary["is_primary_estimand"]].copy().sort_values("cohort")
    yi = rows["estimate"].to_numpy(float)
    vi = rows["se"].to_numpy(float) ** 2
    k = len(yi)
    if k != 3 or not np.isfinite(yi).all() or not np.isfinite(vi).all() or (vi <= 0).any():
        raise RuntimeError("meta输入必须是三个有限的cohort-level estimate/SE。")
    # metafor的默认tau²初值和REML Fisher-scoring步进；threshold=1e-5。
    x = np.ones((k, 1), dtype=float)
    y_centered = yi - float(yi.mean())
    p0 = np.eye(k) - x @ np.linalg.solve(x.T @ x, x.T)
    v_matrix = np.diag(vi)
    tau2 = max(float((y_centered @ p0 @ y_centered - np.trace(p0 @ v_matrix)) / (k - 1)), 0.0)
    tau2_initial = tau2
    threshold = 1e-5
    change = threshold + 1.0
    iterations = 0
    while change > threshold:
        old_tau2 = tau2
        weights_matrix = np.diag(1.0 / (vi + tau2))
        p_matrix = weights_matrix - weights_matrix @ x @ np.linalg.solve(x.T @ weights_matrix @ x, x.T @ weights_matrix)
        pp_matrix = p_matrix @ p_matrix
        denominator = float(np.trace(pp_matrix))
        if denominator <= 0 or not np.isfinite(denominator):
            raise RuntimeError("metafor兼容REML步进的分母无效。")
        adjustment = float((y_centered @ pp_matrix @ y_centered - np.trace(p_matrix)) / denominator)
        while tau2 + adjustment < 0:
            adjustment /= 2.0
        tau2 = max(tau2 + adjustment, 0.0)
        change = abs(old_tau2 - tau2)
        iterations += 1
        if iterations > 100:
            raise RuntimeError("metafor兼容REML tau²迭代未收敛。")
    weights = 1.0 / (vi + tau2)
    mu = float(np.sum(weights * yi) / np.sum(weights))
    rss_knha = float(np.sum(weights * (yi - mu) ** 2))
    kh_scale = max(1.0, rss_knha / (k - 1))  # 修正KH（adhoc）的方差修正因子下限固定为1。
    se = math.sqrt(kh_scale / np.sum(weights))
    critical = float(student_t.ppf(0.975, k - 1))
    q_fixed_weights = 1.0 / vi
    mu_fixed = float(np.sum(q_fixed_weights * yi) / np.sum(q_fixed_weights))
    q_fixed = float(np.sum(q_fixed_weights * (yi - mu_fixed) ** 2))
    denominator_vt = float(np.sum(q_fixed_weights) ** 2 - np.sum(q_fixed_weights**2))
    vt = float((k - 1) * np.sum(q_fixed_weights) / denominator_vt) if denominator_vt > 0 else np.nan
    i2 = 100.0 * tau2 / (vt + tau2) if np.isfinite(vt) and vt + tau2 > 0 else np.nan
    return {"cohort": "Overall (random-effects)", "estimate": mu, "se": se, "ci_lower": mu - critical * se, "ci_upper": mu + critical * se, "tau2": tau2, "i2_percent": i2, "q_fixed": q_fixed, "kh_q_star": kh_scale, "kh_scale": kh_scale, "vt": vt, "tau2_initial": tau2_initial, "reml_iterations": iterations, "n_cohorts": k, "participants": "3 cohorts"}


def verify_metafor_fixture(meta: dict[str, Any], primary: pd.DataFrame) -> dict[str, Any]:
    """用锁定的metafor 5.0-1 fixture验证secondary合成，不依赖运行时R。"""
    fixture = json.loads(METAFOR_FIXTURE_PATH.read_text(encoding="utf-8"))
    # fixture输入顺序固定为HRS、ELSA、KLoSA，与结果合同的cohort顺序一致。
    rows = primary.loc[primary["is_primary_estimand"]].set_index("cohort").loc[list(COHORTS)].reset_index()
    if not np.allclose(rows["estimate"].to_numpy(float), fixture["yi"], rtol=0.0, atol=1e-15) or not np.allclose(rows["se"].to_numpy(float), fixture["sei"], rtol=0.0, atol=1e-15):
        raise RuntimeError("冻结cohort estimate/SE与metafor fixture输入不一致。")
    comparisons = {}
    for key, expected in fixture["expected"].items():
        if key == "degrees_of_freedom":
            continue
        actual_key = "kh_scale" if key == "kh_scale" else key
        actual = float(meta[actual_key])
        delta = abs(actual - float(expected))
        comparisons[key] = {"actual": actual, "expected": float(expected), "absolute_error": delta}
        if delta > 1e-12:
            raise RuntimeError(f"metafor fixture不一致：{key} (误差={delta})")
    if meta["n_cohorts"] != fixture["expected"]["degrees_of_freedom"] + 1:
        raise RuntimeError("metafor fixture自由度不一致。")
    return {"status": "PASS", "fixture_path": relative(METAFOR_FIXTURE_PATH), "fixture_sha256": sha256_file(METAFOR_FIXTURE_PATH), "metafor_version": fixture["metafor_version"], "comparisons": comparisons}


def figure2(primary: pd.DataFrame, meta: dict[str, Any], out_dir: Path) -> None:
    rows = primary.loc[primary["is_primary_estimand"], ["cohort", "estimate", "ci_lower", "ci_upper"]].copy()
    rows["participants"] = [EXPECTED[c]["participants"] for c in rows["cohort"]]
    meta_row = pd.DataFrame([{"cohort": meta["cohort"], "estimate": meta["estimate"], "ci_lower": meta["ci_lower"], "ci_upper": meta["ci_upper"], "participants": meta["participants"]}])
    rows = pd.concat([rows, meta_row], ignore_index=True)
    rows.to_csv(out_dir / "figure_2_source.csv", index=False, encoding="utf-8")
    lower = float(rows["ci_lower"].min()); upper = float(rows["ci_upper"].max()); span = max(upper - lower, 0.001); pad = 0.12 * span
    fig, ax = plt.subplots(figsize=(7.09, 3.35), constrained_layout=True)
    y = np.arange(len(rows))[::-1]
    for yi, row in zip(y, rows.itertuples(index=False)):
        is_overall = row.cohort.startswith("Overall")
        color = "#0072B2" if not is_overall else "#222222"
        ax.errorbar(row.estimate, yi, xerr=[[row.estimate - row.ci_lower], [row.ci_upper - row.estimate]], fmt="D" if is_overall else "o", color=color, ecolor=color, elinewidth=1.1, capsize=3.0, markersize=5.5, markeredgecolor="black", markeredgewidth=0.35)
        value = f"{row.estimate:.4f} ({row.ci_lower:.4f}, {row.ci_upper:.4f})"
        ax.text(1.01, yi, value, transform=ax.get_yaxis_transform(), va="center", ha="left", fontsize=8.2)
    ax.axvline(0, color="#555555", linewidth=0.8)
    ax.set_yticks(y, rows["cohort"])
    ax.set_xlabel("Adjusted difference in annual FI change (FI units/year)", fontsize=8.5)
    ax.set_xlim(lower - pad, upper + pad)
    ax.tick_params(axis="both", labelsize=8.2)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.7); ax.spines["bottom"].set_linewidth(0.7)
    fig.savefig(out_dir / "figure_2.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "figure_2.png", dpi=600, bbox_inches="tight")
    fig.savefig(out_dir / "figure_2.tiff", dpi=600, bbox_inches="tight", pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)
    normalize_rgb_raster(out_dir / "figure_2.png")
    normalize_rgb_raster(out_dir / "figure_2.tiff")
    Image.open(out_dir / "figure_2.png").convert("L").save(out_dir / "figure_2_grayscale.png")
    (out_dir / "figure_2_caption.md").write_text("Figure 2. Wealth recovery and subsequent frailty progression, by cohort. Points and horizontal lines show adjusted differences in annual FI change and 95% confidence intervals from cohort-specific unweighted linear mixed-effects models. Negative values indicate slower annual FI increase with substantial wealth recovery. Overall (random-effects) is a secondary three-cohort synthesis.", encoding="utf-8")


def figure3(predictions_df: pd.DataFrame, out_dir: Path) -> dict[str, Any]:
    minimum = float(predictions_df["ci_lower"].min()); maximum = float(predictions_df["ci_upper"].max()); span = max(maximum - minimum, 0.05); pad = 0.10 * span
    lower = max(0.0, math.floor((minimum - pad) / 0.05) * 0.05); upper = min(1.0, math.ceil((maximum + pad) / 0.05) * 0.05)
    if upper <= lower:
        upper = min(1.0, lower + 0.05)
    predictions_df.to_csv(out_dir / "figure_3_source.csv", index=False, encoding="utf-8")
    colors = {"Substantial wealth recovery": "#0072B2", "Persistent loss or limited recovery": "#D55E00"}
    styles = {"Substantial wealth recovery": "-", "Persistent loss or limited recovery": "--"}
    fig, axes = plt.subplots(1, 3, figsize=(7.09, 3.75), sharex=True, sharey=True, constrained_layout=True)
    for ax, cohort in zip(axes, COHORTS):
        subset = predictions_df[predictions_df["cohort"].eq(cohort)]
        for group in colors:
            s = subset[subset["recovery_status"].eq(group)]
            ax.plot(s["time_years"], s["estimate"], color=colors[group], linestyle=styles[group], linewidth=1.5, label=group)
            ax.fill_between(s["time_years"].to_numpy(), s["ci_lower"].to_numpy(), s["ci_upper"].to_numpy(), color=colors[group], alpha=0.14, linewidth=0)
        ax.set_title(cohort, fontsize=9, fontweight="bold")
        ax.set_xlim(0, 8); ax.set_xticks([0, 2, 4, 6, 8]); ax.tick_params(labelsize=8)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        ax.spines["left"].set_linewidth(0.7); ax.spines["bottom"].set_linewidth(0.7)
    axes[0].set_ylabel("Frailty index", fontsize=8.5)
    axes[1].set_xlabel("Time since landmark, years", fontsize=8.5)
    axes[0].legend(frameon=False, fontsize=7.7, loc="upper left")
    for ax in axes:
        ax.set_ylim(lower, upper)
    fig.savefig(out_dir / "figure_3.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "figure_3.png", dpi=600, bbox_inches="tight")
    fig.savefig(out_dir / "figure_3.tiff", dpi=600, bbox_inches="tight", pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)
    normalize_rgb_raster(out_dir / "figure_3.png")
    normalize_rgb_raster(out_dir / "figure_3.tiff")
    Image.open(out_dir / "figure_3.png").convert("L").save(out_dir / "figure_3_grayscale.png")
    (out_dir / "figure_3_caption.md").write_text("Figure 3. Adjusted frailty trajectories after the landmark, by wealth-recovery status and cohort. Lines show fixed-effect marginal predictions from the same primary mixed-effects models used for Figure 2; shaded areas are pointwise 95% confidence intervals. Each participant contributed once under each recovery scenario with the observed prespecified covariates, and random effects were set to zero. The display range was restricted to 0–8 years because of observed follow-up support.", encoding="utf-8")
    return {"y_axis_lower": lower, "y_axis_upper": upper, "raw_ci_lower": minimum, "raw_ci_upper": maximum}


def write_results_draft(primary_rows: pd.DataFrame, meta: dict[str, Any], output_path: Path) -> None:
    """按合同写出三段结果文字，并固定cohort顺序。"""
    ordered = primary_rows.set_index("cohort").loc[list(COHORTS)].reset_index()
    cohort_sentences = [
        f"In {row.cohort}, substantial wealth recovery was associated with an adjusted difference in annual FI change of {row.estimate:.4f} (95% CI {row.ci_lower:.4f} to {row.ci_upper:.4f}) compared with persistent loss or limited recovery."
        for row in ordered.itertuples(index=False)
    ]
    paragraphs = [
        " ".join(cohort_sentences),
        f"Across the three cohorts, the secondary random-effects estimate was {meta['estimate']:.4f} (95% CI {meta['ci_lower']:.4f} to {meta['ci_upper']:.4f}; τ²={meta['tau2']:.6f}; I²={meta['i2_percent']:.1f}%).",
        "Figure 3 shows the corresponding adjusted frailty trajectories over the prespecified 0–8-year display range, allowing the landmark FI levels and subsequent separation of the two recovery groups to be viewed directly.",
    ]
    output_path.write_text("## 3.2. Wealth Recovery and Subsequent Frailty Progression\n\n" + "\n\n".join(paragraphs) + "\n", encoding="utf-8")


def build(output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    input_info = verify_inputs()
    contract = load_contract()
    all_estimates: list[pd.DataFrame] = []
    all_cov: list[pd.DataFrame] = []
    all_predictions: list[pd.DataFrame] = []
    constants_rows: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    model_anchors = []
    freeze = json.loads((ROOT / "00_admin/panel_manifests/analysis_panel/analysis_panel_v1_freeze.json").read_text(encoding="utf-8"))
    for cohort in COHORTS:
        frame = pd.read_parquet(ROOT / freeze["cohorts"][cohort]["parquet_path"])
        frame, landmark, constants = prepare_frame(frame, cohort)
        result, _, estimates, cov, diag = fit_model(frame, cohort)
        all_estimates.append(estimates); all_cov.append(cov); diagnostics.append(diag)
        all_predictions.append(predictions(result, frame, landmark, cohort))
        for variable, values in constants.items():
            constants_rows.append({"cohort": cohort, "variable": variable, **values})
        model_anchors.append({"cohort": cohort, "participants": len(landmark), "later_fi_observations": int((frame["fi_time_role"] == "later").sum()), "model_rows": len(frame), "recovery_groups": landmark["recovery_status"].value_counts().to_dict()})
    estimates_df = pd.concat(all_estimates, ignore_index=True)
    cov_df = pd.concat(all_cov, ignore_index=True)
    prediction_df = pd.concat(all_predictions, ignore_index=True)
    meta = meta_analysis(estimates_df)
    metafor_fixture_validation = verify_metafor_fixture(meta, estimates_df)
    out_models = output_dir / "Primary Models"; out_fig2 = output_dir / "Figure 2"; out_fig3 = output_dir / "Figure 3"
    out_models.mkdir(exist_ok=True); out_fig2.mkdir(exist_ok=True); out_fig3.mkdir(exist_ok=True)
    estimates_df.to_csv(out_models / "primary_lmm_fixed_effects.csv", index=False, encoding="utf-8")
    cov_df.to_csv(out_models / "primary_lmm_fixed_effect_covariance.csv", index=False, encoding="utf-8")
    pd.DataFrame(constants_rows).to_csv(out_models / "primary_lmm_standardization_constants.csv", index=False, encoding="utf-8")
    pd.DataFrame(diagnostics).to_json(out_models / "primary_lmm_diagnostics.json", orient="records", indent=2, force_ascii=False)
    primary_rows = estimates_df[estimates_df["is_primary_estimand"]].copy()
    primary_rows["analysis_specification"] = "Primary unweighted ML random-intercept LMM"
    primary_rows["participants"] = primary_rows["cohort"].map({c: EXPECTED[c]["participants"] for c in COHORTS})
    primary_rows["model_rows"] = primary_rows["cohort"].map({c: EXPECTED[c]["rows"] for c in COHORTS})
    primary_rows["later_fi_observations"] = primary_rows["cohort"].map({c: EXPECTED[c]["later"] for c in COHORTS})
    primary_rows[["cohort", "analysis_specification", "participants", "model_rows", "later_fi_observations", "estimate", "se", "ci_lower", "ci_upper"]].to_csv(out_models / "table_s6_primary_rows.csv", index=False, encoding="utf-8")
    pd.DataFrame([meta]).to_csv(out_models / "primary_lmm_meta_summary.csv", index=False, encoding="utf-8")
    prediction_df.to_csv(out_fig3 / "figure_3_prediction_grid.csv", index=False, encoding="utf-8")
    figure2(primary_rows, meta, out_fig2)
    y_axis = figure3(prediction_df, out_fig3)
    table_anchor_validation = verify_table_anchors(model_anchors)
    result_object = {
        "result_object_version": "primary_lmm_result_object_v1",
        "contract_version": contract.get("contract_version"),
        "fixed_effects": estimates_df.to_dict(orient="records"),
        "fixed_effect_covariance": cov_df.to_dict(orient="records"),
        "standardization_constants": constants_rows,
        "diagnostics": diagnostics,
        "primary_estimand_rows": primary_rows.to_dict(orient="records"),
        "random_effects_meta": meta,
        "figure_3_predictions": prediction_df.to_dict(orient="records"),
        "figure_3_y_axis": y_axis,
        "input_info": input_info,
    }
    (out_models / "primary_lmm_result_object.json").write_text(json.dumps(result_object, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_models / "primary_lmm_model_spec.json").write_text(json.dumps({"contract": contract, "formula": FORMULA, "model_anchors": model_anchors, "input_info": input_info}, ensure_ascii=False, indent=2), encoding="utf-8")
    write_results_draft(primary_rows, meta, output_dir / "results_3_2_draft.md")
    validation = {"status": "PASS", "generated_utc": now_utc(), "model_anchors": model_anchors, "table_anchor_validation": table_anchor_validation, "metafor_fixture_validation": metafor_fixture_validation, "figure_3_y_axis": y_axis, "meta": meta, "no_pvalues_or_stars_in_figures": True, "raw_wealth_loss_untransformed": True, "all_followup_used_for_model": True, "figure_3_display_range_years": [0, 8]}
    (output_dir / "section_3_2_validation.json").write_text(json.dumps(validation, ensure_ascii=False, indent=2), encoding="utf-8")
    files = [p for p in output_dir.rglob("*") if p.is_file()]
    manifest = {"release_version": "section_3_2_v1", "generated_utc": now_utc(), "git_commit": git_value(["rev-parse", "HEAD"]), "git_dirty": bool(git_value(["status", "--porcelain"])), "inputs": {"analysis_panel_freeze": input_info, "contract": {"path": relative(ROOT / "00_admin/model_manifests/primary_lmm/primary_lmm_v1_contract.json"), "sha256": sha256_file(ROOT / "00_admin/model_manifests/primary_lmm/primary_lmm_v1_contract.json")}}, "outputs": {output_manifest_path(p, output_dir): sha256_file(p) for p in files}, "runtime": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__, "statsmodels": statsmodels.__version__, "matplotlib": matplotlib.__version__}, "release_scope": "results_3_2_primary_lmm_figures", "submission_ready": False}
    (output_dir / "section_3_2_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return validation


UNCHANGED_V1_ASSETS = (
    "Primary Models/primary_lmm_fixed_effects.csv",
    "Primary Models/primary_lmm_fixed_effect_covariance.csv",
    "Primary Models/primary_lmm_standardization_constants.csv",
    "Primary Models/primary_lmm_diagnostics.json",
    "Primary Models/primary_lmm_model_spec.json",
    "Primary Models/table_s6_primary_rows.csv",
    "Figure 3/figure_3.pdf",
    "Figure 3/figure_3.png",
    "Figure 3/figure_3.tiff",
    "Figure 3/figure_3_grayscale.png",
    "Figure 3/figure_3_prediction_grid.csv",
    "Figure 3/figure_3_source.csv",
    "Figure 3/figure_3_caption.md",
)


def _copy_v1_active_assets(v1_dir: Path, candidate_dir: Path) -> None:
    """将v1 active release复制到候选目录，不复制历史releases或旧freeze。"""
    candidate_dir.mkdir(parents=True, exist_ok=False)
    for item in v1_dir.iterdir():
        if item.name == "releases" or item.name == "section_3_2_freeze.json":
            continue
        destination = candidate_dir / item.name
        if item.is_dir():
            shutil.copytree(item, destination)
        else:
            shutil.copy2(item, destination)


def _file_hashes(directory: Path, relative_paths: tuple[str, ...]) -> dict[str, str]:
    return {relative_path: sha256_file(directory / relative_path) for relative_path in relative_paths}


def verify_v1_archive(active_dir: Path, archive_dir: Path) -> dict[str, Any]:
    """验证归档v1与归档前active v1逐文件一致。"""
    freeze_path = active_dir / "section_3_2_freeze.json"
    if not freeze_path.is_file():
        raise RuntimeError("active目录缺少section_3_2_v1 freeze。")
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    if freeze.get("freeze_version") != "section_3_2_v1" or freeze.get("freeze_status") != "FROZEN":
        raise RuntimeError("active目录不是预期的section_3_2_v1。")
    if not archive_dir.is_dir():
        raise RuntimeError("v1归档目录不存在。")
    mismatches: list[str] = []
    for item in active_dir.iterdir():
        if item.name == "releases":
            continue
        archived = archive_dir / item.name
        if item.is_dir():
            for source_file in item.rglob("*"):
                if source_file.is_file():
                    target = archive_dir / source_file.relative_to(active_dir)
                    if not target.is_file() or sha256_file(source_file) != sha256_file(target):
                        mismatches.append(source_file.relative_to(active_dir).as_posix())
        elif not archived.is_file() or sha256_file(item) != sha256_file(archived):
            mismatches.append(item.name)
    if mismatches:
        raise RuntimeError(f"v1归档哈希不一致：{mismatches}")
    return {"status": "PASS", "archive_path": relative(archive_dir), "archive_sha256": sha256_file(archive_dir / "section_3_2_manifest.json"), "asset_count": sum(1 for p in archive_dir.rglob("*") if p.is_file())}


def correct_meta_v2(v1_dir: Path, output_dir: Path) -> dict[str, Any]:
    """从冻结v1复制完整候选，仅重建metafor兼容的secondary meta派生资产。"""
    v1_freeze_path = v1_dir / "section_3_2_freeze.json"
    v1_manifest_path = v1_dir / "section_3_2_manifest.json"
    if not v1_freeze_path.is_file() or not v1_manifest_path.is_file():
        raise RuntimeError("找不到冻结section_3_2_v1的freeze或manifest。")
    v1_freeze = json.loads(v1_freeze_path.read_text(encoding="utf-8"))
    if v1_freeze.get("freeze_status") != "FROZEN" or v1_freeze.get("freeze_version") != "section_3_2_v1":
        raise RuntimeError("meta修正只接受冻结section_3_2_v1。")
    candidate = output_dir.resolve()
    _copy_v1_active_assets(v1_dir.resolve(), candidate)
    input_info = verify_inputs()
    fixed_path = candidate / "Primary Models/primary_lmm_fixed_effects.csv"
    # pandas 3默认解析会截短长浮点；round_trip保证冻结β3/SE逐位回读。
    fixed = pd.read_csv(fixed_path, float_precision="round_trip")
    primary = fixed.loc[fixed["is_primary_estimand"]].copy()
    if primary["cohort"].tolist() != list(COHORTS):
        raise RuntimeError("v1 cohort primary rows顺序或集合发生漂移。")
    unchanged_hashes = _file_hashes(v1_dir, UNCHANGED_V1_ASSETS)
    candidate_unchanged_hashes = _file_hashes(candidate, UNCHANGED_V1_ASSETS)
    if unchanged_hashes != candidate_unchanged_hashes:
        raise RuntimeError("复制v1 cohort model或Figure 3资产时发生哈希漂移。")
    meta = meta_analysis(primary)
    fixture_validation = verify_metafor_fixture(meta, fixed)
    pd.DataFrame([meta]).to_csv(candidate / "Primary Models/primary_lmm_meta_summary.csv", index=False, encoding="utf-8")
    figure2(primary, meta, candidate / "Figure 2")
    result_object_path = candidate / "Primary Models/primary_lmm_result_object.json"
    result_object = json.loads(result_object_path.read_text(encoding="utf-8"))
    result_object["result_object_version"] = "primary_lmm_result_object_v2"
    result_object["random_effects_meta"] = meta
    result_object["metafor_fixture_validation"] = fixture_validation
    result_object_path.write_text(json.dumps(result_object, ensure_ascii=False, indent=2), encoding="utf-8")
    write_results_draft(primary, meta, candidate / "results_3_2_draft.md")
    validation = {
        "status": "PASS",
        "generated_utc": now_utc(),
        "release_version": "section_3_2_v2",
        "supersedes": "section_3_2_v1",
        "correction_scope": "secondary_meta_only",
        "cohort_primary_results_changed": False,
        "scientific_conclusion_changed": False,
        "model_anchors": json.loads((candidate / "section_3_2_validation.json").read_text(encoding="utf-8")).get("model_anchors", []),
        "table_anchor_validation": verify_table_anchors([]),
        "metafor_fixture_validation": fixture_validation,
        "v1_unchanged_asset_hashes": unchanged_hashes,
        "v2_unchanged_asset_hashes": candidate_unchanged_hashes,
        "figure_3_y_axis": json.loads((candidate / "section_3_2_validation.json").read_text(encoding="utf-8")).get("figure_3_y_axis", {}),
        "meta": meta,
        "figure_3_display_range_years": [0, 8],
    }
    (candidate / "section_3_2_validation.json").write_text(json.dumps(validation, ensure_ascii=False, indent=2), encoding="utf-8")
    # manifest自身不纳入outputs哈希：它是在此之后写入，避免自引用造成CheckOnly漂移。
    files = [p for p in candidate.rglob("*") if p.is_file() and p.name != "section_3_2_manifest.json"]
    contract_path = ROOT / "00_admin/model_manifests/primary_lmm/primary_lmm_v1_contract.json"
    code_path = ROOT / "02_code/03_results/section_3_2/run_section_3_2.py"
    manifest = {
        "release_version": "section_3_2_v2",
        "generated_utc": now_utc(),
        "git_commit": git_value(["rev-parse", "HEAD"]),
        "git_dirty": bool(git_value(["status", "--porcelain"])),
        "supersedes": {"release_version": "section_3_2_v1", "freeze_sha256": sha256_file(v1_freeze_path), "manifest_sha256": sha256_file(v1_manifest_path)},
        "v1_active_asset_hashes": {p.relative_to(v1_dir).as_posix(): sha256_file(p) for p in v1_dir.rglob("*") if p.is_file() and "releases" not in p.relative_to(v1_dir).parts},
        "correction_scope": "secondary_meta_only",
        "inputs": {"analysis_panel_freeze": input_info, "v1_freeze": {"path": relative(v1_freeze_path), "sha256": sha256_file(v1_freeze_path)}, "v1_manifest": {"path": relative(v1_manifest_path), "sha256": sha256_file(v1_manifest_path)}, "contract": {"path": relative(contract_path), "sha256": sha256_file(contract_path)}, "metafor_fixture": {"path": relative(METAFOR_FIXTURE_PATH), "sha256": sha256_file(METAFOR_FIXTURE_PATH)}, "code": {"path": relative(code_path), "sha256": sha256_file(code_path)}},
        "outputs": {output_manifest_path(p, candidate): sha256_file(p) for p in files},
        "runtime": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__, "statsmodels": statsmodels.__version__, "matplotlib": matplotlib.__version__},
        "release_scope": "results_3_2_secondary_meta_correction",
        "submission_ready": False,
    }
    (candidate / "section_3_2_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return validation


def check(output_dir: Path) -> None:
    verify_inputs()
    required = [
        output_dir / "section_3_2_manifest.json",
        output_dir / "section_3_2_validation.json",
        output_dir / "Primary Models/primary_lmm_result_object.json",
        output_dir / "Primary Models/primary_lmm_fixed_effects.csv",
        output_dir / "Primary Models/primary_lmm_fixed_effect_covariance.csv",
        output_dir / "Primary Models/primary_lmm_standardization_constants.csv",
        output_dir / "Primary Models/primary_lmm_diagnostics.json",
        output_dir / "Primary Models/primary_lmm_meta_summary.csv",
        output_dir / "Primary Models/table_s6_primary_rows.csv",
        output_dir / "Figure 2/figure_2_source.csv",
        output_dir / "Figure 3/figure_3_prediction_grid.csv",
        output_dir / "Figure 2/figure_2.png",
        output_dir / "Figure 2/figure_2.pdf",
        output_dir / "Figure 2/figure_2.tiff",
        output_dir / "Figure 3/figure_3.png",
        output_dir / "Figure 3/figure_3.pdf",
        output_dir / "Figure 3/figure_3.tiff",
    ]
    missing = [str(p) for p in required if not p.is_file()]
    if missing:
        raise RuntimeError(f"3.2输出缺失：{missing}")
    validation = json.loads((output_dir / "section_3_2_validation.json").read_text(encoding="utf-8"))
    if validation.get("status") != "PASS":
        raise RuntimeError("3.2 validation不是PASS。")
    expected_models = {c: EXPECTED[c] for c in COHORTS}
    for anchor in validation.get("model_anchors", []):
        expected = expected_models.get(anchor.get("cohort"))
        if expected is None or anchor.get("participants") != expected["participants"] or anchor.get("later_fi_observations") != expected["later"] or anchor.get("model_rows") != expected["rows"]:
            raise RuntimeError("3.2模型样本锚点不匹配。")
        if anchor["model_rows"] != anchor["participants"] + anchor["later_fi_observations"]:
            raise RuntimeError("模型记录未闭合为landmark加later FI。")
    diagnostics = json.loads((output_dir / "Primary Models/primary_lmm_diagnostics.json").read_text(encoding="utf-8"))
    if len(diagnostics) != 3:
        raise RuntimeError("缺少cohort模型诊断。")
    for diag in diagnostics:
        if not (diag["design_matrix_columns"] == diag["design_matrix_rank"] == 42 and diag["lbfgs_converged"] and diag["bfgs_converged"] and not diag["hessian_singular"]):
            raise RuntimeError(f"{diag['cohort']}模型诊断未通过。")
        if diag["beta3_optimizer_absolute_difference"] >= 1e-6 or diag["loglik_optimizer_absolute_difference"] >= 1e-4 or diag["random_intercept_variance"] <= 0 or diag["residual_variance"] <= 0:
            raise RuntimeError(f"{diag['cohort']}优化器或方差诊断未通过。")
    if validation.get("table_anchor_validation", {}).get("status") != "PASS":
        raise RuntimeError("Table 1/Table S5重叠锚点未通过。")
    fig2 = pd.read_csv(output_dir / "Figure 2/figure_2_source.csv")
    if fig2["cohort"].tolist() != ["HRS", "ELSA", "KLoSA", "Overall (random-effects)"] or fig2.iloc[-1]["participants"] != "3 cohorts":
        raise RuntimeError("Figure 2来源行序或overall参与者标签不符合合同。")
    fig3 = pd.read_csv(output_dir / "Figure 3/figure_3_prediction_grid.csv")
    if set(fig3["cohort"].unique().tolist()) != set(COHORTS) or len(fig3) != 198 or fig3["time_years"].min() != 0 or fig3["time_years"].max() != 8:
        raise RuntimeError("Figure 3预测网格不符合0–8年合同。")
    manifest = json.loads((output_dir / "section_3_2_manifest.json").read_text(encoding="utf-8"))
    for relative_path, expected_hash in manifest.get("outputs", {}).items():
        path = ROOT / relative_path
        if not path.is_file() or sha256_file(path) != expected_hash:
            raise RuntimeError(f"输出哈希漂移：{relative_path}")
    for path in [output_dir / "Figure 2/figure_2.png", output_dir / "Figure 3/figure_3.png"]:
        with Image.open(path) as image:
            if image.width < 1000 or image.height < 300 or image.mode != "RGB" or min(image.info.get("dpi", (0, 0))) < 599:
                raise RuntimeError(f"图像分辨率不足：{path}")
    for path in [output_dir / "Figure 2/figure_2.tiff", output_dir / "Figure 3/figure_3.tiff"]:
        with Image.open(path) as image:
            if image.mode != "RGB" or min(image.info.get("dpi", (0, 0))) < 599:
                raise RuntimeError(f"TIFF不是600 dpi RGB：{path}")
    print("3.2 CheckOnly PASS")


def check_v2(output_dir: Path) -> None:
    """验证active section_3_2_v2及其v1不可变资产。"""
    check(output_dir)
    manifest = json.loads((output_dir / "section_3_2_manifest.json").read_text(encoding="utf-8"))
    validation = json.loads((output_dir / "section_3_2_validation.json").read_text(encoding="utf-8"))
    if manifest.get("release_version") != "section_3_2_v2" or validation.get("release_version") != "section_3_2_v2":
        raise RuntimeError("active结果不是section_3_2_v2。")
    if validation.get("supersedes") != "section_3_2_v1" or validation.get("correction_scope") != "secondary_meta_only" or validation.get("cohort_primary_results_changed") is not False or validation.get("scientific_conclusion_changed") is not False:
        raise RuntimeError("v2修正范围标记不符合合同。")
    if validation.get("metafor_fixture_validation", {}).get("status") != "PASS":
        raise RuntimeError("v2 metafor fixture未通过。")
    archive = output_dir / "releases/section_3_2_v1"
    if not archive.is_dir():
        raise RuntimeError("缺少section_3_2_v1历史归档。")
    for relative_path, expected_hash in manifest.get("v1_active_asset_hashes", {}).items():
        archived_path = archive / relative_path
        if not archived_path.is_file() or sha256_file(archived_path) != expected_hash:
            raise RuntimeError(f"v1历史归档哈希漂移：{relative_path}")
    # 这些资产在v2中必须同时保持active与v1归档逐字节一致。
    for relative_path in UNCHANGED_V1_ASSETS:
        active_path = output_dir / relative_path
        archived_path = archive / relative_path
        if not active_path.is_file() or not archived_path.is_file() or sha256_file(active_path) != sha256_file(archived_path):
            raise RuntimeError(f"v2不可变资产哈希漂移：{relative_path}")
    draft = (output_dir / "results_3_2_draft.md").read_text(encoding="utf-8").strip().split("\n\n")
    if len(draft) != 4 or not draft[0].startswith("## 3.2"):
        raise RuntimeError("Results 3.2 draft不是标题加三段。")
    meta = pd.read_csv(output_dir / "Primary Models/primary_lmm_meta_summary.csv", float_precision="round_trip").iloc[0].to_dict()
    obj_meta = json.loads((output_dir / "Primary Models/primary_lmm_result_object.json").read_text(encoding="utf-8"))["random_effects_meta"]
    for field in ("estimate", "se", "ci_lower", "ci_upper", "tau2", "i2_percent", "kh_scale", "q_fixed"):
        if not np.isclose(float(meta[field]), float(obj_meta[field]), rtol=0.0, atol=1e-15):
            raise RuntimeError(f"v2 meta summary与result object不一致：{field}")
    if (output_dir / "section_3_2_freeze.json").is_file():
        freeze = json.loads((output_dir / "section_3_2_freeze.json").read_text(encoding="utf-8"))
        if freeze.get("freeze_status") != "FROZEN" or freeze.get("freeze_version") != "section_3_2_v2":
            raise RuntimeError("v2 freeze记录不一致。")
    print("3.2 v2 CheckOnly PASS")


def freeze(output_dir: Path) -> None:
    manifest = json.loads((output_dir / "section_3_2_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("release_version") == "section_3_2_v2":
        check_v2(output_dir)
    else:
        check(output_dir)
    freeze_path = output_dir / "section_3_2_freeze.json"
    if freeze_path.exists():
        raise RuntimeError("3.2已冻结；拒绝覆盖。")
    manifest = json.loads((output_dir / "section_3_2_manifest.json").read_text(encoding="utf-8"))
    freeze_version = manifest.get("release_version", "section_3_2_v1")
    freeze_path.write_text(json.dumps({"freeze_status": "FROZEN", "freeze_version": freeze_version, "frozen_utc": now_utc(), "manifest_sha256": sha256_file(output_dir / "section_3_2_manifest.json"), "manifest": manifest}, ensure_ascii=False, indent=2), encoding="utf-8")
    print("3.2 Freeze PASS")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("build", "correct-meta-v2", "verify-v1-archive", "check-only", "check-v2", "freeze"))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--v1-dir")
    parser.add_argument("--archive-dir")
    args = parser.parse_args()
    output_dir = Path(args.output_dir).resolve()
    if args.command == "build":
        build(output_dir)
    elif args.command == "correct-meta-v2":
        v1_dir = Path(args.v1_dir).resolve() if args.v1_dir else output_dir
        correct_meta_v2(v1_dir, output_dir)
    elif args.command == "verify-v1-archive":
        if not args.v1_dir or not args.archive_dir:
            raise RuntimeError("verify-v1-archive需要--v1-dir和--archive-dir。")
        verify_v1_archive(Path(args.v1_dir).resolve(), Path(args.archive_dir).resolve())
    elif args.command == "check-only":
        check(output_dir)
    elif args.command == "check-v2":
        check_v2(output_dir)
    else:
        freeze(output_dir)


if __name__ == "__main__":
    main()
