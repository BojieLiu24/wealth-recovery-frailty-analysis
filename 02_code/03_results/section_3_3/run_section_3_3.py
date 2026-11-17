from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import platform
import shutil
import subprocess
import sys
import tempfile
import warnings
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import patsy
from scipy.stats import t as student_t
import statsmodels.api as sm


ROOT = Path(__file__).resolve().parents[3]
OUTPUT_ROOT = ROOT / "03_output/3.3 Robustness of the Main Findings"
PRIMARY_DIR = ROOT / "03_output/3.2 Wealth Recovery and Subsequent Frailty Progression"
CONTRACT_DIR = ROOT / "00_admin/model_manifests/sensitivity_lmm"
PRIMARY_CONSTANTS = PRIMARY_DIR / "Primary Models/primary_lmm_standardization_constants.csv"
COHORTS = ("HRS", "ELSA", "KLoSA")
CONTINUOUS = ("age", "household_size", "household_income_rank", "wealth_loss_fraction", "fi_t")
CATEGORICAL = ("sex", "education", "marital_status", "labour_force_status", "current_smoking", "wealth_quintile")


def load_primary_module():
    path = ROOT / "02_code/03_results/section_3_2/run_section_3_2.py"
    spec = importlib.util.spec_from_file_location("primary_lmm_runtime", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("无法加载冻结primary模型实现")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PRIMARY = load_primary_module()


def now_utc() -> str:
    return datetime.now(UTC).isoformat()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def rel(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def verify_inputs() -> dict[str, Any]:
    PRIMARY.verify_inputs()
    freeze_path = PRIMARY_DIR / "section_3_2_freeze.json"
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    if freeze.get("freeze_status") != "FROZEN" or freeze.get("freeze_version") != "section_3_2_v3":
        raise RuntimeError("3.3只接受冻结section_3_2_v3")
    required = [
        CONTRACT_DIR / "sensitivity_lmm_v1_contract.json",
        CONTRACT_DIR / "weighted_sensitivity_v1_contract.json",
        CONTRACT_DIR / "mi_sensitivity_v1_preflight_candidate.json",
        CONTRACT_DIR / "mi_sensitivity_v1_contract.json",
        PRIMARY_CONSTANTS,
    ]
    for path in required:
        if not path.is_file():
            raise RuntimeError(f"缺少合同输入：{path}")
    return {rel(p): sha256_file(p) for p in [freeze_path, *required]}


def load_panels() -> dict[str, pd.DataFrame]:
    freeze = json.loads((ROOT / "00_admin/panel_manifests/analysis_panel/analysis_panel_v1_freeze.json").read_text(encoding="utf-8"))
    return {c: pd.read_parquet(ROOT / freeze["cohorts"][c]["parquet_path"]) for c in COHORTS}


def constants_by_cohort() -> dict[str, dict[str, tuple[float, float]]]:
    table = pd.read_csv(PRIMARY_CONSTANTS, float_precision="round_trip")
    result: dict[str, dict[str, tuple[float, float]]] = {}
    for cohort in COHORTS:
        block = table.loc[table["cohort"].eq(cohort)].set_index("variable")
        if set(block.index) != set(CONTINUOUS):
            raise RuntimeError(f"{cohort}: primary标准化常数不完整")
        result[cohort] = {v: (float(block.loc[v, "mean"]), float(block.loc[v, "sample_sd"])) for v in CONTINUOUS}
    return result


def prepare(frame: pd.DataFrame, cohort: str, constants: dict[str, tuple[float, float]], analysis: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    if analysis in {"recovery_25", "recovery_100", "common22"}:
        frame = frame.loc[frame["primary_complete_case_eligible"].astype(bool)].copy()
    elif analysis == "weighted":
        frame = frame.loc[frame["weighted_sensitivity_eligible"].astype(bool)].copy()
    else:
        raise RuntimeError(f"未知analysis：{analysis}")
    frame["person_id"] = frame["person_id"].astype(str)
    if analysis == "common22":
        valid = frame.loc[frame["fi_common22"].notna()].copy()
        role = valid.groupby(["person_id", "fi_time_role"]).size().unstack(fill_value=0)
        eligible = role.index[(role.get("landmark", 0).eq(1)) & (role.get("later", 0).ge(1))]
        frame = valid.loc[valid["person_id"].isin(eligible)].copy()
        frame["fi_primary"] = frame["fi_common22"].astype(float)
    landmark = frame.loc[frame["fi_time_role"].eq("landmark")].copy()
    later = frame.loc[frame["fi_time_role"].eq("later")].copy()
    if landmark.groupby("person_id").size().ne(1).any() or later.groupby("person_id").size().lt(1).any():
        raise RuntimeError(f"{cohort}/{analysis}: landmark/later支持不合法")
    if frame[["person_id", "wave"]].duplicated().any():
        raise RuntimeError(f"{cohort}/{analysis}: 重复person-wave")
    if analysis == "recovery_25":
        frame["recovery"] = frame["recovery_fraction_raw"].astype(float).ge(0.25).astype(int)
    elif analysis == "recovery_100":
        frame["recovery"] = frame["recovery_fraction_raw"].astype(float).ge(1.0).astype(int)
    else:
        frame["recovery"] = frame["recovery_status"].eq("substantial_recovery").astype(int)
    if landmark.assign(recovery=frame.loc[landmark.index, "recovery"])["recovery"].nunique() != 2:
        raise RuntimeError(f"{cohort}/{analysis}: recovery缺少两个组")
    for variable, (mean, sd) in constants.items():
        if not np.isfinite(mean) or not np.isfinite(sd) or sd <= 0:
            raise RuntimeError(f"{cohort}/{variable}: 冻结标准化常数无效")
        frame[f"{variable}_z"] = (pd.to_numeric(frame[variable]).astype(float) - mean) / sd
    for variable in CATEGORICAL:
        frame[variable] = frame[variable].astype(str)
    if frame[["fi_primary", "time_years", "recovery", *[f"{v}_z" for v in CONTINUOUS]]].isna().any().any():
        raise RuntimeError(f"{cohort}/{analysis}: 模型字段存在缺失")
    anchor = {
        "cohort": cohort,
        "analysis_spec": analysis,
        "participants": int(frame["person_id"].nunique()),
        "fi_observations": int(len(frame)),
        "later_fi_observations": int(len(later)),
        "recovery_1": int(landmark.assign(recovery=frame.loc[landmark.index, "recovery"])["recovery"].sum()),
        "recovery_0": int(len(landmark) - landmark.assign(recovery=frame.loc[landmark.index, "recovery"])["recovery"].sum()),
        "shock_time_frailty_adjustment": "fi_t_primary_unchanged",
    }
    return frame, anchor


def primary_row(estimates: pd.DataFrame, spec_name: str, anchor: dict[str, Any]) -> dict[str, Any]:
    row = estimates.loc[estimates["is_primary_estimand"]].iloc[0]
    return {
        "analysis_spec": spec_name,
        "cohort": anchor["cohort"],
        "is_overall": False,
        "participants": anchor["participants"],
        "fi_observations": anchor["fi_observations"],
        "estimate": float(row["estimate"]),
        "se": float(row["se"]),
        "ci_lower": float(row["ci_lower"]),
        "ci_upper": float(row["ci_upper"]),
    }


def add_meta(rows: list[dict[str, Any]], spec_name: str) -> dict[str, Any]:
    df = pd.DataFrame(rows)
    temp = df.rename(columns={"analysis_spec": "term"}).copy()
    temp["is_primary_estimand"] = True
    meta = PRIMARY.meta_analysis(temp)
    rows.append({"analysis_spec": spec_name, "cohort": "Overall (random-effects)", "is_overall": True,
                 "participants": "3 cohorts", "fi_observations": None, "estimate": meta["estimate"],
                 "se": meta["se"], "ci_lower": meta["ci_lower"], "ci_upper": meta["ci_upper"]})
    return meta


def write_domain(domain: str, rows: list[dict[str, Any]], diagnostics: list[dict[str, Any]], anchors: list[dict[str, Any]], metas: dict[str, Any], inputs: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=False)
    pd.DataFrame(rows).to_csv(output_dir / "cohort_estimates.csv", index=False, encoding="utf-8")
    result = {"result_version": f"{domain.lower()}_sensitivity_v1", "status": "PASS", "rows": rows, "meta": metas,
              "anchors": anchors, "primary_parameterisation_inherited": True,
              "primary_standardisation_constants_sha256": sha256_file(PRIMARY_CONSTANTS)}
    (output_dir / "result_object.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "model_diagnostics.json").write_text(json.dumps(diagnostics, ensure_ascii=False, indent=2), encoding="utf-8")
    validation = {"status": "PASS", "domain": domain, "only_one_component_changed": True,
                  "primary_constants_reused_without_recalculation": True, "anchors": anchors}
    (output_dir / "validation.json").write_text(json.dumps(validation, ensure_ascii=False, indent=2), encoding="utf-8")
    files = [p for p in output_dir.rglob("*") if p.is_file()]
    manifest = {"release_version": f"{domain.lower()}_sensitivity_v1", "generated_utc": now_utc(), "inputs": inputs,
                "outputs": {p.relative_to(output_dir).as_posix(): sha256_file(p) for p in files}, "runtime": {"python": platform.python_version()},
                "submission_ready": False}
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def build_unweighted(domain: str, output_dir: Path) -> None:
    inputs = verify_inputs()
    panels = load_panels()
    constants = constants_by_cohort()
    analyses = ("recovery_25", "recovery_100") if domain == "Exposure" else ("common22",)
    rows: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    anchors: list[dict[str, Any]] = []
    metas: dict[str, Any] = {}
    labels = {"recovery_25": "Recovery threshold: 25%", "recovery_100": "Recovery threshold: 100% (return to pre-shock wealth)", "common22": "Common-22 frailty index"}
    for analysis in analyses:
        spec_rows: list[dict[str, Any]] = []
        for cohort in COHORTS:
            frame, anchor = prepare(panels[cohort], cohort, constants[cohort], analysis)
            _, _, estimates, _, diag = PRIMARY.fit_model(frame, cohort)
            row = primary_row(estimates, labels[analysis], anchor)
            spec_rows.append(row); diagnostics.append({"analysis_spec": analysis, **diag}); anchors.append(anchor)
        metas[analysis] = add_meta(spec_rows, labels[analysis])
        rows.extend(spec_rows)
    write_domain(domain, rows, diagnostics, anchors, metas, inputs, output_dir)


def build_weighted(output_dir: Path) -> None:
    inputs = verify_inputs()
    panels = load_panels(); constants = constants_by_cohort()
    rows: list[dict[str, Any]] = []; diagnostics: list[dict[str, Any]] = []; anchors: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="section_3_3_weighted_", dir=ROOT / "03_output/tmp") as tmp:
        tmp_path = Path(tmp)
        for cohort in COHORTS:
            frame, anchor = prepare(panels[cohort], cohort, constants[cohort], "weighted")
            landmark_weight = frame.groupby("person_id")["landmark_weight_raw"].first().astype(float)
            if not np.isfinite(landmark_weight).all() or (landmark_weight <= 0).any():
                raise RuntimeError(f"{cohort}: weighted样本含无效权重")
            scaled = landmark_weight / landmark_weight.mean()
            frame["participant_weight_scaled"] = frame["person_id"].map(scaled)
            frame["level1_weight"] = 1.0
            cols = ["person_id", "fi_primary", "recovery", "time_years", *[f"{v}_z" for v in CONTINUOUS], *CATEGORICAL, "level1_weight", "participant_weight_scaled"]
            source = tmp_path / f"{cohort.lower()}_weighted_input.csv"; target = tmp_path / f"{cohort.lower()}_weighted_result.csv"
            frame[cols].to_csv(source, index=False, encoding="utf-8")
            cmd = ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(ROOT / "00_admin/run_r.ps1"),
                   str(Path(__file__).with_name("run_weighted_sensitivity.R")), str(source), str(target)]
            subprocess.run(cmd, cwd=ROOT, check=True)
            result = pd.read_csv(target, float_precision="round_trip").iloc[0].to_dict()
            if result.get("status") != "PASS" or abs(float(result["scaled_weight_mean_unique"]) - 1.0) > 1e-12:
                raise RuntimeError(f"{cohort}: WeMix输出未通过")
            rows.append({"analysis_spec": "Landmark-weighted analysis", "cohort": cohort, "is_overall": False,
                         "participants": anchor["participants"], "fi_observations": anchor["fi_observations"],
                         "estimate": result["estimate"], "se": result["se"], "ci_lower": result["ci_lower"], "ci_upper": result["ci_upper"]})
            diagnostics.append({"cohort": cohort, **result}); anchors.append(anchor)
    write_domain("Weighting", rows, diagnostics, anchors, {}, inputs, output_dir)


def build_mi_participant_frame(panel: pd.DataFrame, cohort: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """构建一人一行的MI输入和不变的longitudinal outcome rows。"""
    long = panel.copy()
    long["person_id"] = long["person_id"].astype(str)
    landmark = long.loc[long["fi_time_role"].eq("landmark")].copy()
    later = long.loc[long["fi_time_role"].eq("later")].copy().sort_values(["person_id", "time_years"])
    if landmark.groupby("person_id").size().ne(1).any() or later.groupby("person_id").size().lt(1).any():
        raise RuntimeError(f"{cohort}: MI pool landmark/later支持不合法")
    if long[["fi_primary", "time_years", "recovery_status", "recovery_fraction_raw"]].isna().any().any():
        raise RuntimeError(f"{cohort}: MI不得插补的字段存在缺失")
    participant = landmark[["person_id", "index_shock_wave", "recovery_status", *CONTINUOUS, *CATEGORICAL]].copy()
    participant["recovery"] = participant["recovery_status"].eq("substantial_recovery").astype(int)
    participant["landmark_fi_primary"] = landmark["fi_primary"].astype(float).to_numpy()
    summaries = []
    for person_id, block in later.groupby("person_id", sort=False):
        x = block["time_years"].astype(float).to_numpy(); y = block["fi_primary"].astype(float).to_numpy()
        estimable = len(block) >= 2 and np.ptp(x) > 0
        slope = float(np.polyfit(x, y, 1)[0]) if estimable else 0.0
        summaries.append({"person_id": person_id, "later_fi_count": len(block), "maximum_followup_years": float(x.max()),
                          "later_fi_mean": float(y.mean()), "first_later_fi": float(y[0]), "last_later_fi": float(y[-1]),
                          "later_fi_slope": slope, "later_fi_slope_estimable": int(estimable)})
    participant = participant.merge(pd.DataFrame(summaries), on="person_id", validate="one_to_one")
    transformed = np.arcsinh(participant["wealth_loss_fraction"].astype(float).to_numpy())
    participant["wealth_loss_asinh_z"] = (transformed - transformed.mean()) / transformed.std(ddof=1)
    if cohort == "HRS":
        fi_freeze = json.loads((ROOT / "00_admin/panel_manifests/fi_panel/fi_panel_v2_freeze.json").read_text(encoding="utf-8"))
        fi = pd.read_parquet(ROOT / fi_freeze["cohorts"][cohort]["parquet_path"], columns=["person_id", "wave", "fi_common22"])
        fi["person_id"] = fi["person_id"].astype(str)
        shock = participant[["person_id", "index_shock_wave"]].merge(fi, left_on=["person_id", "index_shock_wave"], right_on=["person_id", "wave"], how="left", validate="one_to_one")
        raw = pd.to_numeric(shock["fi_common22"], errors="coerce").astype(float)
        participant["fi_common22_shock_missing"] = raw.isna().astype(int).to_numpy()
        participant["fi_common22_shock_filled"] = raw.fillna(raw.median()).to_numpy()
    participant = participant.drop(columns=["recovery_status"])
    participant["current_smoking"] = participant["current_smoking"].astype("string").map({"True": "True", "False": "False", "true": "True", "false": "False"})
    participant["wealth_quintile"] = participant["wealth_quintile"].astype("string")
    return participant, long


def fit_mi_once(frame: pd.DataFrame, cohort: str, start_params: Any | None = None) -> tuple[Any, float, float, dict[str, Any]]:
    y, x = patsy.dmatrices(PRIMARY.FORMULA, frame, return_type="dataframe", NA_action="raise")
    if x.shape[1] != 42 or np.linalg.matrix_rank(x.to_numpy()) != 42:
        raise RuntimeError("MI completed dataset设计矩阵不是42/42满秩")
    model = sm.MixedLM(endog=y.iloc[:, 0], exog=x, groups=frame["person_id"].to_numpy(), exog_re=np.ones((len(frame), 1)))
    fallback_used = False
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        if start_params is None:
            # 每个cohort首个插补严格复用primary的BFGS起点加正式L-BFGS路径。
            result, _, _, _, _ = PRIMARY.fit_model(frame, cohort)
        else:
            try:
                result = model.fit(start_params=start_params, reml=False, method="lbfgs", maxiter=1000, disp=False)
                if not result.converged:
                    raise RuntimeError("warm-start L-BFGS未收敛")
            except Exception:
                # 任一warm start异常时回到完整primary优化器复核，不删变量或降级模型。
                fallback_used = True
                result, _, _, _, _ = PRIMARY.fit_model(frame, cohort)
    if not result.converged:
        raise RuntimeError("MI completed dataset MixedLM未收敛")
    term = "recovery:time_years"
    estimate = float(result.fe_params[term])
    variance = float(result.cov_params().loc[term, term])
    if not np.isfinite(estimate) or not np.isfinite(variance) or variance <= 0:
        raise RuntimeError("MI completed dataset估计或方差无效")
    random_variance = float(np.asarray(result.cov_re)[0, 0])
    residual_variance = float(result.scale)
    if random_variance <= 0 or residual_variance <= 0:
        raise RuntimeError("MI completed dataset方差分量病态")
    return result, estimate, variance, {"random_intercept_variance": random_variance, "residual_variance": residual_variance,
                                        "variance_ratio": random_variance / residual_variance, "fallback_full_optimizer": fallback_used,
                                        "warning_count": len(captured)}


def build_mi(output_dir: Path) -> None:
    inputs = verify_inputs(); panels = load_panels(); constants = constants_by_cohort()
    contract_path = CONTRACT_DIR / "mi_sensitivity_v1_contract.json"
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    if contract.get("status") != "LOCKED" or contract["wealth_loss_auxiliary_transformation"]["function"] != "asinh":
        raise RuntimeError("MI正式合同尚未锁定")
    all_rows: list[dict[str, Any]] = []; imputation_rows: list[dict[str, Any]] = []; diagnostics: list[dict[str, Any]] = []; anchors = []
    tmp_parent = ROOT / "03_output/tmp"; tmp_parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="section_3_3_mi_", dir=tmp_parent) as tmp:
        tmp_path = Path(tmp)
        for cohort in COHORTS:
            participant, long = build_mi_participant_frame(panels[cohort], cohort)
            source = tmp_path / f"{cohort.lower()}_mi_input.csv"; completed_path = tmp_path / f"{cohort.lower()}_mi_completed.csv"
            participant.to_csv(source, index=False, encoding="utf-8")
            seed = int(contract["cohort_seeds"][cohort])
            cmd = ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(ROOT / "00_admin/run_r.ps1"),
                   str(Path(__file__).with_name("run_mi_imputation.R")), str(source), str(completed_path), cohort, str(seed)]
            subprocess.run(cmd, cwd=ROOT, check=True)
            completed = pd.read_csv(completed_path, dtype={"person_id": str})
            if sorted(completed[".imp"].unique().tolist()) != list(range(1, 51)):
                raise RuntimeError(f"{cohort}: 未生成50个completed datasets")
            original = participant.set_index("person_id")
            observed_masks = {v: original[v].notna() for v in ("fi_t", "education", "marital_status", "current_smoking", "labour_force_status")}
            start = None
            for imp, block in completed.groupby(".imp", sort=True):
                block = block.set_index("person_id")
                for variable, mask in observed_masks.items():
                    observed_ids = mask.index[mask]
                    if variable == "fi_t":
                        unchanged = np.allclose(block.loc[observed_ids, variable].astype(float), original.loc[observed_ids, variable].astype(float), rtol=0.0, atol=1e-14)
                    else:
                        unchanged = block.loc[observed_ids, variable].astype(str).equals(original.loc[observed_ids, variable].astype(str))
                    if not unchanged:
                        raise RuntimeError(f"{cohort}/imp{imp}: observed {variable}被改变")
                model_frame = long.copy()
                for variable in ("fi_t", "education", "marital_status", "current_smoking", "labour_force_status"):
                    model_frame[variable] = model_frame["person_id"].map(block[variable])
                model_frame["recovery"] = model_frame["recovery_status"].eq("substantial_recovery").astype(int)
                for variable, (mean, sd) in constants[cohort].items():
                    model_frame[f"{variable}_z"] = (pd.to_numeric(model_frame[variable]).astype(float) - mean) / sd
                for variable in CATEGORICAL:
                    model_frame[variable] = model_frame[variable].astype(str)
                result, estimate, variance, fit_diag = fit_mi_once(model_frame, cohort, start)
                start = result.params
                imputation_rows.append({"cohort": cohort, "imputation": int(imp), "estimate": estimate, "within_variance": variance, "converged": True, **fit_diag})
            block_rows = [r for r in imputation_rows if r["cohort"] == cohort]
            q = np.array([r["estimate"] for r in block_rows]); u = np.array([r["within_variance"] for r in block_rows])
            m = len(q); qbar = float(q.mean()); ubar = float(u.mean()); b = float(q.var(ddof=1)); total = ubar + (1 + 1 / m) * b
            if b > 0:
                df = float((m - 1) * (1 + ubar / ((1 + 1 / m) * b)) ** 2)
            else:
                df = math.inf
            critical = float(student_t.ppf(0.975, df)) if np.isfinite(df) else 1.959963984540054
            se = math.sqrt(total)
            all_rows.append({"analysis_spec": "Multiple-imputation analysis", "cohort": cohort, "is_overall": False,
                             "participants": int(participant["person_id"].nunique()), "fi_observations": int(len(long)),
                             "estimate": qbar, "se": se, "ci_lower": qbar - critical * se, "ci_upper": qbar + critical * se,
                             "rubin_df": df, "within_variance": ubar, "between_variance": b, "fraction_missing_information": float(((1 + 1/m) * b) / total)})
            diag = pd.read_csv(str(completed_path) + ".diagnostics.csv").iloc[0].to_dict()
            diagnostics.append({"cohort": cohort, **diag, "all_50_models_converged": True})
            anchors.append({"cohort": cohort, "participants": int(len(participant)), "fi_observations": int(len(long)),
                            "missing_adjustment_participants": int(participant[["fi_t", "education", "marital_status", "current_smoking", "labour_force_status"]].isna().any(axis=1).sum())})
    meta_input = pd.DataFrame(all_rows); meta_temp = meta_input.copy(); meta_temp["is_primary_estimand"] = True
    meta = PRIMARY.meta_analysis(meta_temp)
    all_rows.append({"analysis_spec": "Multiple-imputation analysis", "cohort": "Overall (random-effects)", "is_overall": True,
                     "participants": "3 cohorts", "fi_observations": None, "estimate": meta["estimate"], "se": meta["se"],
                     "ci_lower": meta["ci_lower"], "ci_upper": meta["ci_upper"]})
    output_dir.mkdir(parents=True, exist_ok=False)
    pd.DataFrame(all_rows).to_csv(output_dir / "cohort_estimates.csv", index=False, encoding="utf-8")
    pd.DataFrame(imputation_rows).to_csv(output_dir / "imputation_estimates.csv", index=False, encoding="utf-8")
    (output_dir / "model_diagnostics.json").write_text(json.dumps(diagnostics, ensure_ascii=False, indent=2), encoding="utf-8")
    result = {"result_version": "mi_sensitivity_v1", "status": "PASS", "rows": all_rows, "meta": meta, "anchors": anchors,
              "contract_sha256": sha256_file(contract_path), "participant_level_outputs_retained": False}
    (output_dir / "result_object.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    validation = {"status": "PASS", "domain": "MI", "imputations_per_cohort": 50, "iterations": 20,
                  "all_models_converged": True, "observed_values_unchanged": True, "temporary_microdata_removed_on_exit": True,
                  "primary_constants_reused_without_recalculation": True, "anchors": anchors}
    (output_dir / "validation.json").write_text(json.dumps(validation, ensure_ascii=False, indent=2), encoding="utf-8")
    files = [p for p in output_dir.rglob("*") if p.is_file()]
    manifest = {"release_version": "mi_sensitivity_v1", "generated_utc": now_utc(), "inputs": {**inputs, rel(contract_path): sha256_file(contract_path)},
                "outputs": {p.relative_to(output_dir).as_posix(): sha256_file(p) for p in files},
                "runtime": {"python": platform.python_version(), "mice": "3.19.0"}, "submission_ready": False}
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def freeze_domain(output_dir: Path) -> None:
    check_domain(output_dir)
    freeze_path = output_dir / "freeze.json"
    if freeze_path.exists():
        raise RuntimeError("domain已冻结，拒绝覆盖")
    freeze_path.write_text(json.dumps({"freeze_status": "FROZEN", "frozen_utc": now_utc(),
                                       "manifest_sha256": sha256_file(output_dir / "manifest.json")}, ensure_ascii=False, indent=2), encoding="utf-8")


def check_domain(output_dir: Path) -> None:
    verify_inputs()
    for name in ("cohort_estimates.csv", "result_object.json", "model_diagnostics.json", "validation.json", "manifest.json"):
        if not (output_dir / name).is_file():
            raise RuntimeError(f"缺少domain资产：{output_dir / name}")
    validation = json.loads((output_dir / "validation.json").read_text(encoding="utf-8"))
    if validation.get("status") != "PASS" or validation.get("primary_constants_reused_without_recalculation") is not True:
        raise RuntimeError("domain validation失败")
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    for path_text, digest in manifest["outputs"].items():
        # 早期候选曾把staging前缀写入manifest；文件名仍是稳定且唯一的domain内路径。
        candidate = Path(path_text)
        path = output_dir / (candidate.name if "03_output/tmp/section_3_3_" in candidate.as_posix() else candidate)
        if not path.is_file() or sha256_file(path) != digest:
            raise RuntimeError(f"domain输出漂移：{path_text}")
    rows = pd.read_csv(output_dir / "cohort_estimates.csv", float_precision="round_trip")
    numeric = rows[["estimate", "se", "ci_lower", "ci_upper"]].to_numpy(float)
    if not np.isfinite(numeric).all() or (rows["se"] <= 0).any() or not (rows["ci_lower"] <= rows["estimate"]).all() or not (rows["estimate"] <= rows["ci_upper"]).all():
        raise RuntimeError("domain estimate/SE/CI无效")
    domain = validation.get("domain")
    expected_rows = {"Exposure": 8, "Outcome": 4, "Weighting": 3, "MI": 4}
    if len(rows) != expected_rows.get(domain):
        raise RuntimeError(f"{domain}: 结果行数不符合合同")
    overall_count = int(rows["is_overall"].astype(str).str.lower().eq("true").sum())
    if overall_count != ({"Exposure": 2, "Outcome": 1, "Weighting": 0, "MI": 1}[domain]):
        raise RuntimeError(f"{domain}: Overall行规则不符合合同")
    diagnostics = json.loads((output_dir / "model_diagnostics.json").read_text(encoding="utf-8"))
    if domain in {"Exposure", "Outcome"}:
        if not all(d.get("design_matrix_columns") == d.get("design_matrix_rank") == 42 and d.get("lbfgs_converged") and d.get("bfgs_converged") and not d.get("hessian_singular") for d in diagnostics):
            raise RuntimeError(f"{domain}: MixedLM诊断未通过")
    elif domain == "Weighting":
        if not all(d.get("status") == "PASS" and d.get("fixed_effect_count") == 42 and d.get("covariance") == "participant_level_CR0" and abs(float(d.get("scaled_weight_mean_unique")) - 1.0) <= 1e-12 for d in diagnostics):
            raise RuntimeError("Weighting: WeMix/CR0诊断未通过")
    else:
        imps = pd.read_csv(output_dir / "imputation_estimates.csv")
        if len(imps) != 150 or not imps["converged"].astype(bool).all() or set(imps.groupby("cohort").size()) != {50}:
            raise RuntimeError("MI: 150个completed-dataset模型未全部收敛")
        if not all(d.get("all_50_models_converged") is True for d in diagnostics):
            raise RuntimeError("MI: 诊断未通过")
    freeze_path = output_dir / "freeze.json"
    if freeze_path.is_file():
        freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
        if freeze.get("freeze_status") != "FROZEN" or freeze.get("manifest_sha256") != sha256_file(output_dir / "manifest.json"):
            raise RuntimeError("domain freeze与manifest不一致")


def preflight(output_dir: Path) -> None:
    verify_inputs()
    output_dir.mkdir(parents=True, exist_ok=False)
    candidate = json.loads((CONTRACT_DIR / "mi_sensitivity_v1_preflight_candidate.json").read_text(encoding="utf-8"))
    report = {"status": "BLOCKED_AUTHOR_DECISION_REQUIRED", "real_mi_executed": False,
              "blocking_field": "wealth_loss_auxiliary_transformation", "candidate_contract": candidate,
              "methods_sync_path": rel(ROOT / "07_manuscript/Table/methods_sync_3_3.md")}
    (output_dir / "mi_preflight_validation.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "README.md").write_text("# MI sensitivity preflight\n\n真实MI尚未运行。必须先由作者批准并冻结wealth-loss auxiliary transformation及完整predictor matrix。\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("preflight", "build", "check", "freeze"))
    parser.add_argument("--analysis", choices=("Exposure", "Outcome", "Weighting", "MI"), required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(); output_dir = Path(args.output_dir).resolve()
    if args.command == "preflight":
        if args.analysis != "MI": raise RuntimeError("preflight仅用于MI")
        preflight(output_dir)
    elif args.command == "build":
        if args.analysis in {"Exposure", "Outcome"}: build_unweighted(args.analysis, output_dir)
        elif args.analysis == "Weighting": build_weighted(output_dir)
        else: build_mi(output_dir)
    elif args.command == "check": check_domain(output_dir)
    else: freeze_domain(output_dir)


if __name__ == "__main__":
    main()
