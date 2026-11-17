"""冻结来源、历史例外、Primary行集与设计核验；只读既有资产。"""
from __future__ import annotations

import ast
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import patsy

from common import ROOT, PRIMARY, PANEL_FREEZE, COHORTS, read_json, sha, rel, require, repo_path
from specification import FORMULA, CONTINUOUS, CATEGORIES, COHORT_ANCHORS, KNOTS
from time_basis import rcs_basis

CLOSEOUT = "06_project_progress/foundation_audit_addendum_v1/closeout_manifest.json"
SOURCE_LINEAGE = "06_project_progress/foundation_audit_addendum_v1/foundation_audit_addendum_v1_freeze.json"
HISTORICAL_EXCEPTIONS = {
    "07_manuscript/manuscript_v1.docx": ("historical_manuscript", ["07_manuscript/supplementary_materials/manuscript_v1.docx"]),
    "07_manuscript/manuscript_v1_1.docx": ("historical_manuscript", ["07_manuscript/supplementary_materials/manuscript_v1_1.docx"]),
    "07_manuscript/supplementary_materials/figures/build_figure_s1.py": ("sample_flow_presentation_code", []),
    "07_manuscript/supplementary_materials/figures/figure_s1_author_review.pdf": ("sample_flow_presentation", []),
    "07_manuscript/supplementary_materials/figures/figure_s1_caption_and_review.md": ("sample_flow_caption", []),
    "07_manuscript/supplementary_materials/main.tex": ("supplement_compilation", []),
    "07_manuscript/supplementary_materials/output/Supplementary_Materials.pdf": ("compiled_supplement", [
        "07_manuscript/supplementary_materials/output/Supplementary_Material.pdf", "07_manuscript/Submit/Supplementary_Material.pdf"]),
    "07_manuscript/supplementary_materials/README.md": ("supplement_documentation", []),
    "07_manuscript/supplementary_materials/supplementary_methods.tex": ("scientific_description_not_execution_authority", []),
    "07_manuscript/supplementary_materials/tables/table_s2.tex": ("wealth_definition_presentation", []),
    "07_manuscript/supplementary_materials/tables/table_s3.tex": ("FI_definition_presentation", []),
    "07_manuscript/supplementary_materials/tables/table_s5.tex": ("followup_presentation", []),
}


def classify_history(expected, root, closeout):
    """仅明确列出的历史呈现路径可作为例外；未知异常默认阻断。"""
    root = Path(root)
    rows = []
    for name, old_sha in expected.items():
        path = repo_path(name, root)
        current_sha = sha(path) if path.is_file() else None
        if current_sha == old_sha:
            continue
        known = HISTORICAL_EXCEPTIONS.get(name)
        role, alternatives = known if known else ("unresolved_statistical_source", [])
        candidates = [name] if current_sha else alternatives
        evidence = []
        for candidate in candidates:
            p = repo_path(candidate, root)
            if p.is_file():
                h = sha(p)
                evidence.append({"path": candidate, "sha256": h, "matches_old_sha": h == old_sha,
                                 "matches_later_closeout": closeout.get(candidate) == h})
        rows.append({"old_path": name, "old_sha256": old_sha, "current_sha256_at_old_path": current_sha,
                     "current_candidates": evidence, "role": role,
                     "classification": "NONBLOCKING_HISTORICAL_EXCEPTION" if known else "BLOCKING_STATISTICAL_SOURCE_EXCEPTION",
                     "affects_round1b_model_input": not bool(known), "evidence_path": CLOSEOUT,
                     "reason": "历史稿件或呈现资产；不作为Round1B统计输入；不推断仅格式变化" if known else "未解释的来源异常，禁止自动豁免",
                     "later_evidence_verified": bool(evidence) and all(x["matches_later_closeout"] for x in evidence)})
    return rows


def read_assignment(text, name):
    for node in ast.parse(text).body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
            return ast.literal_eval(node.value)
    raise ValueError("源码缺少常量：" + name)


def verify_sources():
    """历史manifest保护异常和实际统计依赖分别核验，不重写冻结。"""
    freeze = read_json(PRIMARY / "section_3_2_freeze.json")
    manifest = read_json(PRIMARY / "section_3_2_manifest.json")
    require(freeze["freeze_status"] == "FROZEN" and freeze["freeze_version"] == "section_3_2_v3", "Primary发布身份不符")
    require(sha(PRIMARY / "section_3_2_manifest.json") == freeze["manifest_sha256"], "Primary manifest哈希异常")
    closeout = read_json(ROOT / CLOSEOUT)["protected_hashes_after"]
    expected = {**manifest["outputs"], **manifest["protected_asset_hashes"]}
    exceptions = classify_history(expected, ROOT, closeout)
    blockers = [r for r in exceptions if r["classification"].startswith("BLOCKING")]
    inputs = {}

    def pin(path, expected_sha=None):
        path = Path(path)
        require(path.is_file(), "必需统计来源缺失：" + str(path))
        h = sha(path)
        if expected_sha:
            require(h == expected_sha, "统计来源哈希异常：" + rel(path))
        inputs[rel(path)] = h

    for name, h in manifest["outputs"].items():
        pin(ROOT / name, h)
    contract_record = manifest["inputs"]["contract"]
    pin(ROOT / contract_record["path"], contract_record["sha256"])
    panel_record = manifest["inputs"]["analysis_panel_freeze"]["analysis_panel_freeze"]
    pin(ROOT / panel_record["path"], panel_record["sha256"])
    pin(PRIMARY / "section_3_2_freeze.json")
    pin(PRIMARY / "section_3_2_manifest.json")
    pin(ROOT / CLOSEOUT)
    pin(ROOT / SOURCE_LINEAGE)
    lineage = read_json(ROOT / SOURCE_LINEAGE)
    for name, h in lineage["historical_model_assets_sha256"].items():
        pin(ROOT / name, h)
    s = lineage["source_lineage"]
    old_path, current_path = ROOT / s["historical_source_backup"], ROOT / s["current_repaired_source_path"]
    pin(old_path, s["historical_source_sha256"])
    pin(current_path, s["current_repaired_source_sha256"])
    old_text, current_text = old_path.read_text(encoding="utf-8"), current_path.read_text(encoding="utf-8")
    require(read_assignment(old_text, "FORMULA") == FORMULA == read_assignment(current_text, "FORMULA"), "Primary固定效应公式冲突")
    functions = lambda t: {n.name: ast.dump(n, include_attributes=False) for n in ast.parse(t).body if isinstance(n, ast.FunctionDef)}
    old_funcs, current_funcs = functions(old_text), functions(current_text)
    require(set(old_funcs) == set(current_funcs), "历史及当前Primary函数集合变化")
    different = [name for name in old_funcs if old_funcs[name] != current_funcs[name]]
    require(different == ["meta_analysis"], "历史与KH源码的队列模型逻辑有差异")
    primary_contract = read_json(ROOT / contract_record["path"])
    require(primary_contract["status"] == "LOCKED" and primary_contract["model"]["reml"] is False,
            "Primary合同未锁定或不是ML")
    require(primary_contract["sample"] == "primary_complete_case_eligible" and primary_contract["outcome"] == "fi_primary", "Primary样本或结局冲突")
    panel = read_json(PANEL_FREEZE)
    require(panel["freeze_status"] == "FROZEN", "Analysis Panel未冻结")
    for cohort, rec in panel["cohorts"].items():
        for pkey, hkey in [("parquet_path", "parquet_sha256"), ("manifest_path", "manifest_sha256")]:
            pin(ROOT / rec[pkey], rec[hkey])
    design_path = ROOT / panel["design_register"]["path"]
    register = pd.read_csv(design_path, encoding="utf-8", dtype=str, keep_default_na=False)
    for key, value in panel["design_register"]["required_locked_decisions"].items():
        row = register.loc[register.decision_id.eq(key)]
        require(len(row) == 1 and row.iloc[0]["status"] == "locked" and row.iloc[0]["selected_value"] == value,
                "冻结所需design决策冲突：" + key)
    pin(design_path)
    pin(ROOT / "uv.lock")
    pin(ROOT / "pyproject.toml")
    import statsmodels.regression.mixed_linear_model as mixed_backend
    pin(Path(mixed_backend.__file__))
    # 稿件现版本SHA仅为历史例外证据，不混入统计输入的执行阻断集合。
    historical_evidence = {candidate["path"]: candidate["sha256"]
                           for row in exceptions for candidate in row["current_candidates"]}
    require(not blockers, "存在阻断性统计来源异常：" + str([r["old_path"] for r in blockers]))
    return {"status": "PASS", "inputs": inputs, "historical_items_checked": len(expected),
            "primary_outputs_checked": len(manifest["outputs"]), "historical_exceptions": exceptions,
            "historical_evidence_hashes": historical_evidence,
            "statistical_blockers": blockers, "different_source_functions": different,
            "design_register_historical_sha": panel["design_register"]["sha256"],
            "design_register_current_sha": sha(design_path), "required_locked_design_decisions": "PASS"}


def frozen_primary(cohort, columns):
    obj = read_json(PRIMARY / "Primary Models/primary_lmm_result_object.json")
    rows = [r for r in obj["fixed_effects"] if r["cohort"] == cohort]
    require([r["term"] for r in rows] == list(columns), "冻结Primary参数term顺序漂移")
    covariance = pd.DataFrame([r for r in obj["fixed_effect_covariance"] if r["cohort"] == cohort])
    covariance = covariance.pivot(index="term_i", columns="term_j", values="covariance").loc[columns, columns].to_numpy(float)
    diagnostics = next(r for r in obj["diagnostics"] if r["cohort"] == cohort)
    beta = np.array([r["estimate"] for r in rows])
    csv = pd.read_csv(PRIMARY / "Primary Models/primary_lmm_fixed_effects.csv", encoding="utf-8", float_precision="round_trip")
    csv = csv.loc[csv.cohort.eq(cohort)]
    require(csv.term.tolist() == list(columns), "冻结Primary CSV标签不符")
    np.testing.assert_array_equal(beta, csv.estimate.to_numpy(float))
    constants = pd.read_csv(PRIMARY / "Primary Models/primary_lmm_standardization_constants.csv", encoding="utf-8", float_precision="round_trip")
    constants = constants.loc[constants.cohort.eq(cohort)].set_index("variable")
    for row in obj["standardization_constants"]:
        if row["cohort"] == cohort:
            np.testing.assert_array_equal([row["mean"], row["sample_sd"]],
                                          constants.loc[row["variable"], ["mean", "sample_sd"]].to_numpy(float))
    return beta, covariance, diagnostics


def load_cohort(cohort):
    require(cohort in COHORTS, "未知队列")
    rec = read_json(PANEL_FREEZE)["cohorts"][cohort]
    require(sha(ROOT / rec["parquet_path"]) == rec["parquet_sha256"], "Panel物理哈希漂移")
    all_rows = pd.read_parquet(ROOT / rec["parquet_path"])
    require(all_rows.shape[1] == 29, "Panel不是29列冻结结构")
    require(all_rows.primary_complete_case_eligible.notna().all(), "Primary资格flag缺失")
    frame = all_rows.loc[all_rows.primary_complete_case_eligible.eq(True)].copy()
    require(frame.cohort.eq(cohort).all() and not frame[["person_id", "wave"]].duplicated().any(), "队列或person-wave键错误")
    frame["person_id"] = frame.person_id.astype(str)
    lm = frame.loc[frame.fi_time_role.eq("landmark")].copy()
    later = frame.loc[frame.fi_time_role.eq("later")]
    roles = frame.groupby(["person_id", "fi_time_role"]).size().unstack(fill_value=0)
    require(set(roles.columns) == {"landmark", "later"} and roles.landmark.eq(1).all() and roles.later.ge(1).all(), "个人landmark/later支持错误")
    anchor = {"participants": len(lm), "rows": len(frame), "later": len(later),
              "recovery": int(lm.recovery_status.eq("substantial_recovery").sum()),
              "reference": int(lm.recovery_status.eq("persistent_or_limited_recovery").sum())}
    require(anchor == COHORT_ANCHORS[cohort], "Primary样本锚点漂移")
    require(frame.groupby("person_id").recovery_status.nunique().eq(1).all(), "个人恢复状态不恒定")
    require(lm.time_years.eq(0).all() and later.time_years.gt(0).all(), "时间原点或later时间错误")
    require(frame.groupby("person_id").time_years.nunique().eq(frame.groupby("person_id").size()).all(), "个人实际时间重复")
    month_index = frame.interview_year.astype(float)*12 + frame.interview_month.astype(float)
    origins = pd.Series(month_index.loc[lm.index].to_numpy(), index=lm.person_id)
    elapsed = month_index - frame.person_id.map(origins)
    require(np.array_equal(elapsed.to_numpy(), frame.elapsed_months.to_numpy()), "实际访谈年月与elapsed_months冲突")
    require(np.array_equal(elapsed.to_numpy()/12, frame.time_years.to_numpy()), "实际时间尺度冲突")
    require(np.isfinite(frame[["fi_primary", "time_years", *CONTINUOUS]].to_numpy(float)).all(), "Primary模型数值缺失")
    require(frame.fi_primary.between(0, 1).all(), "FI超出冻结合法范围")
    constants = pd.read_csv(PRIMARY / "Primary Models/primary_lmm_standardization_constants.csv", encoding="utf-8", float_precision="round_trip")
    constants = constants.loc[constants.cohort.eq(cohort)].set_index("variable")
    require(set(constants.index) == set(CONTINUOUS), "冻结标准化项不完整")
    for variable in CONTINUOUS:
        row = constants.loc[variable]
        require(row.sample_sd > 0 and int(row.n_unique_participants) == len(lm), "冻结标准化分母错误")
        require(frame.groupby("person_id")[variable].nunique().eq(1).all(), "调整变量个人内不固定")
        np.testing.assert_allclose([lm[variable].mean(), lm[variable].std(ddof=1)], [row["mean"], row.sample_sd], rtol=0, atol=1e-10)
        frame[variable + "_z"] = (frame[variable].astype(float)-row["mean"])/row.sample_sd
    for variable, levels in CATEGORIES.items():
        require(frame[variable].notna().all(), "分类调整变量缺失")
        frame[variable] = frame[variable].astype(str)
        require(set(frame[variable]).issubset(levels) and levels[0] in set(frame[variable]), "分类水平或参照错误")
        require(frame.groupby("person_id")[variable].nunique().eq(1).all(), "分类调整变量个人内不恒定")
    frame["recovery"] = frame.recovery_status.eq("substantial_recovery").astype(int)
    lm = frame.loc[frame.fi_time_role.eq("landmark")].copy()
    knots = [0., *np.quantile(later.time_years.to_numpy(float), [.5, .9], method="linear")]
    require([float(v).hex() for v in knots] == [v.hex() for v in KNOTS[cohort]], "RCS完整精度结点漂移")
    x = patsy.dmatrix(FORMULA.split("~", 1)[1], frame, NA_action="raise", return_type="dataframe")
    require(x.shape[1] == 42 and np.linalg.matrix_rank(x) == 42, "Primary矩阵不是42/42")
    beta, covariance, diag = frozen_primary(cohort, list(x.columns))
    identity = frame[["person_id", "wave"]].to_json(orient="values", force_ascii=False)
    anchor.update({"cohort": cohort, "ordered_person_wave_sha256": hashlib.sha256(identity.encode("utf-8")).hexdigest(),
                   "parquet_sha256": rec["parquet_sha256"]})
    return frame, lm, x, beta, covariance, diag, anchor


def model_design(frame, x_primary, cohort, spec):
    require(spec in ("RS01", "NL01"), "未知规格")
    x = x_primary.copy()
    if spec == "NL01":
        h = rcs_basis(frame.time_years.to_numpy(float), KNOTS[cohort])
        x["rcs_time"] = h
        x["recovery:rcs_time"] = frame.recovery.to_numpy(float)*h
    singular = np.linalg.svd(x.to_numpy(float), compute_uv=False)
    rank = int(np.linalg.matrix_rank(x))
    require(rank == x.shape[1] == (42 if spec == "RS01" else 44), "模型设计降秩")
    return x, {"columns": list(x.columns), "n_columns": x.shape[1], "rank": rank,
               "singular_values": singular.tolist(), "condition_number": float(singular[0]/singular[-1])}


def reconstruct_primary(frame, x, beta, diagnostic):
    residual = frame.fi_primary.to_numpy(float)-x.to_numpy(float) @ beta
    group = pd.DataFrame({"id": frame.person_id.to_numpy(), "residual": residual}).groupby("id").residual.agg(["size", "sum"])
    tau, sigma = diagnostic["random_intercept_variance"], diagnostic["residual_variance"]
    bhat = tau*group["sum"]/(sigma+tau*group["size"])
    fitted = x.to_numpy(float) @ beta + frame.person_id.map(bhat).to_numpy(float)
    conditional = frame.fi_primary.to_numpy(float)-fitted
    actual = {"residual_mean": float(conditional.mean()), "residual_sd": float(conditional.std(ddof=1)),
              "residual_rmse": float(np.sqrt(np.mean(conditional**2))),
              "residual_time_correlation": float(np.corrcoef(conditional, frame.time_years)[0, 1]),
              "residual_fitted_correlation": float(np.corrcoef(conditional, fitted)[0, 1])}
    errors = {key: abs(value-diagnostic[key]) for key, value in actual.items()}
    require(max(errors.values()) <= 1e-10, "冻结Primary残差重构不一致；禁止重拟合或放宽容差")
    return {"status": "PASS", "summary": actual, "absolute_errors": errors,
            "tolerance": 1e-10, "fit_calls": 0, "row_level_independent_validation": False}


def time_support(frame):
    rows = []
    for width in (1, 2):
        temp = frame.assign(bin=np.where(frame.time_years.eq(0), 0, np.ceil(frame.time_years/width).astype(int)))
        for (interval, recovery), g in temp.groupby(["bin", "recovery_status"]):
            n = g.person_id.nunique()
            rows.append({"width_years": width, "bin": int(interval), "recovery_status": recovery,
                         "observations": len(g), "participants": n, "sparse_display_only": n < 30})
    counts = frame.groupby("person_id").size()
    return {"bins": rows, "participant_observation_counts": counts.value_counts().sort_index().to_dict(),
            "exactly_two_observations_percent": float(counts.eq(2).mean()*100),
            "maximum_observed_time": float(frame.time_years.max()), "support_is_not_survival_risk_set": True}
