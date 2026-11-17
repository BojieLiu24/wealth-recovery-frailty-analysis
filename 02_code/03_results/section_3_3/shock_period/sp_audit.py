"""Round 1E只读来源、时点、聚合描述及SP01设计；不实现拟合。"""
from __future__ import annotations

from contextlib import contextmanager
from io import BytesIO
from pathlib import Path
import hashlib
import json
import re
import subprocess
import sys
import xml.etree.ElementTree as ET
import zipfile

import numpy as np
import pandas as pd
import scipy.optimize as optimization
import statsmodels.api as sm

CODE = Path(__file__).resolve().parent
ROOT = CODE.parents[3]
sys.path.insert(0, str(CODE.parent / "mortality_attrition"))
from audit_sources import verify_sources as prior_sources, snapshot, verify_records
from prepare_inputs import load_cohort, reconstruct_primary
from common import sha, rel, require, clean, write_json, provenance
from specification import FORMULA

COHORTS = ("HRS", "ELSA", "KLoSA")
LEVELS = {"HRS": tuple(range(3, 15)), "ELSA": tuple(range(2, 9)), "KLoSA": tuple(range(2, 8))}
INDEX_N = {"HRS": 8204, "ELSA": 1284, "KLoSA": 3125}
EXPECTED_COLUMNS = {"HRS": 64, "ELSA": 54, "KLoSA": 52}
PINNED = {
    "00_admin/model_manifests/primary_lmm/primary_lmm_v1_contract.json": "b7271c754591d0095c4feaf5be90c932396f44e7d7ca2e0da130742b4761e416",
    "03_output/3.2 Wealth Recovery and Subsequent Frailty Progression/section_3_2_freeze.json": "110b87ab21e9085c1f54cb6744da40a846b62b1503753cda2e5580abdc0ebc9b",
    "00_admin/panel_manifests/analysis_panel/analysis_panel_v1_freeze.json": "192e2b4675be3f0b5d5979a8289d61e3377ca4fb44b3ab1c437bc58c0a3aecbb",
}
INTERPRETATION = (
    "SP01 evaluates whether the recovery-by-time association changes after adjustment for "
    "shock-wave differences in FI level and average progression. It assumes a common "
    "recovery-associated annual slope difference across shock waves and does not estimate "
    "period-specific recovery associations."
)
SOURCE_ROLES = {
    "A_INDEX_SHOCK": "窗口内首次合格冲击；不要求恢复、FI、协变量或未来回访",
    "B_PRIMARY": "冻结Primary完整案例参与者；与原模型同一有序FI行集",
}


@contextmanager
def no_models(counter):
    """在Stage 0入口硬阻断模型拟合和优化；退出时恢复第三方接口。"""
    def blocked(*args, **kwargs):
        counter["forbidden_calls"] += 1
        raise RuntimeError("ROUND1E_STAGE0_FORBIDS_MODEL_OR_OPTIMIZATION")
    classes = (sm.MixedLM, sm.OLS, sm.GLM, sm.GEE)
    fits = {cls: cls.fit for cls in classes}
    names = ("minimize", "fmin_bfgs", "fmin_l_bfgs_b", "least_squares", "root", "minimize_scalar")
    opts = {name: getattr(optimization, name) for name in names}
    try:
        for cls in classes:
            cls.fit = blocked
        for name in names:
            setattr(optimization, name, blocked)
        yield
    finally:
        for cls, fit in fits.items():
            cls.fit = fit
        for name, opt in opts.items():
            setattr(optimization, name, opt)


def verify_all_sources():
    """仅复用旧模块的只读接口，额外绑定本轮指定身份及Round 1D发布。"""
    verify_records(PINNED)
    sources = prior_sources()
    records = sources["records"]
    releases = []
    for rid in ("wealth_measurement_v1", "model_robustness_round1b_v1",
                "landmark_fi_round1c_v1", "mortality_attrition_round1d_v1"):
        base = ROOT / "03_output/3.3 Robustness of the Main Findings/releases" / rid
        fr = json.loads((base / "freeze.json").read_text(encoding="utf-8"))
        m = json.loads((base / "release_manifest.json").read_text(encoding="utf-8"))
        require(fr["release_id"] == rid and fr["status"] == "FROZEN_WITH_CAVEATS", "既有发布身份错误")
        for name, key in (("release_manifest.json", "release_manifest_sha256"),
                          ("release_validation.json", "release_validation_sha256")):
            require(sha(base / name) == fr[key], "既有发布管理文件SHA变化")
        inventory = m.get("inventory", m.get("analysis_inventory"))
        for item in inventory:
            p = (base / item.get("path", item.get("release_path"))).resolve()
            require(p.is_relative_to(base.resolve()) and sha(p) == item["sha256"], "发布包内容变化")
            records[rel(p)] = item["sha256"]
        for field in ("external_statistical_sources", "external_execution_sources",
                      "external_sources", "restricted_internal_references"):
            verify_records(m.get(field, {}))
            records.update(m.get(field, {}))
        for item in m.get("source_assets", []):
            for pathkey, hashkey in (("path", "sha256"), ("freeze_path", "freeze_sha256")):
                verify_records({item[pathkey]: item[hashkey]})
                records[item[pathkey]] = item[hashkey]
        for item in m.get("external_evidence_references", []):
            verify_records({item["path"]: item["sha256"]})
            records[item["path"]] = item["sha256"]
        for item in m.get("code_provenance", []):
            p = base / item["snapshot_path"]
            require(sha(p) == item["sha256"], "历史执行源码快照变化")
            records[rel(p)] = item["sha256"]
        for name in ("freeze.json", "release_manifest.json", "release_validation.json"):
            records[rel(base / name)] = sha(base / name)
        releases.append({"release_id": rid, "freeze_sha256": sha(base / "freeze.json"),
                         "payload_checked": len(inventory), "status": "PASS"})
    for p in CODE.glob("*"):
        if p.suffix in {".py", ".md"}:
            records[rel(p)] = sha(p)
    sources["releases"] = releases
    return sources


def read_panel(sources, kind, cohort):
    df = pd.read_parquet(sources["paths"][kind][cohort])
    df["person_id"] = df.person_id.astype(str)
    return df


def first_shocks(exposure, levels):
    """首次冲击资格只读取冲击当前及过去的冻结事实，不读取恢复或未来FI。"""
    selected = exposure.loc[exposure.major_shock.eq(True) & exposure.candidate_shock_wave.isin(levels)]
    require(not selected[["person_id", "candidate_shock_wave"]].duplicated().any(), "候选冲击键重复")
    return selected.groupby("person_id").candidate_shock_wave.min()


def join_spine(person, spine, field, prefix):
    names = ("interview_year", "interview_month", "observed_in_wave", "nominal_year")
    require(not spine[["person_id", "wave"]].duplicated().any(), "Spine连接键重复")
    right = spine[["person_id", "wave", *names]].rename(columns={n: prefix + "_" + n for n in names})
    before = person.person_id.tolist()
    result = person.merge(right, left_on=["person_id", field], right_on=["person_id", "wave"],
                          how="left", validate="one_to_one", sort=False).drop(columns="wave")
    require(before == result.person_id.tolist(), "连接改变个人顺序或资格分母")
    return result


def month_index(year, month):
    """缺失或非法年月保持未知，不用名义年份或虚构日补齐。"""
    y, m = pd.to_numeric(year, errors="coerce"), pd.to_numeric(month, errors="coerce")
    valid = y.between(1900, 2100) & m.between(1, 12) & y.mod(1).eq(0) & m.mod(1).eq(0)
    return (y.astype(float) * 12 + m.astype(float)).where(valid)


def crisis_indicators(person):
    a = person.shock_interview_year.between(2008, 2009).astype("boolean").where(person.shock_interview_year.notna(), pd.NA)
    start = month_index(person.pre_interview_year, person.pre_interview_month)
    end = month_index(person.shock_interview_year, person.shock_interview_month)
    known = start.notna() & end.notna()
    require((end.loc[known] > start.loc[known]).all(), "冲击区间非正，停止来源核查")
    overlap = ((end >= 2008 * 12 + 1) & (start < 2009 * 12 + 12)).astype("boolean").where(known, pd.NA)
    return {"shock_interview_2008_2009": a, "interval_overlap_2008_2009": overlap}


def build_design(frame, primary_x, cohort):
    """显式按数值顺序加入波次主效应与时间项，不加入恢复时期交互。"""
    levels = LEVELS[cohort]
    require(frame.index.equals(primary_x.index), "原设计与FI行顺序不同")
    require(primary_x.shape[1] == 42 and np.linalg.matrix_rank(primary_x) == 42, "原Primary基底不是42/42")
    require(frame.index_shock_wave.notna().all() and set(frame.index_shock_wave).issubset(levels), "冲击波次缺失或越界")
    require(set(frame.index_shock_wave) == set(levels), "预定类别为空，不能自动合并")
    require(frame.groupby("person_id").index_shock_wave.nunique().eq(1).all(), "个人冲击波次不恒定")
    x = primary_x.copy()
    for wave in levels[1:]:
        x[f"shock_wave[W{wave}]"] = frame.index_shock_wave.eq(wave).astype(float)
    for wave in levels[1:]:
        x[f"shock_wave[W{wave}]:time_years"] = frame.index_shock_wave.eq(wave).astype(float) * frame.time_years
    require(np.array_equal(x.iloc[:, :42].to_numpy(), primary_x.to_numpy()), "改变原Primary设计")
    a = x.to_numpy(float)
    require(np.isfinite(a).all(), "非有限设计矩阵")
    singular = np.linalg.svd(a, compute_uv=False)
    rank = int(np.linalg.matrix_rank(a))
    require(rank == x.shape[1] == EXPECTED_COLUMNS[cohort], "新增设计秩亏，禁止自动删列")
    scaled = np.linalg.svd(a / np.linalg.norm(a, axis=0), compute_uv=False)
    original = primary_x.to_numpy(float)
    orig_sv = np.linalg.svd(original / np.linalg.norm(original, axis=0), compute_uv=False)
    require("recovery:time_years" in x, "核心参数丢失")
    return x, {"spec_id": "SP01", "levels": list(levels), "reference": levels[0],
        "primary_columns": primary_x.columns.tolist(), "columns": x.columns.tolist(),
        "added_columns": x.columns[42:].tolist(), "rank": rank, "n_columns": x.shape[1],
        "singular_values": singular.tolist(), "condition_number": float(singular[0] / singular[-1]),
        "column_norm_scaled_singular_values": scaled.tolist(),
        "column_norm_scaled_condition": float(scaled[0] / scaled[-1]),
        "primary_column_norm_scaled_condition": float(orig_sv[0] / orig_sv[-1]),
        "core_parameter_estimable": True, "estimability_basis": "full_column_rank_no_null_space",
        "original_columns_unchanged": True, "recovery_period_interactions": False,
        "interpretation": INTERPRETATION, "condition_scaling_is_diagnostic_only": True,
        "inference_or_convergence_verified": False,
        "decision": "PERIOD_MODEL_FEASIBLE", "scientific_contract_status": "PENDING_STAGE1_AUTHOR_APPROVAL"}


def cohort_data(sources, cohort):
    ledger = read_panel(sources, "ledger", cohort)
    exposure = read_panel(sources, "exposure", cohort)
    spine = read_panel(sources, "spine", cohort)
    require(len(ledger) == INDEX_N[cohort] and not ledger.person_id.duplicated().any(), "首次冲击人数或键漂移")
    first = first_shocks(exposure, LEVELS[cohort])
    idx = exposure.loc[exposure.is_index_shock.eq(True)].set_index("person_id")
    require(set(ledger.person_id) == set(first.index) == set(idx.index), "首次冲击身份不一致")
    require(ledger.index_shock_wave.eq(ledger.person_id.map(first)).all(), "首次冲击非窗口内最早")
    for field, other in (("index_shock_wave", "candidate_shock_wave"), ("tminus1_wave", "tminus1_wave"),
                         ("landmark_wave", "tplus1_wave")):
        require(ledger[field].eq(ledger.person_id.map(idx[other])).all(), "Exposure与Ledger时点冲突")
    person = ledger.copy()
    for field, prefix in (("tminus1_wave", "pre"), ("index_shock_wave", "shock"), ("landmark_wave", "lm")):
        person = join_spine(person, spine, field, prefix)
    require(person.pre_observed_in_wave.eq(True).all() and person.shock_observed_in_wave.eq(True).all(), "冲击访谈未确认")
    require(person.index_shock_wave.eq(person.tminus1_wave + 1).all() & person.landmark_wave.eq(person.index_shock_wave + 1).all(), "非exact相邻波次")
    person["interval_years"] = (month_index(person.shock_interview_year, person.shock_interview_month)
                                  - month_index(person.pre_interview_year, person.pre_interview_month)) / 12
    crisis_indicators(person)
    frame, lm, original_x, beta, covariance, diagnostic, anchor = load_cohort(cohort)
    primary = person.loc[person.person_id.isin(lm.person_id)].copy()
    require(len(primary) == anchor["participants"], "Primary参与者连接缺失")
    primary_dates = primary.set_index("person_id")
    require(lm.interview_year.eq(lm.person_id.map(primary_dates.lm_interview_year)).all()
            and lm.interview_month.eq(lm.person_id.map(primary_dates.lm_interview_month)).all(), "原landmark时间改变")
    require(frame.index_shock_wave.eq(frame.person_id.map(primary_dates.index_shock_wave)).all(), "模型行冲击身份改变")
    for field in ("pre_interview_year", "pre_interview_month", "shock_interview_year", "shock_interview_month"):
        require(primary[field].notna().all(), "Primary冲击时间未知，不能锁定时期")
    support = frame.groupby("person_id").agg(FI_rows=("wave", "size"),
        later_FI_rows=("fi_time_role", lambda x: int(x.eq("later").sum())), last_FI_time=("time_years", "max"))
    primary = primary.merge(support, left_on="person_id", right_index=True, validate="one_to_one", sort=False)
    x, design = build_design(frame, original_x, cohort)
    design["anchor"] = anchor
    design["primary_reconstruction"] = reconstruct_primary(frame, original_x, beta, diagnostic)
    groups = primary.groupby(["index_shock_wave", "recovery_status"]).size().unstack(fill_value=0)
    require((groups > 0).all().all(), "波次与恢复组完全分离")
    design["wave_group_counts"] = groups.to_dict(orient="index")
    latest = primary.loc[primary.index_shock_wave.eq(LEVELS[cohort][-1])]
    require(latest.later_FI_rows.eq(1).all(), "最新波次后续支持锚点改变")
    design["latest_wave_support"] = [{"group": g, "participants": len(b), "FI_rows": int(b.FI_rows.sum()),
        "later_rows": int(b.later_FI_rows.sum()), "max_followup": float(b.last_FI_time.max())}
        for g, b in latest.groupby("recovery_status")]
    return person, primary, frame, spine, design


def label(value):
    return "UNKNOWN" if pd.isna(value) else str(int(value)) if isinstance(value, (int, float, np.number)) else str(value)


def distributions(cohort, pools):
    """每个比例写明真实分母；恢复未知者单列，恢复率另用可分类分母。"""
    rows = []
    for population, person in pools.items():
        for dimension, field in (("shock_year", "shock_interview_year"), ("shock_wave", "index_shock_wave"),
                                 ("landmark_year", "lm_interview_year")):
            for value, part in person.groupby(field, dropna=False, sort=True):
                common = {"cohort": cohort, "population": population, "population_definition": SOURCE_ROLES[population],
                          "dimension": dimension, "period": label(value), "unit": "unique_participants",
                          "population_n": len(person), "period_n": len(part), "internal_review_only": True}
                rows.append({**common, "metric": "period_distribution", "group": "ALL", "n": len(part),
                    "denominator": len(person), "proportion": len(part) / len(person),
                    "denominator_role": "whole_population"})
                known = part.recovery_status.notna()
                for group, block in part.groupby("recovery_status", dropna=False):
                    g = label(group)
                    group_total = int(person.recovery_status.isna().sum()) if g == "UNKNOWN" else int(person.recovery_status.eq(group).sum())
                    rows.append({**common, "metric": "group_composition_within_period", "group": g, "n": len(block),
                        "denominator": len(part), "proportion": len(block) / len(part), "denominator_role": "all_in_period"})
                    rows.append({**common, "metric": "period_distribution_within_group", "group": g, "n": len(block),
                        "denominator": group_total, "proportion": len(block) / group_total,
                        "denominator_role": "whole_recovery_group_or_unknown_group"})
                nknown = int(known.sum())
                recovered = int(part.recovery_status.eq("substantial_recovery").sum())
                rows.append({**common, "metric": "recovery_proportion", "group": "CLASSIFIABLE", "n": recovered,
                    "denominator": nknown, "proportion": recovered / nknown if nknown else np.nan,
                    "denominator_role": "classifiable_recovery_only"})
    return rows


def crisis_summary(cohort, pools):
    rows = []
    for population, person in pools.items():
        for definition, indicator in crisis_indicators(person).items():
            temp = person.assign(indicator=indicator)
            for group, block in [("ALL", temp), *temp.groupby("recovery_status", dropna=False)]:
                for state in (True, False, pd.NA):
                    mask = block.indicator.isna() if pd.isna(state) else block.indicator.eq(state).fillna(False)
                    n = int(mask.sum())
                    rows.append({"cohort": cohort, "population": population, "definition": definition,
                        "group": label(group), "state": "UNKNOWN" if pd.isna(state) else "YES" if state else "NO",
                        "n": n, "denominator": len(block), "proportion": n / len(block),
                        "interval_definition": "(pre_interview_month,shock_interview_month] overlaps [2008-01,2009-12]",
                        "does_not_identify_shock_cause": True, "internal_review_only": True})
    return rows


def support_summary(cohort, person, primary, frame, spine):
    rows = []
    for population, participants in (("A_INDEX_SHOCK", person), ("B_PRIMARY", primary)):
        for wave, part in participants.groupby("index_shock_wave"):
            for group, block in [("ALL", part), *part.groupby("recovery_status", dropna=False)]:
                r = {"cohort": cohort, "population": population, "shock_wave": int(wave), "group": label(group),
                     "participants": len(block), "internal_review_only": True,
                     "opportunity_is_person_specific": False, "not_a_survival_or_censoring_risk_set": True}
                for field in ("pre_interview_year", "shock_interview_year", "lm_interview_year", "interval_years"):
                    v = block[field].dropna().astype(float)
                    r[field + "_known_n"] = len(v)
                    r[field + "_unknown_n"] = int(block[field].isna().sum())
                    for q, name in ((0, "min"), (.25, "q1"), (.5, "median"), (.75, "q3"), (1, "max")):
                        r[field + "_" + name] = float(v.quantile(q)) if len(v) else np.nan
                r["landmark_FI_available_n"] = int(block.landmark_fi_available.sum())
                r["later_FI_rows_ledger"] = int(block.later_fi_count.sum())
                r["two_or_more_later_FI_ledger_n"] = int(block.later_fi_count.ge(2).sum())
                if population == "B_PRIMARY":
                    r["FI_rows"] = int(block.FI_rows.sum())
                    r["landmark_rows"] = len(block)
                    r["later_FI_rows"] = int(block.later_FI_rows.sum())
                    r["two_or_more_later_FI_n"] = int(block.later_FI_rows.ge(2).sum())
                    r["two_or_more_total_FI_n"] = int(block.FI_rows.ge(2).sum())
                    r["last_FI_time_median"] = float(block.last_FI_time.median())
                    r["last_FI_time_max"] = float(block.last_FI_time.max())
                    b = frame.loc[frame.person_id.isin(block.person_id)]
                    r["T_gt8_rows"] = int(b.time_years.gt(8).sum())
                    r["T_gt8_participants"] = int(b.loc[b.time_years.gt(8), "person_id"].nunique())
                    r["primary_support_computable"] = True
                else:
                    r["primary_support_computable"] = False
                rows.append(r)
    # 两年分箱仅描述已观察FI，不把未观察波次视为死亡或失访。
    temp = frame.assign(time_bin=np.where(frame.time_years.eq(0), 0, np.ceil(frame.time_years / 2).astype(int)))
    for (wave, group, b), part in temp.groupby(["index_shock_wave", "recovery_status", "time_bin"]):
        rows.append({"cohort": cohort, "population": "B_PRIMARY", "shock_wave": int(wave), "group": group,
            "record_type": "observed_two_year_time_bin", "bin_lower_exclusive": max(0, (int(b)-1)*2),
            "bin_upper_inclusive": int(b)*2, "landmark_bin": int(b) == 0,
            "participants": part.person_id.nunique(), "FI_rows": len(part),
            "internal_review_only": True, "not_a_survival_or_censoring_risk_set": True})
    return rows


def docx_paragraphs(data):
    with zipfile.ZipFile(BytesIO(data)) as z:
        e = ET.fromstring(z.read("word/document.xml"))
    ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    return ["".join(t.text or "" for t in p.findall(".//w:t", ns)) for p in e.findall(".//w:p", ns)]


def pdf_text(path):
    return subprocess.run(["pdftotext", "-layout", "-enc", "UTF-8", str(path), "-"],
                          capture_output=True, check=True).stdout.decode("utf-8")


def study_year_mentions(paragraphs):
    """引用标题的年份不冒充研究时间；正文位置保留供作者定位。"""
    result, in_references = [], False
    for i, t in enumerate(paragraphs, 1):
        if t.strip().lower() in {"references", "reference", "参考文献"}:
            in_references = True
        if not in_references and (re.search(r"2002\s*[-–—]\s*202[03]", t) or
                                  "survey periods" in t.lower() or "survey period" in t.lower()):
            result.append({"paragraph": i, "text": t, "scope": "study_body_not_reference"})
    return result


def document_audit(records):
    """登记当前源码、投稿副本和容器内历史版本；不假设本机文件等于实际上传版本。"""
    result = []
    docs = ("07_manuscript/Manuscript.docx", "07_manuscript/Round1/healthcare-4603924.docx",
            "07_manuscript/Submit/healthcare-4603924-precheck/healthcare-4603924-precheck_revised.docx")
    for name in docs:
        p = ROOT / name
        records[name] = sha(p)
        result.append({"path": name, "sha256": records[name], "artifact_role": "local_manuscript_copy",
            "mentions": study_year_mentions(docx_paragraphs(p.read_bytes())),
            "actual_journal_upload_identity_verified": False})
    name = "07_manuscript/supplementary_materials/tables/table_s1.tex"
    p = ROOT / name
    records[name] = sha(p)
    result.append({"path": name, "sha256": records[name], "artifact_role": "current_supplement_source",
        "mentions": [{"line": i, "text": line} for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1)
                     if "ELSA" in line or "2021" in line]})
    for name in ("07_manuscript/Submit/Supplementary_Material.pdf",
                 "07_manuscript/supplementary_materials/output/Supplementary_Material.pdf"):
        p = ROOT / name
        records[name] = sha(p)
        pages = pdf_text(p).split("\f")
        mentions = []
        for i, text in enumerate(pages, 1):
            if "Table S1" in text and "ELSA" in text:
                mentions.append({"page": i, "text": text.strip(),
                    "contains_2002_2023": bool(re.search(r"2002\s*[-–—]\s*2023", text)),
                    "contains_2021_2023": bool(re.search(r"2021\s*[-–—]\s*2023", text))})
        require(mentions, "Supplement缺少可定位的S1，停止一致性结论")
        result.append({"path": name, "sha256": records[name], "artifact_role": "local_compiled_or_submit_copy",
                       "mentions": mentions, "actual_journal_upload_identity_verified": False})
    for name in ("07_manuscript/Submit/Supplementary_Material.zip", "07_manuscript/Submit/healthcare-4603924-precheck.zip"):
        p = ROOT / name
        records[name] = sha(p)
        members = []
        with zipfile.ZipFile(p) as z:
            for entry in z.infolist():
                if entry.is_dir():
                    continue
                data = z.read(entry)
                r = {"name": entry.filename, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
                if entry.filename.lower().endswith(".docx"):
                    r["mentions"] = study_year_mentions(docx_paragraphs(data))
                if entry.filename.lower().endswith(".pdf"):
                    text = subprocess.run(["pdftotext", "-layout", "-enc", "UTF-8", "-", "-"],
                        input=data, capture_output=True, check=True).stdout.decode("utf-8")
                    r["S1_mentions"] = [{"page": i, "text": page.strip()}
                        for i, page in enumerate(text.split("\f"), 1) if "Table S1" in page and "ELSA" in page]
                    r["study_range_candidates"] = [{"page": i, "text": line.strip(),
                        "requires_context_review_not_automatic_error": True}
                        for i, page in enumerate(text.split("\f"), 1) for line in page.splitlines()
                        if re.search(r"2002\s*[-–—]\s*202[03]", line)]
                if entry.filename == "Supplementary_Material.pdf":
                    r["matches_current_submit_pdf"] = r["sha256"] == records["07_manuscript/Submit/Supplementary_Material.pdf"]
                members.append(r)
        result.append({"path": name, "sha256": records[name], "artifact_role": "local_submission_archive",
                       "members": members, "actual_journal_upload_identity_verified": False})
    return result


def document_time_decision(documents):
    """差异分类依据实际读取的副本，不将未知上传身份冒充完整核实。"""
    compiled = [d for d in documents if d["artifact_role"] == "local_compiled_or_submit_copy"]
    source = next(d for d in documents if d["artifact_role"] == "current_supplement_source")
    source_ok = any("ELSA" in m["text"] and "2002--2023" in m["text"] for m in source["mentions"])
    compiled_ok = all(any(m["contains_2002_2023"] and m["contains_2021_2023"] for m in d["mentions"]) for d in compiled)
    body_old = [{"path": d["path"], **m} for d in documents if d["artifact_role"] == "local_manuscript_copy"
                for m in d["mentions"] if re.search(r"2002\s*[-–—]\s*2020", m["text"])]
    archive_changes = [{"path": d["path"], "member": m["name"], "classification": "LOCAL_ARCHIVE_DIFFERS_FROM_CURRENT_SUBMIT_COPY"}
        for d in documents for m in d.get("members", []) if m.get("matches_current_submit_pdf") is False]
    return {"current_source_matches_actual_ELSA_year_coverage": source_ok,
        "current_compiled_copies_state_2002_2023_and_W10_2021_2023": compiled_ok,
        "current_body_2002_2020_findings": body_old, "local_archive_differences": archive_changes,
        "classification": "LOCAL_CURRENT_COPIES_CONSISTENT_EXTERNAL_UPLOAD_IDENTITY_PENDING"
            if source_ok and compiled_ok and not body_old else "NONBLOCKING_MANUSCRIPT_TIME_DESCRIPTION_REVIEW",
        "actual_journal_upload_identity_verified": False,
        "not_a_statistical_source_error_by_itself": True, "manuscript_modified": False}


def elsa_year_evidence(frame, primary):
    w10 = frame.loc[frame.wave.eq(10)]
    counts = w10.groupby(["interview_year", "recovery_status"]).size().unstack(fill_value=0)
    expected = {2021: (9, 14), 2022: (83, 146), 2023: (9, 11)}
    require(set(counts.index) == set(expected) and len(w10) == w10.person_id.nunique() == 272, "ELSA W10年份锚点漂移")
    for year, (r, c) in expected.items():
        require(int(counts.loc[year, "substantial_recovery"]) == r and
                int(counts.loc[year, "persistent_or_limited_recovery"]) == c, "ELSA晚期FI年份锚点漂移")
    late = primary.loc[primary.index_shock_wave.eq(8)]
    require(len(late) == 64 and late.FI_rows.eq(2).all() and late.later_FI_rows.eq(1).all(), "ELSA W8时点支持漂移")
    require(late.recovery_status.eq("substantial_recovery").sum() == 19 and
            late.recovery_status.eq("persistent_or_limited_recovery").sum() == 45, "ELSA W8两组人数漂移")
    return {"source_survey_waves": "W1-W10", "nominal_W10_year": 2020,
        "primary_FI_year_range": [int(frame.interview_year.min()), int(frame.interview_year.max())],
        "primary_shock_year_range": [int(primary.shock_interview_year.min()), int(primary.shock_interview_year.max())],
        "primary_landmark_year_range": [int(primary.lm_interview_year.min()), int(primary.lm_interview_year.max())],
        "W10_rows": 272, "W10_participants": 272, "W10_year_group_counts": counts.to_dict(orient="index"),
        "W8_recovery_participants": 19, "W8_reference_participants": 45,
        "W8_total_FI_per_person": 2, "W8_later_FI_per_person": 1,
        "administrative_individual_censor_date_inferred": False}


def assert_aggregate(frame):
    forbidden = {"person_id", "hhidpn", "idauniq", "pid", "fi_primary", "individual_FI"}
    require(not forbidden.intersection(frame.columns), "聚合交付包含个人记录字段")


def create_targets(audit_id, progress_root, output_root):
    """同时检查两个目标后才创建，路径保持在指定根目录内。"""
    require(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,95}", audit_id) is not None, "非法audit ID")
    targets = [Path(progress_root) / audit_id, Path(output_root) / audit_id]
    require(all(not p.exists() for p in targets), "拒绝覆盖已有审计")
    for p, parent in zip(targets, (progress_root, output_root)):
        require(p.resolve().is_relative_to(Path(parent).resolve()), "输出路径越界")
    for p in targets:
        p.mkdir(parents=True, exist_ok=False)
    return tuple(targets)
