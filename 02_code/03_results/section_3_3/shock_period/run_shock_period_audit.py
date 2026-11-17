"""Round 1E Stage 0唯一命名入口：六份审计产物，禁止拟合与阶段自动推进。"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

from sp_audit import (ROOT, CODE, COHORTS, INTERPRETATION, FORMULA, no_models,
    verify_all_sources, cohort_data, distributions, crisis_summary, support_summary,
    document_audit, document_time_decision, elsa_year_evidence, assert_aggregate, create_targets,
    snapshot, verify_records, write_json, sha, rel, require, clean, provenance)

PROGRESS = ROOT / "06_project_progress/round1e_shock_period_v1/Stage0"
OUTPUT = ROOT / "03_output/3.3 Robustness of the Main Findings/Shock Calendar Period/Audit"


def test_evidence(path):
    """核对实际JUnit结果，拒绝把未执行或失败测试写成通过。"""
    element = ET.parse(path).getroot()
    suites = [element] if element.tag == "testsuite" else element.findall(".//testsuite")
    counts = {key: sum(int(s.attrib.get(key, 0)) for s in suites) for key in ("tests", "failures", "errors", "skipped")}
    require(counts["tests"] > 0 and counts["failures"] == counts["errors"] == counts["skipped"] == 0, "必要测试未全部通过")
    return {"path": str(Path(path).resolve()), "sha256": sha(path), **counts,
            "command": "uv run --locked python -B -m pytest 02_code/03_results/section_3_3/shock_period/test_sp_audit.py -q -p no:cacheprovider --junitxml=<path>",
            "role": "INTERNAL_TEST_EVIDENCE_NOT_STATISTICAL_INPUT"}


def build_report(audit_id, designs, documents, elsa, manifest, period, support, crisis):
    lines = [f"# Round 1E Stage 0审计报告：{audit_id}", "",
        "状态：**VALIDATED_FOR_STAGE1_REVIEW**。本轮完成来源、时期支持与设计预审，没有真实拟合。",
        "三个SP01仅为拟议规格；Stage 1合同未创建、未锁定，真实拟合未授权。", "",
        "## 来源与受保护资产", "",
        f"实际来源登记{len(manifest['sources'])}项；既有受保护资产{len(manifest['protected_hashes_before'])}项前后SHA不变。",
        "Primary的23项活动输出及统计来源通过；12项历史呈现异常继续作为非阻断例外，不改写旧freeze。",
        "Round 1A–1D发布清单、外部统计来源和历史源码快照核验通过。统计输入异常默认阻断。",
        "完整SHA、冻结来源、实际执行源码及环境见stage0_manifest.json。", "",
        "## 原Primary与SP01", "",
        "原Primary无冲击日历时期或冲击波次调整；其42列及全部原协变量×时间保留。",
        "年龄等一般协变量取t，财富五分位取t−1，冲击FI取t；结构资格的t−1年龄条件不变。",
        "SP01在原无权重ML随机截距模型增加分类index_shock_wave主效应及波次×time_years。",
        "HRS参照W3，ELSA与KLoSA参照W2；时间仍从原landmark起算，保留全部冻结FI行。", "",
        "> " + INTERPRETATION, "",
        "这是波次调整敏感性，不是时期异质性分析。波次×时间项不是各时期财富恢复效应。",
        "不能回答金融危机期间恢复是否有不同健康关联，不能估计宏观冲击的因果效应。", "",
        "| 队列 | A首次冲击人数 | Primary人数/行数 | SP01列/秩 | 原始条件数 | 列标准化条件数 |",
        "|---|---:|---:|---:|---:|---:|"]
    for c, d in designs.items():
        a = d["anchor"]
        n = period.loc[(period.cohort == c) & (period.population == "A_INDEX_SHOCK"), "population_n"].iloc[0]
        lines.append(f"| {c} | {n} | {a['participants']}/{a['rows']} | {d['n_columns']}/{d['rank']} | {d['condition_number']:.3f} | {d['column_norm_scaled_condition']:.3f} |")
    lines += ["", "矩阵满秩仅支持预审可识别性，不证明拟合收敛、时期项精度或核心CI可靠。",
        "KLoSA保留完整42列Primary基底，不套用Round 1C post-only的41列参数化。", "",
        "## 分母、实际年份及纵向支持", "",
        "A_INDEX_SHOCK为窗口内首次合格冲击；不按恢复、FI或未来回访筛选。B_PRIMARY为冻结Primary。",
        "恢复不可分类者单列UNKNOWN；恢复比例仅以恢复可分类者为分母，不当成参照组。",
        "分布人数不是波次冲击发生率：没有用各波次风险人群作为发生率分母。",
        "HRS冲击观察年1995–2019（W3–W14），ELSA为2004–2017（W2–W8），KLoSA为2008–2018（W2–W7）。",
        "HRS来源为RAND访谈中点r{wave}iwmid，ELSA/KLoSA为r{wave}iwy/iwm。名义年份不代替实际年月。",
        "全部冲击前/冲击访谈时间均已知，间隔为正；并非财富损失发生的精确日期。",
        "单年、波次、landmark及UNKNOWN分布见period_distribution.csv；区间与两年时间分箱见time_support.csv。", "",
        "| 队列/冲击波次 | 实际冲击年 | A人数 | Primary恢复/参照 | FI行 | 至少两次后续FI人数 | 末次FI中位年 |",
        "|---|---|---:|---:|---:|---:|---:|"]
    for c in COHORTS:
        all_support = support.loc[(support.cohort == c) & (support.population == "A_INDEX_SHOCK") & support.group.eq("ALL")]
        for _, r in support.loc[(support.cohort == c) & (support.population == "B_PRIMARY") & support.group.eq("ALL")].iterrows():
            w = int(r.shock_wave)
            ar = all_support.loc[all_support.shock_wave.eq(w)].iloc[0]
            groupcounts = designs[c]["wave_group_counts"][w]
            yr = f"{int(r.shock_interview_year_min)}–{int(r.shock_interview_year_max)}"
            lines.append(f"| {c} W{w} | {yr} | {int(ar.participants)} | {groupcounts['substantial_recovery']}/{groupcounts['persistent_or_limited_recovery']} | {int(r.FI_rows)} | {int(r.two_or_more_later_FI_n)} | {r.last_FI_time_median:.3f} |")
    lines += ["", "各波次都有两组支持。单年存在极小单元，不推荐新增单年模型或结果导向的时期合并。",
        "最新冲击波次每人仍有landmark及一次后续FI，两次总观察不是不可估计的充分理由。",
        "晚期随访较短与剩余波次有限有关；末次FI不代表退出日、死亡日或个人行政截尾日。",
        "A中的later_FI_rows_ledger是冻结Ledger支持摘要，不能冒充Primary timed-FI行数或完整行政机会。", "",
        "## 金融危机描述", "",
        "分别描述冲击访谈年2008/2009与实际区间(t−1访谈月,t访谈月]同[2008-01,2009-12]重叠。",
        "缺失年月保留UNKNOWN，不填补日期。指标不等于财富损失由金融危机造成，三国不视为同质经济暴露。", "",
        "| 队列 | A访谈指标/重叠指标人数 | Primary访谈指标/重叠指标人数 |",
        "|---|---:|---:|"]
    for c in COHORTS:
        numbers = []
        for pop in ("A_INDEX_SHOCK", "B_PRIMARY"):
            for definition in ("shock_interview_2008_2009", "interval_overlap_2008_2009"):
                numbers.append(int(crisis.loc[(crisis.cohort == c) & (crisis.population == pop) & crisis.definition.eq(definition)
                    & crisis.group.eq("ALL") & crisis.state.eq("YES"), "n"].iloc[0]))
        lines.append(f"| {c} | {numbers[0]}/{numbers[1]} | {numbers[2]}/{numbers[3]} |")
    lines += ["", "KLoSA没有2008年前冲击类别，不能作完整危机前中后比较。本轮不拟合危机模型。", "",
        "## ELSA四层时间一致性核查", "",
        "区分来源W1–W10的实际覆盖、名义W10=2020、Primary使用的年份及稿件描述。",
        "ELSA Primary FI为2006–2023；冲击访谈2004–2017；landmark为2006–2019。",
        "2021/2022/2023年W10的FI行分别23/229/20，共272行、272人；恢复101人，参照171人。",
        f"当前Table S1源码年份与数据一致={manifest['document_time_decision']['current_source_matches_actual_ELSA_year_coverage']}；编译PDF同时说明2002–2023及W10 2021–2023={manifest['document_time_decision']['current_compiled_copies_state_2002_2023_and_W10_2021_2023']}。",
        f"当前独立正文2002–2020研究范围候选差异{len(manifest['document_time_decision']['current_body_2002_2020_findings'])}项；参考文献年份已排除。",
        "下列是本机保存的稿件和提交副本身份。未获得期刊上传回执，因此不把本机副本等同于已证实上传内容。", ""]
    for d in documents:
        lines.append(f"- `{d['path']}`，SHA `{d['sha256']}`。")
        for m in d.get("mentions", []):
            if "page" in m:
                lines.append(f"  PDF第{m['page']}页：2002–2023={m['contains_2002_2023']}；2021–2023={m['contains_2021_2023']}。")
            elif "text" in m:
                lines.append(f"  位置{m.get('line', m.get('paragraph'))}：{m['text']}")
        for m in d.get("members", []):
            if "matches_current_submit_pdf" in m:
                lines.append(f"  ZIP内`{m['name']}`与当前Submit PDF同字节={m['matches_current_submit_pdf']}。")
            for mention in m.get("mentions", []):
                lines.append(f"  ZIP内`{m['name']}`段落{mention['paragraph']}：{mention['text']}")
            for mention in m.get("study_range_candidates", []):
                lines.append(f"  ZIP内`{m['name']}`第{mention['page']}页范围文本（上下文审阅，不自动认定错误）：{mention['text']}")
    lines += ["", f"时间描述分类：{manifest['document_time_decision']['classification']}；上传身份仍属外部待核实。",
        "不修改论文，不截去2021–2023年观察；若后续发现历史副本差异，在统一返修时登记和处理。", "",
        "## 后续ELSA专项QA与解释限制", "",
        "ELSA W8保留19名恢复者、45名参照者，各两次总FI、一次后续FI；不提前合并或删除。",
        "Stage 3保存波次×时间参数SE/CI、完整信息谱、白化SVD和参数相关性；比较优化器及独立观察信息协方差。",
        "用已拟合边际协方差下个体核心系数得分贡献按波次汇总，结合晚期两组支持和残差筛查。",
        "得分筛查不是删波次/删个体拟合或方差重估；不作独立模型差异检验。",
        "宽CI、弱识别与数值失败分开；不引入任意人数或CI宽度通过线。核心CI失效仍HOLD。",
        "若影响集中或支持不足，报告‘晚期观察敏感性尚未充分排除’，追加分析等待作者批准。", "",
        "## 阶段、数值政策和最低必要决定", "",
        "三队列均PERIOD_MODEL_FEASIBLE，建议仅三个SP01。当前状态仅准入Stage 1审阅，不是拟合批准。",
        "Stage 1另行创建一个完整合同，保存作者批准；Stage 2需一次明确授权三个单元。",
        "拟议计算策略：BFGS gtol=1e-8，L-BFGS pgtol=1e-10、factr=100，maxiter=2000。",
        "loglik差<1e-4，核心估计差<1e-6，核心SE相对差≤1e-3，内点标准化梯度≤1e-3。",
        "完整终止消息、底层设置、warning、score、Hessian及谱都保留；不重复HRS/PO01终止原因丢失的问题。",
        "Stage 2仅PASS/BOUNDARY_REVIEW/HOLD_NOT_ESTIMABLE；不能自动接受附限定结果，合法边界按约束检查。",
        "Stage 3完成独立数值与科学审阅，Stage 4作者批准具体结果后冻结，Stage 5只读指定release制表。",
        "拟议Stage 5为英文六行Primary/SP01表及时期支持摘要；不默认新图或论文修改。",
        "不自动增加时期替代模型、危机效应、恢复时期交互、RS/NL/LM组合、IPCW或meta。",
        "Methods解释代理和共同速度差；Results报告队列估计/精度；Supplement给完整支持/QA；Response Letter对应编辑意见。",
        "与Round 1B分别说明随机结构、时间函数与时期调整敏感性，不按P值、AIC或跨零选择模型。",
        "时期调整不能消除财富误差、死亡/失访选择或未测量混杂，也不能识别危机贡献。", "",
        "## 完成本轮实际改动及停止点", "",
        f"必要测试{manifest['tests']['tests']}项通过，模型与优化调用均0，既有资产哈希核验通过。",
        "新增独立Stage 0代码、测试、README和六份审计产物；不另存微观Panel。",
        "聚合细单元均为内部审阅，不是候选投稿表。没有创建合同、授权、结果模型、release或freeze。",
        "没有改动Primary、Round 1A–1D、正文、Supplement、Figure 2/3或Table S6；没有commit/push。",
        "停止于Stage 0，等待作者审阅并决定进入Stage 1。", ""]
    return "\n".join(lines)


def run(audit_id, junit):
    tests = test_evidence(junit)
    before = snapshot()
    counter = {"model_fit_calls": 0, "optimizer_calls": 0, "forbidden_calls": 0}
    with no_models(counter):
        sources = verify_all_sources()
        records = sources["records"]
        documents = document_audit(records)
        periods, supports, crises, designs = [], [], [], {}
        elsa = None
        for cohort in COHORTS:
            person, primary, frame, spine, design = cohort_data(sources, cohort)
            designs[cohort] = design
            pools = {"A_INDEX_SHOCK": person, "B_PRIMARY": primary}
            periods.extend(distributions(cohort, pools))
            supports.extend(support_summary(cohort, person, primary, frame, spine))
            crises.extend(crisis_summary(cohort, pools))
            if cohort == "ELSA":
                elsa = elsa_year_evidence(frame, primary)
        frames = {"period_distribution.csv": pd.DataFrame(periods),
                  "time_support.csv": pd.DataFrame(supports), "crisis_overlap.csv": pd.DataFrame(crises)}
        for frame in frames.values():
            assert_aggregate(frame)
        require(elsa is not None, "ELSA验收缺失")
        verify_records(records)
        verify_records(before)
        require(counter["forbidden_calls"] == 0, "出现禁用执行尝试")
        pdir, odir = create_targets(audit_id, PROGRESS, OUTPUT)
        try:
            for name, frame in frames.items():
                with (odir / name).open("x", encoding="utf-8", newline="") as handle:
                    frame.to_csv(handle, index=False)
            design_result = {"status": "VALIDATED_FOR_STAGE1_REVIEW", "cohorts": designs,
                "specification_proposal": {"formula_base": FORMULA, "add": "shock_wave + shock_wave:time_years",
                    "interpretation": INTERPRETATION, "no_period_specific_recovery_associations": True,
                    "science_approval": "PENDING_STAGE1", "real_fit_authorization": "NOT_GRANTED",
                    "ELSA_stage3_requirements": ["wave_time_identification", "full_covariance_stability",
                        "fixed_covariance_individual_core_score_screen_no_refit"]},
                "ELSA_time_evidence": elsa}
            write_json(odir / "design_diagnostics.json", design_result)
            manifest = {"schema": "round1e_stage0_audit_manifest_v1", "audit_id": audit_id,
                "status": "VALIDATED_FOR_STAGE1_REVIEW", "stage": "Stage0", **provenance(),
                "sources": records, "releases_verified": sources["releases"],
                "historical_exceptions": sources["prior_source_check"]["historical_exceptions"],
                "document_time_evidence": documents, "document_time_decision": document_time_decision(documents),
                "actual_journal_upload_identity_verified": False,
                "ELSA_time_evidence": elsa, "tests": tests, "protected_hashes_before": before,
                "protected_hashes_after": before.copy(), "protected_assets_unchanged": True,
                "participants_exported": False, "internal_review_only": True, **counter,
                "stage1_contract_created": False, "stage1_author_approval": "PENDING",
                "stage2_authorization": "NOT_GRANTED", "stage3_executed": False,
                "freeze_release_created": False, "manuscript_modified": False,
                "git_commit_push_performed": False, "outputs": {}}
            report = build_report(audit_id, designs, documents, elsa, manifest,
                                  frames["period_distribution.csv"], frames["time_support.csv"], frames["crisis_overlap.csv"])
            with (pdir / "stage0_audit_report.md").open("x", encoding="utf-8", newline="\n") as handle:
                handle.write(report)
            verify_records(before)
            verify_records(records)
            require(sha(junit) == tests["sha256"], "测试证据变化")
            artifacts = [pdir / "stage0_audit_report.md", *odir.iterdir()]
            manifest["outputs"] = {rel(p): sha(p) for p in artifacts}
            manifest["aggregate_row_counts"] = {name: len(frame) for name, frame in frames.items()}
            write_json(pdir / "stage0_manifest.json", manifest)
        except Exception as error:
            if not (pdir / "stage0_manifest.json").exists():
                write_json(pdir / "stage0_manifest.json", {"audit_id": audit_id, "status": "HOLD_AUDIT_FAILED",
                    "error": str(error), "sources": records, "model_fit_calls": 0, "optimizer_calls": 0})
            raise
    return {"audit_id": audit_id, "status": "VALIDATED_FOR_STAGE1_REVIEW", "report": rel(pdir / "stage0_audit_report.md"),
            "output": rel(odir), "source_files": len(records), "protected_files": len(before), "tests": tests["tests"], **counter}


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-id", required=True)
    parser.add_argument("--test-evidence", type=Path, required=True)
    args = parser.parse_args()
    import json
    print(json.dumps(clean(run(args.audit_id, args.test_evidence)), ensure_ascii=False, indent=2))
