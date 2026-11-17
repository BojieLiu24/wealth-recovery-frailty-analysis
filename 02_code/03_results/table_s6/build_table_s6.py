from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
from datetime import UTC, datetime
from pathlib import Path

from docx import Document
from docx.enum.section import WD_ORIENT
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT, WD_ROW_HEIGHT_RULE, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor


ROOT = Path(__file__).resolve().parents[3]
SECTION32 = ROOT / "03_output/3.2 Wealth Recovery and Subsequent Frailty Progression"
SECTION33 = ROOT / "03_output/3.3 Robustness of the Main Findings"
OUTPUT = SECTION33 / "Table S6"
COHORT_ORDER = {"HRS": 1, "ELSA": 2, "KLoSA": 3, "Overall (random-effects)": 4}
BLOCKS = [
    ("Primary analysis", "primary"),
    ("Recovery threshold: 25%", "exposure"),
    ("Recovery threshold: 100% (return to pre-shock wealth)", "exposure"),
    ("Common-22 frailty index", "outcome"),
    ("Landmark-weighted analysis", "weighting"),
    ("Multiple-imputation analysis", "mi"),
]
CAPTION = "Supplementary Table S6. Primary and sensitivity estimates for wealth recovery and frailty progression."
NOTES = (
    "Estimates are recovery-by-time interactions from cohort-specific longitudinal models and represent the adjusted difference in annual FI change (FI units/year) between substantial recovery and persistent/limited recovery. "
    "Negative estimates indicate slower annual FI increase in the substantial-recovery group. Primary models were unweighted. The landmark-weighted analysis used positive exact-landmark survey weights rescaled to a mean of one within cohort and participant-level CR0 standard errors. "
    "No cross-cohort summary was generated for the landmark-weighted analysis. Multiple-imputation estimates were combined within cohort using Rubin's rules. Participants and FI model observations may differ across specifications; FI model observations include the landmark and all valid later model records. "
    "Recovery-threshold analyses reclassified substantial recovery at 25% or 100% of shock-related wealth loss regained at the landmark. The Common-22 analysis used the 22 conceptually aligned frailty deficits across cohorts. Overall rows are secondary random-effects summaries of cohort-specific estimates and do not represent participant-level pooled models. FI, frailty index; MI, multiple imputation; HRS, Health and Retirement Study; ELSA, English Longitudinal Study of Ageing; KLoSA, Korean Longitudinal Study of Aging."
)
RELEASE_VERSION = "table_s6_v2"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def finite(value: str) -> float:
    x = float(value)
    if not math.isfinite(x):
        raise RuntimeError(f"非有限数值：{value}")
    return x


def verify_freeze(directory: Path) -> str:
    freeze_path = directory / "freeze.json"
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    if freeze.get("freeze_status") != "FROZEN" or freeze.get("manifest_sha256") != sha(directory / "manifest.json"):
        raise RuntimeError(f"freeze无效：{directory}")
    return sha(freeze_path)


def source_rows() -> tuple[list[dict], dict[str, str]]:
    inputs: dict[str, str] = {}
    primary_path = SECTION32 / "Primary Models/table_s6_primary_rows.csv"
    figure_path = SECTION32 / "Figure 2/figure_2_source.csv"
    meta_path = SECTION32 / "Primary Models/primary_lmm_meta_summary.csv"
    v3_freeze = SECTION32 / "section_3_2_freeze.json"
    v3 = json.loads(v3_freeze.read_text(encoding="utf-8"))
    if v3.get("freeze_version") != "section_3_2_v3" or v3.get("freeze_status") != "FROZEN":
        raise RuntimeError("primary section_3_2_v3未冻结")
    inputs[str(primary_path.relative_to(ROOT)).replace("\\", "/")] = sha(primary_path)
    inputs[str(figure_path.relative_to(ROOT)).replace("\\", "/")] = sha(figure_path)
    inputs[str(meta_path.relative_to(ROOT)).replace("\\", "/")] = sha(meta_path)
    inputs[str(v3_freeze.relative_to(ROOT)).replace("\\", "/")] = sha(v3_freeze)
    rows: list[dict] = []
    for row in read_csv(primary_path):
        rows.append({"analysis_family": "primary", "analysis_spec": "Primary analysis", "cohort": row["cohort"], "is_overall": False,
                     "participants": int(row["participants"]), "fi_observations": int(row["model_rows"]),
                     "estimate": finite(row["estimate"]), "se": finite(row["se"]), "ci_lower": finite(row["ci_lower"]), "ci_upper": finite(row["ci_upper"]),
                     "source_result_object": "section_3_2_v3/primary_lmm_result_object.json", "source_freeze_hash": sha(v3_freeze)})
    overall = [r for r in read_csv(figure_path) if r["cohort"] == "Overall (random-effects)"]
    if len(overall) != 1:
        raise RuntimeError("primary Overall锚点不唯一")
    row = overall[0]
    meta_row = read_csv(meta_path)[0]
    rows.append({"analysis_family": "primary", "analysis_spec": "Primary analysis", "cohort": row["cohort"], "is_overall": True,
                 "participants": "3 cohorts", "fi_observations": None, "estimate": finite(row["estimate"]), "se": finite(meta_row["se"]),
                 "ci_lower": finite(row["ci_lower"]), "ci_upper": finite(row["ci_upper"]),
                 "source_result_object": "section_3_2_v3/primary_lmm_result_object.json", "source_freeze_hash": sha(v3_freeze)})
    domains = [("Exposure Definition", "exposure"), ("Outcome Definition", "outcome"), ("Weighting", "weighting"), ("Missing Data", "mi")]
    for directory_name, family in domains:
        directory = SECTION33 / directory_name
        freeze_hash = verify_freeze(directory)
        result_path = directory / "result_object.json"
        inputs[str((directory / "freeze.json").relative_to(ROOT)).replace("\\", "/")] = freeze_hash
        for item in json.loads(result_path.read_text(encoding="utf-8"))["rows"]:
            rows.append({"analysis_family": family, "analysis_spec": item["analysis_spec"], "cohort": item["cohort"],
                         "is_overall": bool(item["is_overall"]), "participants": item["participants"], "fi_observations": item["fi_observations"],
                         "estimate": float(item["estimate"]), "se": float(item["se"]), "ci_lower": float(item["ci_lower"]), "ci_upper": float(item["ci_upper"]),
                         "source_result_object": str(result_path.relative_to(ROOT)).replace("\\", "/"), "source_freeze_hash": freeze_hash})
    block_order = {name: i + 1 for i, (name, _) in enumerate(BLOCKS)}
    for item in rows:
        item["display_order"] = block_order[item["analysis_spec"]] * 10 + COHORT_ORDER[item["cohort"]]
        item["estimand"] = "adjusted_difference_in_annual_fi_change"
        item["outcome"] = "fi_common22" if item["analysis_family"] == "outcome" else "fi_primary"
        item["recovery_threshold"] = 0.25 if item["analysis_spec"].endswith("25%") else (1.0 if "100%" in item["analysis_spec"] else 0.5)
        item["weighted"] = item["analysis_family"] == "weighting"
        item["imputed"] = item["analysis_family"] == "mi"
    return sorted(rows, key=lambda r: r["display_order"]), inputs


def fmt(x: float) -> str:
    # 显示层抑制四舍五入后产生的负零，底层全精度字段不变。
    if round(float(x), 4) == 0:
        return "0.0000"
    return f"{x:.4f}".replace("-", "−")


def tex_num(x: float) -> str:
    """将显示数字安全写入 LaTeX 数学模式，避免 -- 被解释为破折号。"""
    value = fmt(x)
    if value.startswith("−"):
        return f"$-{value[1:]}$"
    return value


def set_run(run, size: float = 8.5, bold: bool = False, italic: bool = False) -> None:
    run.font.name = "Times New Roman"; run.font.size = Pt(size); run.font.bold = bold; run.font.italic = italic
    run.font.color.rgb = RGBColor(0, 0, 0)
    rpr = run._element.get_or_add_rPr(); fonts = rpr.get_or_add_rFonts(); fonts.set(qn("w:ascii"), "Times New Roman"); fonts.set(qn("w:hAnsi"), "Times New Roman")


def set_cell_border(cell, **edges) -> None:
    tc_pr = cell._tc.get_or_add_tcPr(); borders = tc_pr.first_child_found_in("w:tcBorders")
    if borders is None:
        borders = OxmlElement("w:tcBorders"); tc_pr.append(borders)
    for edge, attrs in edges.items():
        tag = f"w:{edge}"; element = borders.find(qn(tag))
        if element is None: element = OxmlElement(tag); borders.append(element)
        for key, value in attrs.items(): element.set(qn(f"w:{key}"), str(value))


def clear_borders(table) -> None:
    for row in table.rows:
        for cell in row.cells:
            set_cell_border(cell, top={"val": "nil"}, bottom={"val": "nil"}, left={"val": "nil"}, right={"val": "nil"}, insideH={"val": "nil"}, insideV={"val": "nil"})


def add_cell_text(cell, text: str, align, bold: bool = False, italic: bool = False, size: float = 8.5) -> None:
    cell.text = ""; p = cell.paragraphs[0]; p.alignment = align; p.paragraph_format.space_after = Pt(0); p.paragraph_format.space_before = Pt(0); p.paragraph_format.line_spacing = 1.0
    set_run(p.add_run(text), size=size, bold=bold, italic=italic); cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
    tc_pr = cell._tc.get_or_add_tcPr(); mar = tc_pr.first_child_found_in("w:tcMar")
    if mar is None: mar = OxmlElement("w:tcMar"); tc_pr.append(mar)
    # 纵向内边距保持紧凑，确保完整注释与表体同页；横向仍保留足够留白。
    for side, value in (("top", "15"), ("bottom", "15"), ("left", "45"), ("right", "45")):
        element = OxmlElement(f"w:{side}"); element.set(qn("w:w"), value); element.set(qn("w:type"), "dxa"); mar.append(element)


def build_docx(rows: list[dict], path: Path) -> None:
    doc = Document(); section = doc.sections[0]; section.orientation = WD_ORIENT.LANDSCAPE
    section.page_width = Cm(29.7); section.page_height = Cm(21.0); section.left_margin = section.right_margin = Cm(1.2); section.top_margin = section.bottom_margin = Cm(1.0)
    normal = doc.styles["Normal"]; normal.font.name = "Times New Roman"; normal.font.size = Pt(8.5)
    p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.LEFT; p.paragraph_format.space_after = Pt(3); set_run(p.add_run(CAPTION), 9, bold=True)
    table = doc.add_table(rows=1, cols=6); table.alignment = WD_TABLE_ALIGNMENT.CENTER; table.autofit = False
    widths = [Cm(5.0), Cm(3.8), Cm(2.6), Cm(3.0), Cm(5.0), Cm(4.1)]
    headers = ["Analysis specification", "Cohort", "Participants, n", "FI model observations, n", "Adjusted difference in annual FI change, FI units/year", "95% CI"]
    for i, (cell, text) in enumerate(zip(table.rows[0].cells, headers)):
        cell.width = widths[i]; add_cell_text(cell, text, WD_ALIGN_PARAGRAPH.LEFT if i < 2 else WD_ALIGN_PARAGRAPH.RIGHT, bold=True)
    trpr = table.rows[0]._tr.get_or_add_trPr(); repeat = OxmlElement("w:tblHeader"); repeat.set(qn("w:val"), "true"); trpr.append(repeat)
    clear_borders(table)
    line = {"val": "single", "sz": "8", "color": "000000"}
    for cell in table.rows[0].cells: set_cell_border(cell, top=line, bottom=line)
    by_spec = {name: [r for r in rows if r["analysis_spec"] == name] for name, _ in BLOCKS}
    last_data_row = None
    for spec, _ in BLOCKS:
        heading = table.add_row(); heading.height_rule = WD_ROW_HEIGHT_RULE.AT_LEAST
        add_cell_text(heading.cells[0], spec, WD_ALIGN_PARAGRAPH.LEFT, bold=True)
        for cell in heading.cells[1:]: add_cell_text(cell, "", WD_ALIGN_PARAGRAPH.LEFT)
        for cell in heading.cells: set_cell_border(cell, top={"val": "single", "sz": "5", "color": "808080"})
        for item in by_spec[spec]:
            row = table.add_row(); overall = item["is_overall"]
            values = ["", item["cohort"], str(item["participants"]) if isinstance(item["participants"], str) else f"{int(item['participants']):,}",
                      "—" if item["fi_observations"] in (None, "", "nan") or (isinstance(item["fi_observations"], float) and math.isnan(item["fi_observations"])) else f"{int(float(item['fi_observations'])):,}",
                      fmt(item["estimate"]), f"{fmt(item['ci_lower'])} to {fmt(item['ci_upper'])}"]
            for i, (cell, text) in enumerate(zip(row.cells, values)):
                add_cell_text(cell, text, WD_ALIGN_PARAGRAPH.LEFT if i < 2 else WD_ALIGN_PARAGRAPH.RIGHT, bold=overall)
            last_data_row = row
    if last_data_row is not None:
        for cell in last_data_row.cells: set_cell_border(cell, bottom=line)
    p = doc.add_paragraph(); p.paragraph_format.space_before = Pt(2); p.paragraph_format.space_after = Pt(0); p.paragraph_format.line_spacing = Pt(8.8)
    set_run(p.add_run("Notes: "), 8, italic=True); set_run(p.add_run(NOTES), 8)
    doc.save(path)


def tex_escape(text: str) -> str:
    return text.replace("%", "\\%").replace("≥", "$\\geq$").replace("−", "--")


def build_tex(rows: list[dict], path: Path) -> None:
    lines = ["% Generated from frozen Results 3.2 and 3.3 result objects.", "\\begin{landscape}", "\\begin{table}[p]", "\\caption{Primary and sensitivity estimates for wealth recovery and frailty progression.}\\label{tab:s6_sensitivity}", "\\centering", "\\fontsize{8}{9.2}\\selectfont", "\\setlength{\\tabcolsep}{4pt}", "\\begin{tabularx}{\\linewidth}{@{}Y L{3.2cm}rrrL{3.7cm}@{}}", "\\toprule", "Analysis specification & Cohort & Participants, n & FI model observations, n & Adjusted difference in annual FI change, FI units/year & 95\\% CI \\\\", "\\midrule"]
    for spec, _ in BLOCKS:
        lines.append(f"\\multicolumn{{6}}{{@{{}}l}}{{\\textbf{{{tex_escape(spec)}}}}} \\\\")
        for item in [r for r in rows if r["analysis_spec"] == spec]:
            participants = item["participants"] if isinstance(item["participants"], str) else f"{int(item['participants']):,}"
            obs = "--" if item["fi_observations"] is None or (isinstance(item["fi_observations"], float) and math.isnan(item["fi_observations"])) else f"{int(float(item['fi_observations'])):,}"
            cohort = f"\\textbf{{{item['cohort']}}}" if item["is_overall"] else item["cohort"]
            lines.append(f" & {cohort} & {participants} & {obs} & {tex_num(item['estimate'])} & {tex_num(item['ci_lower'])} to {tex_num(item['ci_upper'])} \\\\")
        lines.append("\\addlinespace[2pt]")
    lines.extend(["\\bottomrule", "\\end{tabularx}", "\\vspace{4pt}", f"\\parbox{{\\linewidth}}{{\\raggedright\\fontsize{{8}}{{9.2}}\\selectfont\\textit{{Notes:}} {tex_escape(NOTES)}}}", "\\end{table}", "\\end{landscape}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def draft(rows: list[dict]) -> str:
    def find(spec, cohort): return next(r for r in rows if r["analysis_spec"] == spec and r["cohort"] == cohort)
    def effect(r): return f"{fmt(r['estimate'])} (95% CI {fmt(r['ci_lower'])} to {fmt(r['ci_upper'])})"
    p25 = find("Recovery threshold: 25%", "Overall (random-effects)"); p100 = find("Recovery threshold: 100% (return to pre-shock wealth)", "Overall (random-effects)")
    common = find("Common-22 frailty index", "Overall (random-effects)"); mi = find("Multiple-imputation analysis", "Overall (random-effects)")
    weighted = [find("Landmark-weighted analysis", c) for c in ("HRS", "ELSA", "KLoSA")]
    return (
        "## 3.3 Robustness of the Main Findings\n\n"
        f"Changing the recovery definition produced an overall adjusted difference in annual FI change of {effect(p25)} at the 25% threshold and {effect(p100)} when recovery required return to pre-shock wealth (Supplementary Table S6).\n\n"
        f"Using the Common-22 frailty index yielded an overall estimate of {effect(common)}. In landmark-weighted analyses, the cohort-specific estimates were {effect(weighted[0])} in HRS, {effect(weighted[1])} in ELSA, and {effect(weighted[2])} in KLoSA.\n\n"
        f"After multiple imputation of missing adjustment data, the overall estimate was {effect(mi)}. Across these specifications, estimates remained small, while their direction and precision continued to vary across cohorts and sensitivity definitions.\n"
    )


def build(output_dir: Path) -> None:
    if output_dir.exists(): raise RuntimeError("Table S6输出已存在，拒绝覆盖")
    rows, inputs = source_rows(); output_dir.mkdir(parents=True)
    fields = ["analysis_family", "analysis_spec", "display_order", "cohort", "is_overall", "participants", "fi_observations", "estimate", "se", "ci_lower", "ci_upper", "estimand", "outcome", "recovery_threshold", "weighted", "imputed", "source_result_object", "source_freeze_hash"]
    with (output_dir / "table_s6_summary_long.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
    display = {"caption": CAPTION, "notes": NOTES, "columns": ["Analysis specification", "Cohort", "Participants, n", "FI model observations, n", "Adjusted difference in annual FI change, FI units/year", "95% CI"],
               "block_order": [x[0] for x in BLOCKS], "estimate_decimals": 4, "ci_format": "lower to upper", "layout": "A4_landscape_Times_New_Roman_8.5pt_no_vertical_rules"}
    (output_dir / "table_s6_display_spec.json").write_text(json.dumps(display, ensure_ascii=False, indent=2), encoding="utf-8")
    build_docx(rows, output_dir / "table_s6_editable.docx"); build_tex(rows, output_dir / "table_s6.tex")
    (SECTION33 / "results_3_3_draft.md").write_text(draft(rows), encoding="utf-8")
    validation = {"status": "PASS", "rows": len(rows), "blocks": len(BLOCKS), "overall_rows": sum(r["is_overall"] for r in rows),
                  "weighted_overall_absent": not any(r["analysis_family"] == "weighting" and r["is_overall"] for r in rows),
                  "all_sources_frozen": True, "table_does_not_fit_models": True, "native_editable_docx": True}
    (output_dir / "table_s6_validation.json").write_text(json.dumps(validation, ensure_ascii=False, indent=2), encoding="utf-8")
    files = [p for p in output_dir.iterdir() if p.is_file()]
    manifest = {"release_version": RELEASE_VERSION, "supersedes": "table_s6_v1", "revision_scope": "display_and_self_explanation_only", "full_precision_results_changed": False, "scientific_conclusion_changed": False, "generated_utc": datetime.now(UTC).isoformat(), "inputs": inputs,
                "outputs": {p.name: sha(p) for p in files}, "runtime": {"python": platform.python_version(), "python_docx": "bundled"}, "submission_ready": False}
    (output_dir / "table_s6_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def check(output_dir: Path) -> None:
    rows, _ = source_rows(); summary = read_csv(output_dir / "table_s6_summary_long.csv")
    if len(rows) != len(summary) or len(rows) != 23: raise RuntimeError("Table S6应有23个数据行")
    manifest = json.loads((output_dir / "table_s6_manifest.json").read_text(encoding="utf-8"))
    for name, digest in manifest["outputs"].items():
        if sha(output_dir / name) != digest: raise RuntimeError(f"Table S6输出漂移：{name}")
    doc = Document(output_dir / "table_s6_editable.docx")
    if len(doc.tables) != 1 or len(doc.tables[0].columns) != 6: raise RuntimeError("DOCX不是单个6列原生表格")
    text = "\n".join(p.text for p in doc.paragraphs) + "\n" + "\n".join(c.text for row in doc.tables[0].rows for c in row.cells)
    for forbidden in ("P value", "SMD", "***"):
        if forbidden in text: raise RuntimeError(f"DOCX含禁用内容：{forbidden}")
    freeze = output_dir / "table_s6_freeze.json"
    if freeze.is_file():
        obj = json.loads(freeze.read_text(encoding="utf-8"))
        if obj.get("manifest_sha256") != sha(output_dir / "table_s6_manifest.json"): raise RuntimeError("Table S6 freeze漂移")


def freeze(output_dir: Path) -> None:
    check(output_dir); target = output_dir / "table_s6_freeze.json"
    if target.exists(): raise RuntimeError("Table S6已冻结")
    target.write_text(json.dumps({"freeze_status": "FROZEN", "freeze_version": RELEASE_VERSION, "supersedes": "table_s6_v1", "revision_scope": "display_and_self_explanation_only", "full_precision_results_changed": False, "scientific_conclusion_changed": False, "frozen_utc": datetime.now(UTC).isoformat(), "manifest_sha256": sha(output_dir / "table_s6_manifest.json")}, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser(); p.add_argument("command", choices=("build", "check", "freeze")); p.add_argument("--output-dir", default=str(OUTPUT)); a = p.parse_args()
    {"build": build, "check": check, "freeze": freeze}[a.command](Path(a.output_dir).resolve())


if __name__ == "__main__": main()
