"""从冻结的资格与分析面板证据生成 Figure S1。"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
import textwrap
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
FIGURE_DIR = Path(__file__).resolve().parent
FREEZE_PATH = (
    ROOT
    / "00_admin"
    / "panel_manifests"
    / "analysis_cohort_ledger"
    / "analysis_cohort_ledger_v1_freeze.json"
)
FINAL_ELIGIBILITY_FREEZE_PATH = (
    ROOT
    / "00_admin"
    / "panel_manifests"
    / "final_eligibility"
    / "final_eligibility_v1_freeze.json"
)
ANALYSIS_PANEL_FREEZE_PATH = (
    ROOT
    / "00_admin"
    / "panel_manifests"
    / "analysis_panel"
    / "analysis_panel_v1_freeze.json"
)
OUTPUT_PDF = FIGURE_DIR / "figure_s1_author_review.pdf"
OUTPUT_NOTE = FIGURE_DIR / "figure_s1_caption_and_review.md"
COHORTS = ("HRS", "ELSA", "KLoSA")

EXPECTED_COUNTS = {
    "HRS": {
        "stages": [8204, 6547, 6335, 5222, 5114],
        "exclusions": [1657, 212, 1113, 108],
        "adjustment_parts": {"fi_t": 72, "other": 36},
        "groups": [1703, 3411],
    },
    "ELSA": {
        "stages": [1284, 968, 968, 772, 699],
        "exclusions": [316, 0, 196, 73],
        "adjustment_parts": {"fi_t": 0, "other": 73},
        "groups": [271, 428],
    },
    "KLoSA": {
        "stages": [3125, 2755, 2755, 2507, 2502],
        "exclusions": [370, 0, 248, 5],
        "adjustment_parts": {"fi_t": 0, "other": 5},
        "groups": [943, 1559],
    },
}

CAPTION_TITLE = "Selection of the analytic samples in HRS, ELSA, and KLoSA."
CAPTION_BODY = (
    "Counts refer to unique participants. Starting from the first qualifying major "
    "wealth shock, participants passed sequential gates for recovery status, "
    "landmark frailty, subsequent frailty follow-up, and complete adjustment data. "
    "Later shocks were not substituted. "
    "FI, frailty index; HRS, Health and Retirement Study; ELSA, English Longitudinal "
    "Study of Ageing; KLoSA, Korean Longitudinal Study of Aging."
)


def sha256_file(path: Path) -> str:
    """流式计算文件 SHA-256。"""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def repo_path(value: str) -> Path:
    """解析 freeze 中的仓库相对路径并阻止越界。"""
    path = (ROOT / value).resolve()
    try:
        path.relative_to(ROOT.resolve())
    except ValueError as error:
        raise RuntimeError(f"Freeze引用路径越出仓库：{value}") from error
    return path


def verify_record(record: dict[str, Any], *, label: str) -> Path:
    """验证带 path/sha256 的冻结记录。"""
    path = repo_path(str(record["path"]))
    if not path.is_file():
        raise RuntimeError(f"缺少{label}：{path}")
    if sha256_file(path) != record["sha256"]:
        raise RuntimeError(f"{label}哈希漂移：{path}")
    return path


def verify_frozen_sources() -> tuple[dict[str, Any], dict[str, Any], Path]:
    """只核对冻结引用，不读取任何人员级数据内容。"""
    if not FREEZE_PATH.is_file():
        raise RuntimeError("Figure S1只接受正式冻结的Analysis Cohort Ledger")
    freeze = json.loads(FREEZE_PATH.read_text(encoding="utf-8"))
    if freeze.get("freeze_status") != "FROZEN":
        raise RuntimeError("Ledger freeze_status不是FROZEN")
    if freeze.get("support_ledger_status") != "SUPPORT_FACTS_ONLY":
        raise RuntimeError("Ledger freeze不是support-facts freeze")
    if freeze.get("final_model_eligibility_status") != "NOT_LOCKED":
        raise RuntimeError("Ledger历史状态不符合冻结记录")

    candidate_path = verify_record(
        freeze["source_release_candidate"], label="Ledger release candidate"
    )
    release = json.loads(candidate_path.read_text(encoding="utf-8"))
    if release.get("cohorts") != freeze.get("cohorts"):
        raise RuntimeError("Ledger freeze与release candidate的队列记录不一致")
    if release.get("review_summary") != freeze.get("review_summary"):
        raise RuntimeError("Ledger freeze与release candidate的review记录不一致")
    if release.get("data_dictionary") != freeze.get("data_dictionary"):
        raise RuntimeError("Ledger freeze与release candidate的数据字典记录不一致")

    review_path = verify_record(freeze["review_summary"], label="Ledger review CSV")
    verify_record(freeze["data_dictionary"], label="Ledger data dictionary")
    for cohort in COHORTS:
        record = freeze["cohorts"][cohort]
        parquet_path = repo_path(record["parquet_path"])
        manifest_path = repo_path(record["manifest_path"])
        if sha256_file(parquet_path) != record["parquet_sha256"]:
            raise RuntimeError(f"{cohort}: frozen Ledger Parquet哈希漂移")
        if sha256_file(manifest_path) != record["manifest_sha256"]:
            raise RuntimeError(f"{cohort}: frozen Ledger manifest哈希漂移")

    for panel_name, panel in freeze["upstream_frozen_snapshot"].items():
        upstream_freeze = repo_path(panel["freeze_path"])
        if sha256_file(upstream_freeze) != panel["freeze_sha256"]:
            raise RuntimeError(f"{panel_name}: 上游freeze哈希漂移")
        for cohort in COHORTS:
            record = panel["cohorts"][cohort]
            data_path = repo_path(record["data_path"])
            manifest_path = repo_path(record["manifest_path"])
            if sha256_file(data_path) != record["data_sha256"]:
                raise RuntimeError(f"{panel_name}/{cohort}: 上游数据哈希漂移")
            if sha256_file(manifest_path) != record["manifest_sha256"]:
                raise RuntimeError(f"{panel_name}/{cohort}: 上游manifest哈希漂移")
    return freeze, release, review_path


def read_review_rows(path: Path) -> list[dict[str, str]]:
    """读取现有无微观数据汇总，并拒绝人员标识列。"""
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError("Ledger review CSV为空")
    if "person_id" in rows[0]:
        raise RuntimeError("Figure S1汇总来源不得包含person_id")
    return rows


def one_count(
    rows: list[dict[str, str]],
    *,
    section: str,
    cohort: str,
    stage_or_reason: str,
    recovery_status: str = "ALL",
) -> int:
    """从汇总中提取唯一计数。"""
    matches = [
        row
        for row in rows
        if row["section"] == section
        and row["cohort"] == cohort
        and row["stage_or_reason"] == stage_or_reason
        and row["recovery_status"] == recovery_status
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"汇总计数不唯一：{section}/{cohort}/{stage_or_reason}/{recovery_status}"
        )
    return int(matches[0]["n"])


def derive_figure_counts(
    freeze: dict[str, Any], rows: list[dict[str, str]]
) -> dict[str, dict[str, Any]]:
    """只从冻结汇总推导逐步留存人数，并执行守恒检查。"""
    result: dict[str, dict[str, Any]] = {}
    for cohort in COHORTS:
        record = freeze["cohorts"][cohort]
        manifest_path = repo_path(record["manifest_path"])
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

        index = one_count(
            rows,
            section="nested_flow",
            cohort=cohort,
            stage_or_reason="shock_experienced_cohort",
        )
        exclusions = {
            reason: one_count(
                rows,
                section="first_failed_gate",
                cohort=cohort,
                stage_or_reason=reason,
            )
            for reason in (
                "recovery_unavailable",
                "landmark_fi_missing",
                "no_later_fi",
                "fi_t_missing",
                "prespecified_covariate_missing",
                "none",
            )
        }
        if sum(exclusions.values()) != index:
            raise RuntimeError(f"{cohort}: first-failed gate人数不守恒")

        recovery_available = index - exclusions["recovery_unavailable"]
        landmark_fi = recovery_available - exclusions["landmark_fi_missing"]
        subsequent_fi = landmark_fi - exclusions["no_later_fi"]
        missing_adjustment = (
            exclusions["fi_t_missing"]
            + exclusions["prespecified_covariate_missing"]
        )
        complete_case = subsequent_fi - missing_adjustment

        flow = record["flow"]
        if flow != manifest["flow"]:
            raise RuntimeError(f"{cohort}: freeze与cohort manifest的flow不一致")
        if exclusions != record["first_failed_gate"]:
            raise RuntimeError(f"{cohort}: review与freeze的gate人数不一致")
        if exclusions != manifest["first_failed_gate"]:
            raise RuntimeError(f"{cohort}: review与cohort manifest的gate人数不一致")
        if subsequent_fi != flow["longitudinal_support_complete"]:
            raise RuntimeError(f"{cohort}: longitudinal support人数不守恒")
        if complete_case != flow["complete_case_support_complete"]:
            raise RuntimeError(f"{cohort}: complete-case人数不守恒")
        if complete_case != exclusions["none"]:
            raise RuntimeError(f"{cohort}: pass人数与complete-case人数不一致")

        substantial = one_count(
            rows,
            section="recovery_group_counts",
            cohort=cohort,
            stage_or_reason="complete_case_support_complete",
            recovery_status="substantial_recovery",
        )
        persistent = one_count(
            rows,
            section="recovery_group_counts",
            cohort=cohort,
            stage_or_reason="complete_case_support_complete",
            recovery_status="persistent_or_limited_recovery",
        )
        manifest_groups = manifest["recovery_group_counts"][
            "complete_case_support_complete"
        ]
        if substantial != manifest_groups["substantial_recovery"]:
            raise RuntimeError(f"{cohort}: substantial recovery人数不一致")
        if persistent != manifest_groups["persistent_or_limited_recovery"]:
            raise RuntimeError(f"{cohort}: comparator人数不一致")
        if substantial + persistent != complete_case:
            raise RuntimeError(f"{cohort}: 两个恢复组之和不等于complete-case人数")

        result[cohort] = {
            "stages": [
                index,
                recovery_available,
                landmark_fi,
                subsequent_fi,
                complete_case,
            ],
            "exclusions": [
                exclusions["recovery_unavailable"],
                exclusions["landmark_fi_missing"],
                exclusions["no_later_fi"],
                missing_adjustment,
            ],
            "adjustment_parts": {
                "fi_t": exclusions["fi_t_missing"],
                "other": exclusions["prespecified_covariate_missing"],
            },
            "groups": [substantial, persistent],
        }
    return result


def verify_current_final_status(counts: dict[str, dict[str, Any]]) -> None:
    """用冻结的Final Eligibility与Analysis Panel确认最终人数和行数。"""
    if not FINAL_ELIGIBILITY_FREEZE_PATH.is_file():
        raise RuntimeError("缺少冻结的Final Eligibility v1")
    if not ANALYSIS_PANEL_FREEZE_PATH.is_file():
        raise RuntimeError("缺少冻结的Analysis Panel v1")

    final_freeze = json.loads(
        FINAL_ELIGIBILITY_FREEZE_PATH.read_text(encoding="utf-8")
    )
    analysis_freeze = json.loads(
        ANALYSIS_PANEL_FREEZE_PATH.read_text(encoding="utf-8")
    )
    if final_freeze.get("freeze_status") != "FROZEN":
        raise RuntimeError("Final Eligibility v1不是FROZEN")
    if final_freeze.get("eligibility_status") != "PARTICIPANT_LEVEL_ELIGIBILITY_LOCKED":
        raise RuntimeError("Final Eligibility v1未锁定participant-level eligibility")
    if analysis_freeze.get("freeze_status") != "FROZEN":
        raise RuntimeError("Analysis Panel v1不是FROZEN")
    if analysis_freeze.get("analysis_panel_status") != "MODEL_READY_LONGITUDINAL_PANEL":
        raise RuntimeError("Analysis Panel v1不是MODEL_READY_LONGITUDINAL_PANEL")

    verify_record(final_freeze["source_release_candidate"], label="Final Eligibility release candidate")
    verify_record(final_freeze["review_summary"], label="Final Eligibility review CSV")
    verify_record(final_freeze["data_dictionary"], label="Final Eligibility data dictionary")
    verify_record(analysis_freeze["source_release_candidate"], label="Analysis Panel release candidate")
    verify_record(analysis_freeze["review_summary"], label="Analysis Panel review CSV")
    verify_record(analysis_freeze["data_dictionary"], label="Analysis Panel data dictionary")

    for cohort in COHORTS:
        final_record = final_freeze["cohorts"][cohort]
        analysis_record = analysis_freeze["cohorts"][cohort]
        for label, record in (
            ("Final Eligibility", final_record),
            ("Analysis Panel", analysis_record),
        ):
            data_path = repo_path(record["parquet_path"])
            manifest_path = repo_path(record["manifest_path"])
            if sha256_file(data_path) != record["parquet_sha256"]:
                raise RuntimeError(f"{cohort}: {label} Parquet哈希漂移")
            if sha256_file(manifest_path) != record["manifest_sha256"]:
                raise RuntimeError(f"{cohort}: {label} manifest哈希漂移")

        expected = counts[cohort]
        final_metrics = final_record["metrics"]
        analysis_metrics = analysis_record["metrics"]
        failed = final_record["final_failed_gate"]
        if final_metrics["rows"] != expected["stages"][0]:
            raise RuntimeError(f"{cohort}: Final Eligibility起始人数不一致")
        if final_metrics["base_participants"] != expected["stages"][3]:
            raise RuntimeError(f"{cohort}: Final Eligibility base人数不一致")
        if final_metrics["primary_participants"] != expected["stages"][4]:
            raise RuntimeError(f"{cohort}: Final Eligibility primary人数不一致")
        if [failed["recovery_unavailable"], failed["landmark_fi_missing"], failed["no_later_fi"]] != expected["exclusions"][:3]:
            raise RuntimeError(f"{cohort}: Final Eligibility前三个gate人数不一致")
        if failed["fi_t_missing"] != expected["adjustment_parts"]["fi_t"]:
            raise RuntimeError(f"{cohort}: Final Eligibility shock FI缺失人数不一致")
        if failed["prespecified_covariate_missing"] != expected["adjustment_parts"]["other"]:
            raise RuntimeError(f"{cohort}: Final Eligibility其他协变量缺失人数不一致")
        if [final_metrics["substantial_recovery"], final_metrics["persistent_or_limited_recovery"]] != expected["groups"]:
            raise RuntimeError(f"{cohort}: Final Eligibility恢复分组人数不一致")
        for field in (
            "shock_experienced_participants",
            "base_participants",
            "primary_participants",
            "primary_fi_rows",
            "substantial_recovery",
            "persistent_or_limited_recovery",
        ):
            final_field = "rows" if field == "shock_experienced_participants" else field
            if analysis_metrics[field] != final_metrics[final_field]:
                raise RuntimeError(f"{cohort}: Analysis Panel与Final Eligibility的{field}不一致")


def latex_count(value: int) -> str:
    """将参与者人数格式化为LaTeX可读的千位分隔文本。"""
    return f"{value:,}"


def node_text(label: str, value: int) -> str:
    """生成主路径或分组框中的标签。"""
    return f"{label}\\\\[1pt]\\textbf{{\\textit{{n}} = {latex_count(value)}}}"


def exclusion_text(label: str, value: int) -> str:
    """生成侧边排除框标签。"""
    return f"{label}\\\\[1pt]\\textbf{{\\textit{{n}} = {latex_count(value)}}}"


def build_tikz(counts: dict[str, dict[str, Any]]) -> str:
    """构建三列平行流程的TikZ代码。"""
    stage_labels = (
        "First qualifying major\\\\wealth shock",
        "Valid recovery status at\\\\landmark ($t+1$)",
        "Valid FI at\\\\landmark ($t+1$)",
        "≥1 subsequent valid FI\\\\assessment ($t+2$ onward)",
        "Primary complete-case\\\\analytic sample",
    )
    exclusion_labels = (
        "Recovery status\\\\unavailable",
        "FI unavailable\\\\at landmark",
        "No subsequent valid FI\\\\assessment",
    )
    y_values = (12.9, 10.55, 8.2, 5.85, 3.5)
    lines = [
        r"\path[use as bounding box] (0,0) rectangle (27,14.5);",
        r"\draw[separator] (9,0.1) -- (9,14.1);",
        r"\draw[separator] (18,0.1) -- (18,14.1);",
    ]
    for index, cohort in enumerate(COHORTS):
        panel_start = index * 9.0
        main_x = panel_start + 3.0
        exclusion_x = panel_start + 7.15
        header_x = panel_start + 0.5
        key = cohort.lower()
        values = counts[cohort]
        lines.append(
            rf"\node[cohorthead,anchor=west] at ({header_x:.2f},14.05) "
            rf"{{{chr(65 + index)}. {cohort}}};"
        )
        for stage_index, (label, value, y_value) in enumerate(
            zip(stage_labels, values["stages"], y_values, strict=True)
        ):
            lines.append(
                rf"\node[mainbox] ({key}_stage{stage_index}) at "
                rf"({main_x:.2f},{y_value:.2f}) "
                rf"{{{node_text(label, value)}}};"
            )
        for stage_index in range(4):
            lines.append(
                rf"\draw[mainarrow] ({key}_stage{stage_index}.south) -- "
                rf"({key}_stage{stage_index + 1}.north);"
            )

        for gate_index, (label, value) in enumerate(
            zip(exclusion_labels, values["exclusions"][:3], strict=True)
        ):
            if value <= 0:
                continue
            gate_y = (y_values[gate_index] + y_values[gate_index + 1]) / 2
            lines.extend(
                [
                    rf"\node[exclusionbox] ({key}_exclude{gate_index}) at "
                    rf"({exclusion_x:.2f},{gate_y:.3f}) "
                    rf"{{{exclusion_text(label, value)}}};",
                    rf"\draw[excludearrow] ($({key}_stage{gate_index}.south)!0.5!"
                    rf"({key}_stage{gate_index + 1}.north)$) -- "
                    rf"({key}_exclude{gate_index}.west);",
                ]
            )

        adjustment_total = values["exclusions"][3]
        if adjustment_total > 0:
            adjustment = values["adjustment_parts"]
            adjustment_text = (
                "Missing adjustment data\\\\[1pt]"
                rf"\textbf{{\textit{{n}} = {latex_count(adjustment_total)}}}"
                "\\\\[1pt]"
                "FI at the shock\\\\interview: "
                f"\\textit{{n}} = {latex_count(adjustment['fi_t'])}\\\\"
                "Other prespecified\\\\covariates: "
                f"\\textit{{n}} = {latex_count(adjustment['other'])}"
            )
            gate_y = (y_values[3] + y_values[4]) / 2
            lines.extend(
                [
                    rf"\node[adjustmentbox] ({key}_exclude3) at "
                    rf"({exclusion_x:.2f},{gate_y:.3f}) {{{adjustment_text}}};",
                    rf"\draw[excludearrow] ($({key}_stage3.south)!0.5!"
                    rf"({key}_stage4.north)$) -- ({key}_exclude3.west);",
                ]
            )

        substantial, persistent = values["groups"]
        substantial_text = node_text(
            r"Substantial wealth\\recovery", substantial
        )
        persistent_text = node_text(
            r"Persistent loss or\\limited recovery", persistent
        )
        lines.extend(
            [
                rf"\node[recoverybox] ({key}_substantial) at "
                rf"({main_x - 1.38:.2f},1.05) "
                rf"{{{substantial_text}}};",
                rf"\node[comparatorbox] ({key}_persistent) at "
                rf"({main_x + 1.38:.2f},1.05) "
                rf"{{{persistent_text}}};",
                rf"\draw[mainarrow] ({key}_stage4.south) -- ++(0,-0.35) -| "
                rf"({key}_substantial.north);",
                rf"\draw[mainarrow] ({key}_stage4.south) -- ++(0,-0.35) -| "
                rf"({key}_persistent.north);",
            ]
        )
    return "\n".join(lines)


def build_latex(counts: dict[str, dict[str, Any]]) -> str:
    """生成独立作者复核页，不修改正式Supplementary Materials入口。"""
    tikz = build_tikz(counts)
    return textwrap.dedent(
        rf"""
        \documentclass[10pt,a4paper]{{article}}
        \usepackage[landscape,margin=1.0cm]{{geometry}}
        \usepackage{{fontspec}}
        \setmainfont{{Times New Roman}}
        \usepackage{{tikz}}
        \usetikzlibrary{{arrows.meta,calc}}
        \usepackage{{caption}}
        \captionsetup[figure]{{
          name=Figure,
          font=small,
          labelfont=bf,
          labelsep=period,
          justification=raggedright,
          singlelinecheck=false,
          skip=5pt
        }}
        \renewcommand{{\thefigure}}{{S1}}
        \pagestyle{{empty}}
        \setlength{{\parindent}}{{0pt}}
        \definecolor{{mainstroke}}{{RGB}}{{31,78,121}}
        \definecolor{{mainfill}}{{RGB}}{{244,249,252}}
        \definecolor{{excludegray}}{{RGB}}{{105,105,105}}
        \definecolor{{excludefill}}{{RGB}}{{248,248,248}}
        \definecolor{{tealstroke}}{{RGB}}{{0,130,130}}
        \definecolor{{tealfill}}{{RGB}}{{239,250,249}}
        \definecolor{{orangestroke}}{{RGB}}{{205,91,28}}
        \definecolor{{orangefill}}{{RGB}}{{253,246,240}}
        \begin{{document}}
        \begin{{figure}}[!ht]
        \centering
        \begin{{tikzpicture}}[
          mainbox/.style={{draw=mainstroke,fill=mainfill,rounded corners=2pt,
            line width=0.7pt,minimum width=5.0cm,minimum height=1.18cm,
            align=center,inner sep=3pt,font=\fontsize{{9.5}}{{10.8}}\selectfont}},
          exclusionbox/.style={{draw=excludegray,fill=excludefill,rounded corners=2pt,
            line width=0.55pt,minimum width=2.55cm,minimum height=1.20cm,
            align=center,inner sep=2.5pt,font=\fontsize{{8.5}}{{9.6}}\selectfont}},
          adjustmentbox/.style={{draw=excludegray,fill=excludefill,rounded corners=2pt,
            line width=0.55pt,minimum width=2.65cm,minimum height=2.08cm,
            align=center,inner sep=2.5pt,font=\fontsize{{8.5}}{{9.6}}\selectfont}},
          recoverybox/.style={{draw=tealstroke,fill=tealfill,rounded corners=2pt,
            line width=0.8pt,text width=2.30cm,minimum height=1.55cm,
            align=center,inner sep=2pt,font=\fontsize{{8.5}}{{9.7}}\selectfont}},
          comparatorbox/.style={{draw=orangestroke,fill=orangefill,rounded corners=2pt,
            line width=0.8pt,text width=2.30cm,minimum height=1.55cm,
            align=center,inner sep=2pt,font=\fontsize{{8.5}}{{9.7}}\selectfont}},
          cohorthead/.style={{font=\fontsize{{12}}{{14}}\selectfont\bfseries,
            text=black}},
          mainarrow/.style={{-{{Latex[length=2.5mm,width=1.6mm]}},
            draw=mainstroke,line width=0.75pt}},
          excludearrow/.style={{-{{Latex[length=2.1mm,width=1.35mm]}},
            draw=excludegray,dashed,line width=0.55pt}},
          separator/.style={{draw=black!15,line width=0.5pt}}
        ]
        {tikz}
        \end{{tikzpicture}}
        \caption{{\textbf{{{CAPTION_TITLE}}} {CAPTION_BODY}}}
        \end{{figure}}
        \end{{document}}
        """
    ).strip() + "\n"


def build_note() -> str:
    """生成不含工程哈希的作者复核说明。"""
    return textwrap.dedent(
        f"""
        # Figure S1

        ## Caption

        **Figure S1. {CAPTION_TITLE}** {CAPTION_BODY}

        ## Source and current status

        - All displayed counts are unique-participant aggregates reconciled across the frozen Analysis Cohort Ledger, Final Eligibility v1, and Analysis Panel v1.
        - Final Eligibility v1 locks participant-level primary eligibility, and Analysis Panel v1 confirms the corresponding longitudinal model-row counts.
        - Existing upstream summaries before index-shock identification use wave- or interval-level denominators and therefore were not added to this participant-level figure.
        - Actual interview year and month were used in the frozen timed-FI selector; later shocks were not substituted for the first qualifying index shock.
        - Any future change to eligibility or row counts requires regeneration from a new frozen authoritative release rather than manual PDF editing.
        """
    ).strip() + "\n"


def require_tool(name: str) -> str:
    """确认PDF构建和检查工具存在。"""
    path = shutil.which(name)
    if not path:
        raise RuntimeError(f"缺少必需工具：{name}")
    return path


def run_checked(command: list[str], *, cwd: Path) -> subprocess.CompletedProcess[str]:
    """运行外部工具并保留UTF-8日志。"""
    result = subprocess.run(
        command,
        cwd=cwd,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"命令失败：{' '.join(command)}\n{result.stdout}\n{result.stderr}"
        )
    return result


def validate_pdf(pdf_path: Path, counts: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """检查单页横向、可提取文本、字体和矢量输出。"""
    pdfinfo = run_checked([require_tool("pdfinfo"), str(pdf_path)], cwd=ROOT).stdout
    pages = re.search(r"^Pages:\s+(\d+)$", pdfinfo, flags=re.MULTILINE)
    size = re.search(
        r"^Page size:\s+([0-9.]+) x ([0-9.]+) pts", pdfinfo, flags=re.MULTILINE
    )
    if pages is None or int(pages.group(1)) != 1:
        raise RuntimeError("Figure S1 PDF必须恰好一页")
    if size is None or float(size.group(1)) <= float(size.group(2)):
        raise RuntimeError("Figure S1 PDF必须为landscape")

    with tempfile.TemporaryDirectory(prefix="figure_s1_text_", dir=ROOT / "tmp") as name:
        text_path = Path(name) / "figure_s1.txt"
        run_checked(
            [require_tool("pdftotext"), "-layout", str(pdf_path), str(text_path)],
            cwd=ROOT,
        )
        extracted = text_path.read_text(encoding="utf-8", errors="replace")
    required_text = [
        "Figure S1",
        "First qualifying major wealth shock",
        "Valid recovery status at landmark",
        "Recovery status unavailable",
        "Valid FI at landmark",
        "FI unavailable at landmark",
        "≥1 subsequent valid FI assessment",
        "No subsequent valid FI assessment",
        "Missing adjustment data",
        "FI at the shock interview",
        "Other prespecified",
        "covariates",
        "Primary complete-case",
        "Substantial wealth",
        "Persistent loss or",
        "Counts refer to unique participants",
        "Later shocks were not substituted",
    ]
    normalised_text = " ".join(extracted.replace(";", ";").split())
    for text in required_text:
        if text not in normalised_text:
            raise RuntimeError(f"Figure S1 PDF缺少文本：{text}")
    expected_caption = " ".join(
        f"Figure S1. {CAPTION_TITLE} {CAPTION_BODY}".split()
    )
    if expected_caption not in normalised_text:
        raise RuntimeError("Figure S1 caption与批准文本不一致")
    for index, cohort in enumerate(COHORTS):
        panel_heading = f"{chr(65 + index)}. {cohort}"
        if panel_heading not in extracted:
            raise RuntimeError(f"Figure S1 PDF缺少队列标题：{panel_heading}")
        for value in [*counts[cohort]["stages"], *counts[cohort]["groups"]]:
            if latex_count(value) not in extracted:
                raise RuntimeError(f"Figure S1 PDF缺少显示人数：{cohort}/{value}")
    if re.search(r"search[ -]window", extracted, flags=re.IGNORECASE):
        raise RuntimeError("Figure S1 PDF不得包含search window术语")

    fonts = run_checked([require_tool("pdffonts"), str(pdf_path)], cwd=ROOT).stdout
    if "TimesNewRoman" not in fonts.replace(" ", ""):
        raise RuntimeError("Figure S1 PDF未检测到Times New Roman字体")
    images = run_checked([require_tool("pdfimages"), "-list", str(pdf_path)], cwd=ROOT)
    image_rows = [line for line in images.stdout.splitlines() if re.match(r"^\s*\d+\s+", line)]
    if image_rows:
        raise RuntimeError("Figure S1包含栅格图像，不是纯矢量/文字资产")
    return {
        "pages": 1,
        "landscape": True,
        "times_new_roman_embedded": True,
        "raster_images": 0,
        "text_checks": "passed",
    }


def main() -> int:
    """验证冻结汇总、编译PDF并原子发布作者复核资产。"""
    freeze, _release, review_path = verify_frozen_sources()
    counts = derive_figure_counts(freeze, read_review_rows(review_path))
    if counts != EXPECTED_COUNTS:
        raise RuntimeError("Figure S1计数与当前批准版本不完全一致")
    verify_current_final_status(counts)
    latex = build_latex(counts)
    note = build_note()
    if "weighted_complete_case" in latex or "Positive landmark weight" in latex:
        raise RuntimeError("Figure S1不得显示加权敏感性支持分支")

    (ROOT / "tmp").mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="figure_s1_build_", dir=ROOT / "tmp") as name:
        build_dir = Path(name)
        tex_path = build_dir / "figure_s1.tex"
        tex_path.write_text(latex, encoding="utf-8")
        latexmk = require_tool("latexmk")
        run_checked(
            [
                latexmk,
                "-xelatex",
                "-interaction=nonstopmode",
                "-halt-on-error",
                "-file-line-error",
                f"-outdir={build_dir}",
                str(tex_path),
            ],
            cwd=build_dir,
        )
        log_text = (build_dir / "figure_s1.log").read_text(
            encoding="utf-8", errors="replace"
        )
        forbidden_log = (
            "Overfull \\hbox",
            "Overfull \\vbox",
            "Undefined control sequence",
            "LaTeX Error:",
            "File `",
        )
        for marker in forbidden_log:
            if marker in log_text:
                raise RuntimeError(f"Figure S1 LaTeX日志出现禁止标记：{marker}")
        staged_pdf = build_dir / "figure_s1.pdf"
        validation = validate_pdf(staged_pdf, counts)
        staged_note = build_dir / OUTPUT_NOTE.name
        staged_note.write_text(note, encoding="utf-8")

        FIGURE_DIR.mkdir(parents=True, exist_ok=True)
        staged_final_pdf = FIGURE_DIR / f".{OUTPUT_PDF.name}.tmp"
        staged_final_note = FIGURE_DIR / f".{OUTPUT_NOTE.name}.tmp"
        shutil.copyfile(staged_pdf, staged_final_pdf)
        shutil.copyfile(staged_note, staged_final_note)
        os.replace(staged_final_pdf, OUTPUT_PDF)
        os.replace(staged_final_note, OUTPUT_NOTE)

    print(
        json.dumps(
            {
                "status": "FIGURE S1 READY",
                "source": "frozen Analysis Cohort Ledger, Final Eligibility v1, and Analysis Panel v1",
                "output_pdf": str(OUTPUT_PDF),
                "caption_and_review": str(OUTPUT_NOTE),
                "counts": counts,
                "pdf_validation": validation,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
