"""Round 1B路径、严格读写和来源记录；不导入历史分析runner。"""
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import patsy
import scipy
import statsmodels

CODE = Path(__file__).resolve().parent
ROOT = CODE.parents[3]
PROGRESS = ROOT / "06_project_progress/round1b_model_robustness_v1"
CONTRACT_PATH = ROOT / "00_admin/model_manifests/model_robustness/model_robustness_round1b_v1_contract.json"
PRIMARY = ROOT / "03_output/3.2 Wealth Recovery and Subsequent Frailty Progression"
RUNS = ROOT / "03_output/3.3 Robustness of the Main Findings/Model Robustness/runs"
PANEL_FREEZE = ROOT / "00_admin/panel_manifests/analysis_panel/analysis_panel_v1_freeze.json"
COHORTS = ("HRS", "ELSA", "KLoSA")
SPECS = ("RS01", "NL01")
TERM = "recovery:time_years"
Z95 = 1.959963984540054


def require(condition, message):
    if not bool(condition):
        raise ValueError(message)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rel(path):
    return Path(path).resolve().relative_to(ROOT.resolve()).as_posix()


def repo_path(value, root=ROOT):
    root = Path(root).resolve()
    path = (root / value).resolve()
    require(path.is_relative_to(root), "来源路径越出仓库")
    return path


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_json(path, value, *, replace_owned=False):
    """默认拒绝覆盖；仅当前新建运行目录的进度文件允许显式原子更新。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(clean(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if not replace_owned:
        with path.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(payload)
    else:
        temporary = path.with_name(path.name + ".tmp")
        with temporary.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(payload)
        temporary.replace(path)


def write_csv(path, rows):
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8", mode="x")


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT).decode("utf-8").strip()


def provenance():
    diff = subprocess.check_output(["git", "diff", "--binary"], cwd=ROOT)
    return {
        "generated_utc": datetime.now(UTC).isoformat(),
        "git_commit": git("rev-parse", "HEAD"),
        "git_dirty": bool(git("status", "--porcelain")),
        "git_diff_sha256": hashlib.sha256(diff).hexdigest(),
        "runtime": {"python": platform.python_version(), "numpy": np.__version__,
                    "pandas": pd.__version__, "patsy": patsy.__version__,
                    "scipy": scipy.__version__, "statsmodels": statsmodels.__version__},
    }


def code_hashes():
    return {rel(p): sha(p) for p in sorted(CODE.rglob("*"))
            if p.is_file() and p.suffix in {".py", ".ps1", ".md"}}


def check_protected(snapshot):
    """检查实施前每一项既有资产，不把历史例外错误地当作本轮变动。"""
    changed = []
    for name, expected in snapshot["protected_hashes"].items():
        path = repo_path(name)
        if not path.is_file() or sha(path) != expected:
            changed.append(name)
    require(not changed, "本轮改变了受保护资产：" + ", ".join(changed))
    require(git("rev-parse", "HEAD") == snapshot["git_head"], "Git HEAD发生变化")
    diff = subprocess.check_output(["git", "diff", "--binary"], cwd=ROOT)
    require(hashlib.sha256(diff).hexdigest() == snapshot["git_diff_sha256"], "既有tracked diff发生变化")
    return {"status": "PASS", "files_checked": len(snapshot["protected_hashes"]),
            "historical_dirty_changes_preserved": True}
