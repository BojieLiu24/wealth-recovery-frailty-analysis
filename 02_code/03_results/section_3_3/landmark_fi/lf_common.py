"""Round1C独立路径及通用来源接口；旧模块仅只读复用。"""
from pathlib import Path
import sys

CODE = Path(__file__).resolve().parent
ROOT = CODE.parents[3]
RB_CODE = CODE.parent / 'model_robustness'
sys.path.insert(0, str(RB_CODE))
from common import (read_json, write_json, write_csv, sha, rel, require, clean,
                    provenance, check_protected, repo_path, TERM, Z95)

PROGRESS = ROOT / '06_project_progress/round1c_landmark_fi_v1'
CONTRACT = ROOT / '00_admin/model_manifests/landmark_fi/landmark_fi_round1c_v1_contract.json'
RUNS = ROOT / '03_output/3.3 Robustness of the Main Findings/Landmark FI Sensitivity/runs'
COHORTS = ('HRS', 'ELSA', 'KLoSA')
SPECS = ('PO01', 'LM01')
UNITS = [f'{c}/{s}' for c in COHORTS for s in SPECS]


def code_hashes():
    return {rel(p): sha(p) for p in sorted(CODE.rglob('*'))
            if p.is_file() and p.suffix in {'.py', '.md', '.ps1'}}


def assert_hashes(records):
    for name, expected in records.items():
        p = repo_path(name)
        require(p.is_file() and sha(p) == expected, '来源或代码变化：' + name)


def source_code_paths():
    # 只登记本轮实际复用的旧函数模块，不导入历史拟合入口。
    return [RB_CODE / n for n in ('common.py', 'specification.py', 'time_basis.py',
                                 'prepare_inputs.py', 'basic_qa.py')]
