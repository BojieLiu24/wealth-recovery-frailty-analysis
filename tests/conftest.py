"""公开测试只运行合成数据及公开聚合，不读取受限数据。"""
from pathlib import Path
import importlib.util
import sys
ROOT = Path(__file__).resolve().parents[1]
for rel in ['code', '02_code/01_panel_build', '02_code/03_results/section_3_2', '02_code/03_results/section_3_3/model_robustness']:
    sys.path.insert(0, str(ROOT/rel))


def module_from_file(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT/relative)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod
