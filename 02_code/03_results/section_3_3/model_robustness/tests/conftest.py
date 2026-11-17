"""本次所有测试禁止调用任何MixedLM优化入口。"""
from pathlib import Path
import sys
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@pytest.fixture(autouse=True)
def forbid_model_fit(monkeypatch):
    import statsmodels.api as sm

    def forbidden(*args, **kwargs):
        raise AssertionError("本轮测试禁止真实或合成模型拟合")

    monkeypatch.setattr(sm.MixedLM, "fit", forbidden)
