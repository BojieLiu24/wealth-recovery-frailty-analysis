"""Stage2持久化与基础结果核验；不实现Stage3完整QA或Release功能。"""
from __future__ import annotations

import numpy as np
import pandas as pd

from common import read_json, sha, require
from contrasts import evaluate_all

PARAMETER_FIELDS = {"fixed_effect_names", "fixed_effects", "fixed_effect_covariance", "parameter_names", "parameters",
                    "random_covariance_scaled", "random_covariance_unscaled", "residual_variance", "hessian"}


def validate_unit(folder, vectors):
    result = read_json(folder / "result_object.json")
    require(result["status"] in {"PASS", "HOLD_NOT_ESTIMABLE", "BOUNDARY_REVIEW"}, "未知基础QA状态")
    require(not result["release_eligible"] and not result["stage3_complete"], "Stage2错误宣称已冻结或完整QA")
    require((folder / "diagnostics.json").is_file() and (folder / "optimization_log.json").is_file(), "基础证据不完整")
    if result["status"] != "HOLD_NOT_ESTIMABLE":
        with np.load(folder / "model_parameters.npz", allow_pickle=False) as parameters:
            require(set(parameters.files) == PARAMETER_FIELDS, "参数专用文件含未知字段")
            names = parameters["fixed_effect_names"].tolist()
            beta, covariance = parameters["fixed_effects"], parameters["fixed_effect_covariance"]
        table = pd.read_csv(folder / "fixed_effects.csv", encoding="utf-8", float_precision="round_trip")
        require(names == table.term.tolist() == result["fixed_effect_names"], "参数标签不一致")
        np.testing.assert_array_equal(beta, table.estimate.to_numpy(float))
        np.testing.assert_array_equal(covariance, np.asarray(result["fixed_effect_covariance"]))
        expected = evaluate_all(beta, covariance, vectors, result["cohort"], result["spec_id"])
        actual = result["contrasts"]
        csv_contrasts = pd.read_csv(folder / "contrasts.csv", encoding="utf-8", float_precision="round_trip")
        require(len(actual) == len(expected), "对比行数不一致")
        require(csv_contrasts.estimand_id.tolist() == [r["estimand_id"] for r in expected], "对比CSV身份不一致")
        for index, (a, b) in enumerate(zip(actual, expected)):
            require((a["estimand_id"], a["time_years"]) == (b["estimand_id"], b["time_years"]), "对比身份不一致")
            for key in ["estimate", "se", "ci_lower", "ci_upper"]:
                np.testing.assert_allclose(a[key], b[key], rtol=1e-12, atol=1e-14)
                np.testing.assert_array_equal(csv_contrasts.iloc[index][key], a[key])
        require(csv_contrasts.status.eq(result["status"]).all(), "CSV基础QA状态不一致")
    else:
        require(bool(result.get("failure_reasons")), "失败状态没有原因")
    return {"status": "PASS_FILE_INTEGRITY", "model_qa_status": result["status"],
            "scientific_approval": "PENDING", "outputs": {p.name: sha(p) for p in sorted(folder.iterdir()) if p.is_file()}}
